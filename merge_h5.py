#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge HDF5 windfield (from your collector), resample per-z onto uniform x–y grid,
zero section edges, and interpolate probe time series. Outputs in float16.

Each case can be processed in parallel (per-process) and finally written into a
single output HDF5. Use ``--workers`` to control the degree of parallelism.

INPUT FORMAT (per case group, e.g. "/0"):
  /<case>/coords           (N, 3) float16   # cell centers (x,y,z), same for all timesteps
  /<case>/time             (T,)   float32   # list of times
  /<case>/<t_str>/velocity (N, 3) float16   # per timestep velocity

OUTPUT:
  /time                                 (T,) if all cases share identical time (and cfg.dt will be written)
  /meta/x_values, /meta/y_values        (Nx_out,), (Ny_out,)
  /meta/plane_grid_json                 json string
  /meta/probe_x_value                   scalar
  /meta/probe_y_values                  (Nprobe,)
  /meta/config_json                     json string

  /cases/case_000/section               (T, Z, Ny, Nx, 3)  float16
  /cases/case_000/z_values_section      (Z,)               float32
  /cases/case_000/probes                (T, Z, Nprobe, 3)  float16

NOTES:
- Section grid: x∈[0,1], step 0.025; y∈[0,1.2], step 0.025; edges (x=0/1 or y=0/1.2) are zeroed.
- Probes: x=0.5; y = [0.027, 0.082, 0.18, 0.333, 0.486, 0.638, 0.791, 0.943, 1.099]; z follows section levels.
"""

import argparse
import glob
import json
import os
from typing import Tuple, Optional, List

import h5py
import numpy as np

# ---------- Configs ----------
# Section grid
X_MIN, X_MAX, X_STEP = 0.0, 1.0, 0.025
Y_MIN, Y_MAX, Y_STEP = 0.0, 1.2, 0.025

# Probes layout (global)
PROBE_X_DEFAULT = 0.5
PROBE_Y_DEFAULT = [0.027, 0.082, 0.18, 0.333, 0.486, 0.638, 0.791, 0.943, 1.099]

# Output dtype
OUT_DTYPE = np.float16     # section/probes on disk
META_DTYPE = np.float32    # meta arrays

# HDF5 compression
COMP = dict(compression="gzip", compression_opts=4, shuffle=True)

# ---------- Grid & helpers ----------
def _grid_xy():
    xs = np.arange(X_MIN, X_MAX + 1e-12, X_STEP, dtype=META_DTYPE)
    ys = np.arange(Y_MIN, Y_MAX + 1e-12, Y_STEP, dtype=META_DTYPE)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, Xg.astype(META_DTYPE), Yg.astype(META_DTYPE)

def _clamp(v: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.minimum(np.maximum(v, vmin), vmax)

def _force_edge_zero(field: np.ndarray) -> None:
    # field shape (..., Ny, Nx, 3)
    field[..., 0, :, :] = 0.0       # y = min
    field[..., -1, :, :] = 0.0      # y = max
    field[..., :, 0, :] = 0.0       # x = min
    field[..., :, -1, :] = 0.0      # x = max

def _unique_z_levels(z: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster z to unique layers by rounding tolerance; return (z_sorted, layer_index_of_each_point)."""
    zr = np.round(z / tol) * tol
    z_unique, inv = np.unique(zr, return_inverse=True)
    order = np.argsort(z_unique)
    z_sorted = z_unique[order]
    remap = {old: i for i, old in enumerate(order)}
    inv_sorted = np.vectorize(remap.get)(inv)
    return z_sorted.astype(META_DTYPE), inv_sorted.astype(np.int64)

# ---------- Interpolation backend (SciPy if available; else IDW) ----------
_SCI_AVAILABLE = False
try:
    from scipy.interpolate import griddata  # type: ignore
    _SCI_AVAILABLE = True
except Exception:
    _SCI_AVAILABLE = False

def _interp_to_grid(xy, val, Xg, Yg, idw_k=16, idw_power=2.0):
    """linear griddata with nearest fallback; fallback to IDW if SciPy missing."""
    if _SCI_AVAILABLE:
        Zi = griddata(xy, val, (Xg, Yg), method="linear")
        nan = np.isnan(Zi)
        if nan.any():
            Zi2 = griddata(xy, val, (Xg, Yg), method="nearest")
            Zi[nan] = Zi2[nan]
        return Zi.astype(np.float32)
    # fallback IDW
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    diffs = pts[:, None, :] - xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    d2[d2 == 0] = 1e-12
    if idw_k < d2.shape[1]:
        idx = np.argpartition(d2, kth=idw_k, axis=1)[:, :idw_k]
        rows = np.arange(d2.shape[0])[:, None]
        d2k = d2[rows, idx]
        wk = 1.0 / (d2k ** (idw_power / 2.0))
        vk = val[idx]
        Zi = np.sum(wk * vk, axis=1) / np.sum(wk, axis=1)
    else:
        w = 1.0 / (d2 ** (idw_power / 2.0))
        Zi = np.sum(w * val[None, :], axis=1) / np.sum(w, axis=1)
    return Zi.reshape(Xg.shape).astype(np.float32)

def _interp_to_points(xy, val, targets_xy, idw_k=16, idw_power=2.0):
    """linear griddata to scattered points; fallback to IDW."""
    if _SCI_AVAILABLE:
        Zi = griddata(xy, val, (targets_xy[:, 0], targets_xy[:, 1]), method="linear")
        nan = np.isnan(Zi)
        if nan.any():
            Zi2 = griddata(xy, val, (targets_xy[:, 0], targets_xy[:, 1]), method="nearest")
            Zi[nan] = Zi2[nan]
        return Zi.astype(np.float32)
    # fallback IDW
    diffs = targets_xy[:, None, :] - xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    d2[d2 == 0] = 1e-12
    if idw_k < d2.shape[1]:
        idx = np.argpartition(d2, kth=idw_k, axis=1)[:, :idw_k]
        rows = np.arange(d2.shape[0])[:, None]
        d2k = d2[rows, idx]
        wk = 1.0 / (d2k ** (idw_power / 2.0))
        vk = val[idx]
        Zi = np.sum(wk * vk, axis=1) / np.sum(wk, axis=1)
    else:
        w = 1.0 / (d2 ** (idw_power / 2.0))
        Zi = np.sum(w * val[None, :], axis=1) / np.sum(w, axis=1)
    return Zi.astype(np.float32)

# ---------- Readers for YOUR schema ----------
def _read_case_schema(g_case: h5py.Group):
    """
    Read your collector's schema:
      coords: (N,3) float16
      time:   (T,)  float32
      per-step subgroup: <t_str>/velocity (N,3) float16
    Return: (coords_f32 (N,3), time_f32 (T,), step_names_sorted (list[str]))
    """
    if "coords" not in g_case or "time" not in g_case:
        raise KeyError(f"Group {g_case.name} missing 'coords' or 'time'.")

    coords = g_case["coords"][...].astype(np.float32, copy=False)  # (N,3) fp32 for compute
    time = g_case["time"][...].astype(np.float32, copy=False)      # (T,)
    # time subgroup names are strings; we sort by float value
    step_names = [k for k, v in g_case.items() if isinstance(v, h5py.Group)]
    # exclude known non-time groups if any (we know coords/time are datasets so ok)
    def _as_float(s):
        try:
            return float(s.lstrip("tT"))
        except Exception:
            # if cannot parse, push to end with large key
            return float("inf")
    step_names.sort(key=_as_float)
    # sanity: number of steps should match len(time) if names are pure times;
    # but we don't hard enforce; we iterate over step_names we found.
    return coords, time, step_names

# ---------- Core: per-case processing (streamed by time) ----------
def process_case(
    fpath: str,
    case_name: str,
    time_chunk: int = 8,
    probe_x: float = PROBE_X_DEFAULT,
    probe_y_list: Optional[List[float]] = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
):
    """Load a single case from ``fpath`` and return interpolated arrays.

    Returns
    -------
    section : np.ndarray
        (T, Z, Ny, Nx, 3) float16 interpolated section field.
    z_vals : np.ndarray
        (Z,) float32 z levels for the section grid.
    probes : np.ndarray
        (T, Z, Nprobe, 3) float16 probe time series.
    time : np.ndarray
        (T,) float32 time array from the case.
    """
    xs, ys, Xg, Yg = _grid_xy()
    Ny, Nx = ys.shape[0], xs.shape[0]

    with h5py.File(fpath, "r") as h5_in:
        g_case_in = h5_in[case_name]
        coords, time, step_names = _read_case_schema(g_case_in)  # coords (N,3), time (T,), step subgroup names

        if probe_y_list is None:
            probe_y_list = PROBE_Y_DEFAULT
        probe_y = np.asarray(probe_y_list, dtype=META_DTYPE)
        probe_y = _clamp(probe_y, Y_MIN, Y_MAX)
        probe_x = float(np.clip(probe_x, X_MIN, X_MAX))
        probe_xy = np.column_stack([np.full_like(probe_y, probe_x), probe_y])
        Nprobe = probe_xy.shape[0]

        # z layers
        z_vals, layer_idx = _unique_z_levels(coords[:, 2], tol=1e-6)  # (Z,), (N,)
        Z = z_vals.shape[0]
        T = len(step_names)

        section_all = np.empty((T, Z, Ny, Nx, 3), dtype=OUT_DTYPE)
        probes_all = np.empty((T, Z, Nprobe, 3), dtype=OUT_DTYPE)

        # pre-split indices and xy by z
        z_to_idx = [np.where(layer_idx == z_id)[0] for z_id in range(Z)]
        z_to_xy = [coords[idx, :2].astype(np.float32, copy=False) for idx in z_to_idx]

        # stream timesteps in chunks to avoid loading all (T,N,3) at once
        for t0 in range(0, T, time_chunk):
            t1 = min(T, t0 + time_chunk)
            tc = t1 - t0
            buf_sec = np.empty((tc, Z, Ny, Nx, 3), dtype=np.float32)
            buf_prb = np.empty((tc, Z, Nprobe, 3), dtype=np.float32)

            # Load velocities for this chunk into per-z blocks (still per-step)
            for tt, step_idx in enumerate(range(t0, t1)):
                step_name = step_names[step_idx]
                if "velocity" not in g_case_in[step_name]:
                    raise KeyError(f"Missing dataset 'velocity' under '{g_case_in.name}/{step_name}'")
                vel = g_case_in[step_name]["velocity"][...]  # (N,3) float16
                vel = vel.astype(np.float32, copy=False)

                # For each z layer, do interpolation for u,v,w
                for z_id in range(Z):
                    xy = z_to_xy[z_id]           # (n_i, 2)
                    idx = z_to_idx[z_id]         # indices in coords
                    vals = vel[idx, :]           # (n_i, 3)

                    # section grid (Ny,Nx,3)
                    for comp in range(3):
                        buf_sec[tt, z_id, :, :, comp] = _interp_to_grid(
                            xy, vals[:, comp], Xg, Yg, idw_k=idw_k, idw_power=idw_power
                        )
                    # probes (Nprobe,3)
                    for comp in range(3):
                        buf_prb[tt, z_id, :, comp] = _interp_to_points(
                            xy, vals[:, comp], probe_xy, idw_k=idw_k, idw_power=idw_power
                        )

            # zero edges on section
            _force_edge_zero(buf_sec)

            section_all[t0:t1, ...] = buf_sec.astype(OUT_DTYPE, copy=False)
            probes_all[t0:t1, ...] = buf_prb.astype(OUT_DTYPE, copy=False)

    return section_all, z_vals, probes_all, time

# ---------- Merge driver ----------
def merge_all(
    input_dir: str,
    output_file: str | None,
    pattern: str = "windfield_all_cases*.h5",
    time_chunk: int = 8,
    probe_x: float = PROBE_X_DEFAULT,
    probe_y_list: Optional[List[float]] = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
    workers: Optional[int] = None,
):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' in {input_dir}")

    if not output_file:
        output_file = os.path.join(input_dir, "dataset_merged_resampled_fp16.h5")
        print(f"[Info] No output path provided. Using default: {output_file}")

    print("Merging & resampling (float16 outputs):")
    for f in files:
        print("  -", f)

    xs, ys, _, _ = _grid_xy()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # gather all (file, case) pairs
    tasks: List[Tuple[str, str]] = []
    for fpath in files:
        with h5py.File(fpath, "r") as h5_in:
            for case_name in h5_in.keys():
                if isinstance(h5_in[case_name], h5py.Group):
                    tasks.append((fpath, case_name))

    workers = os.cpu_count() if workers is None else workers

    with h5py.File(output_file, "w") as h5_out:
        cases_out = h5_out.require_group("cases")
        global_time = None
        global_time_ok = True

        future_map = {}
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, (fpath, case_name) in enumerate(tasks):
                case_id = f"case_{idx:03d}"
                print(f"  > {os.path.basename(fpath)}::{case_name} -> {case_id}")
                fut = ex.submit(
                    process_case,
                    fpath,
                    case_name,
                    time_chunk,
                    probe_x,
                    probe_y_list,
                    idw_k,
                    idw_power,
                )
                future_map[fut] = case_id

            for fut in as_completed(future_map):
                case_id = future_map[fut]
                section, z_vals, probes, time_arr = fut.result()

                g_case_out = cases_out.require_group(case_id)
                g_case_out.create_dataset(
                    "section",
                    data=section,
                    chunks=(min(time_chunk, section.shape[0]), 1, section.shape[2], section.shape[3], 3),
                    **COMP,
                )
                g_case_out.create_dataset("z_values_section", data=z_vals.astype(META_DTYPE))
                g_case_out.create_dataset(
                    "probes",
                    data=probes,
                    chunks=(min(time_chunk, probes.shape[0]), 1, probes.shape[2], 3),
                    **COMP,
                )

                if time_arr is not None and time_arr.size > 0:
                    if global_time is None:
                        global_time = time_arr
                    else:
                        if (global_time.shape != time_arr.shape) or (
                            not np.allclose(global_time, time_arr, rtol=1e-7, atol=1e-10)
                        ):
                            global_time_ok = False

        # write meta & (maybe) global time
        meta = h5_out.require_group("meta")
        meta.create_dataset("x_values", data=np.asarray(xs, dtype=META_DTYPE))
        meta.create_dataset("y_values", data=np.asarray(ys, dtype=META_DTYPE))

        grid_info = {
            "x": {"min": float(X_MIN), "max": float(X_MAX), "step": float(X_STEP), "count": int(len(xs))},
            "y": {"min": float(Y_MIN), "max": float(Y_MAX), "step": float(Y_STEP), "count": int(len(ys))},
            "note": "Section edges (x=0/1 or y=0/1.2) are zeroed after interpolation."
        }
        meta.create_dataset("plane_grid_json", data=np.string_(json.dumps(grid_info)))

        meta.create_dataset("probe_x_value", data=np.asarray(probe_x, dtype=META_DTYPE))
        meta.create_dataset("probe_y_values", data=np.asarray(PROBE_Y_DEFAULT if probe_y_list is None else probe_y_list, dtype=META_DTYPE))

        cfg = {
            "dt": None,
            "layout": {"section": "section(T,Z,Ny,Nx,3)", "probes": "probes(T,Z,Nprobe,3)"},
            "dtype": {"section": "float16", "probes": "float16"},
            "z_crop": {"mode": "random_block", "block": 1, "same_for_XY": True},
            "interpolation": "griddata" if _SCI_AVAILABLE else "idw",
            "chunks": {"section": ["time_chunk", 1, "Ny", "Nx", 3], "probes": ["time_chunk", 1, "Nprobe", 3]},
            "parallel": {"workers": int(workers)},
        }

        # If global time consistent, save once and set dt
        if global_time_ok and global_time is not None:
            h5_out.create_dataset("time", data=global_time.astype(META_DTYPE))
            if len(global_time) >= 2:
                cfg["dt"] = float(np.median(np.diff(global_time)))
            print("Wrote global /time and dt in meta/config_json")

        meta.create_dataset("config_json", data=np.string_(json.dumps(cfg)))

    print(f"Done. Output written to: {output_file}")

# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge & resample windfield (your collector schema) onto uniform x–y grid per z; add probes; float16 outputs."
    )
    p.add_argument("input_dir", help="Directory containing windfield_all_cases*.h5")
    p.add_argument("output_file", nargs="?", default=None, help="Output HDF5 (default: <input_dir>/dataset_merged_resampled_fp16.h5)")
    p.add_argument("--pattern", default="windfield_all_cases*.h5", help="Glob pattern for inputs")
    p.add_argument("--time-chunk", type=int, default=8, help="Streaming chunk over timesteps")
    p.add_argument("--probe-x", type=float, default=PROBE_X_DEFAULT, help="Probe x (default 0.5)")
    p.add_argument("--probe-y", type=str, default=",".join(str(v) for v in PROBE_Y_DEFAULT), help="Comma-separated probe y list")
    p.add_argument("--idw-k", type=int, default=16, help="IDW neighbors (fallback only)")
    p.add_argument("--idw-power", type=float, default=2.0, help="IDW power (fallback only)")
    p.add_argument("--workers", type=int, default=None, help="Worker processes for parallel cases (default: CPU count)")
    args = p.parse_args()

    probe_y_list = [float(tok.strip()) for tok in args.probe_y.split(",") if tok.strip()]

    merge_all(
        input_dir=args.input_dir,
        output_file=args.output_file,
        pattern=args.pattern,
        time_chunk=args.time_chunk,
        probe_x=args.probe_x,
        probe_y_list=probe_y_list,
        idw_k=args.idw_k,
        idw_power=args.idw_power,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()
