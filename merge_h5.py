#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-worker streaming merge & resample windfield:
- Sequentially process cases (workers forced to 1)
- Stream (chunked) write into ONE output HDF5
- Optional intra-case threading over z-layers
- Optional precomputed operators (linear or IDW)

Input schema (per case group, e.g. "/0"):
  /<case>/coords           (N, 3) float16
  /<case>/time             (T,)   float32
  /<case>/<t_str>/velocity (N, 3) float16

Output (single file):
  /time                                 (T,) if identical across cases
  /meta/{x_values,y_values,plane_grid_json,probe_x_value,probe_y_values,config_json}
  /cases/case_000/{section(T,Z,Ny,Nx,3)_f16, z_values_section(Z)_f32, probes(T,Z,Nprobe,3)_f16}
"""

import argparse
import glob
import json
import os
from typing import Tuple, Optional, List

import h5py
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# ---------- SciPy availability ----------
_SCI_KDTREE = False
_SCI_SPARSE = False
_SCI_DELAUNAY = False
try:
    from scipy.spatial import cKDTree, Delaunay  # type: ignore
    _SCI_KDTREE = True
    _SCI_DELAUNAY = True
except Exception:
    pass
try:
    from scipy.sparse import csr_matrix  # type: ignore
    _SCI_SPARSE = True
except Exception:
    pass

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

# HDF5 compression (fast)
COMP = dict(compression="lzf", shuffle=True)  # change to ("gzip", compression_opts=4) if needed

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

# ---------- Readers for your schema ----------
def _read_case_schema(g_case: h5py.Group):
    if "coords" not in g_case or "time" not in g_case:
        raise KeyError(f"Group {g_case.name} missing 'coords' or 'time'.")
    coords = g_case["coords"][...].astype(np.float32, copy=False)  # (N,3)
    time = g_case["time"][...].astype(np.float32, copy=False)      # (T,)
    # list timestep subgroups (strings like "0.1")
    step_names = [k for k, v in g_case.items() if isinstance(v, h5py.Group)]
    def _as_float(s):
        try: return float(s.lstrip("tT"))
        except Exception: return float("inf")
    step_names.sort(key=_as_float)
    return coords, time, step_names

# ---------- A. Build interpolation operators ----------
def _build_idw_operator(xy_src: np.ndarray, xy_tgt: np.ndarray, k: int = 16, power: float = 2.0):
    """Return CSR A such that f_tgt ~= A @ f_src (kNN + IDW)."""
    if not (_SCI_KDTREE and _SCI_SPARSE):
        return None  # operator not available
    tree = cKDTree(xy_src)
    dist, idx = tree.query(xy_tgt, k=min(k, xy_src.shape[0]))
    dist = np.asarray(dist); idx = np.asarray(idx)
    if dist.ndim == 1:  # when k=1
        dist = dist[:, None]; idx = idx[:, None]
    dist[dist == 0] = 1e-12
    w = 1.0 / (dist ** power)
    w /= w.sum(axis=1, keepdims=True)
    M = xy_tgt.shape[0]
    rows = np.repeat(np.arange(M), w.shape[1])
    cols = idx.reshape(-1)
    data = w.reshape(-1)
    A = csr_matrix((data, (rows, cols)), shape=(M, xy_src.shape[0]))
    return A

def _build_linear_operator(xy_src: np.ndarray, xy_tgt: np.ndarray):
    """
    Piecewise-linear operator via Delaunay barycentric weights.
    Matches griddata(method='linear') inside convex hull; outside uses nearest neighbor.
    """
    if not (_SCI_DELAUNAY and _SCI_SPARSE and _SCI_KDTREE):
        return None
    tri = Delaunay(xy_src)
    simp = tri.find_simplex(xy_tgt)                         # (M,)
    rows, cols, data = [], [], []

    # inside hull: barycentric weights on 3 vertices
    mask_in = simp >= 0
    if np.any(mask_in):
        Tm = tri.transform[simp[mask_in]]                   # (M_in, 3, 2)
        X = xy_tgt[mask_in] - Tm[:, 2, :]                   # (M_in, 2)
        r = np.einsum('ijk,ik->ij', Tm[:, :2, :], X)        # (M_in, 2)
        w0 = r[:, 0]; w1 = r[:, 1]; w2 = 1.0 - w0 - w1
        verts = tri.simplices[simp[mask_in]]                # (M_in, 3)
        rows += list(np.repeat(np.where(mask_in)[0], 3))
        cols += list(verts.reshape(-1))
        data += list(np.stack([w0, w1, w2], axis=1).reshape(-1))

    # outside hull: nearest neighbor with weight 1
    mask_out = ~mask_in
    if np.any(mask_out):
        tree = cKDTree(xy_src)
        _, idx = tree.query(xy_tgt[mask_out], k=1)
        rows += list(np.where(mask_out)[0])
        cols += list(idx.reshape(-1))
        data += [1.0] * np.count_nonzero(mask_out)

    A = csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))),
                   shape=(xy_tgt.shape[0], xy_src.shape[0]))
    return A

def _choose_operator(op_mode: str):
    """
    Return a builder function according to mode.
    """
    if op_mode == "linear":
        return _build_linear_operator
    if op_mode == "idw":
        return _build_idw_operator
    # auto
    if _SCI_DELAUNAY and _SCI_SPARSE and _SCI_KDTREE:
        return _build_linear_operator
    if _SCI_KDTREE and _SCI_SPARSE:
        return _build_idw_operator
    return None  # will fallback to per-step interpolation

# ---------- Core: per-case processing (streaming write) ----------
def process_case_streaming(
    h5_out: h5py.File,
    case_out_name: str,
    fpath_in: str,
    case_in_name: str,
    time_chunk: int = 128,
    probe_x: float = PROBE_X_DEFAULT,
    probe_y_list: Optional[List[float]] = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
    operator: str = "auto",           # "auto" | "linear" | "idw"
    intra_threads: Optional[int] = None,
    progress: bool = False,
) -> np.ndarray:
    """
    Stream a single case into h5_out under /cases/<case_out_name>.
    Returns the time array of this case (float32).
    """
    xs, ys, Xg, Yg = _grid_xy()
    Ny, Nx = ys.shape[0], xs.shape[0]
    grid_xy_flat = np.column_stack([Xg.reshape(-1), Yg.reshape(-1)])

    with h5py.File(fpath_in, "r") as h5_in:
        g_case = h5_in[case_in_name]
        coords, time, step_names = _read_case_schema(g_case)

        if probe_y_list is None:
            probe_y_list = PROBE_Y_DEFAULT
        probe_y = np.asarray(probe_y_list, dtype=META_DTYPE)
        probe_y = _clamp(probe_y, Y_MIN, Y_MAX)
        probe_x = float(np.clip(probe_x, X_MIN, X_MAX))
        probe_xy = np.column_stack([np.full_like(probe_y, probe_x), probe_y])
        Nprobe = probe_xy.shape[0]

        # z layers
        z_vals, layer_idx = _unique_z_levels(coords[:, 2], tol=1e-6)
        Z = z_vals.shape[0]
        T = len(step_names)

        # per-z source xy
        z_to_idx = [np.where(layer_idx == z_id)[0] for z_id in range(Z)]
        z_to_xy  = [coords[idx, :2].astype(np.float32, copy=False) for idx in z_to_idx]

        # build operators (optional)
        builder = _choose_operator(operator)
        A_sec_list, A_prb_list = [None]*Z, [None]*Z
        if builder is not None:
            for z_id in range(Z):
                xy = z_to_xy[z_id]
                if builder is _build_idw_operator:
                    A_sec_list[z_id] = builder(xy, grid_xy_flat, k=idw_k, power=idw_power)
                    A_prb_list[z_id] = builder(xy, probe_xy,     k=idw_k, power=idw_power)
                else:
                    A_sec_list[z_id] = builder(xy, grid_xy_flat)
                    A_prb_list[z_id] = builder(xy, probe_xy)

        # ensure case group & datasets exist in output
        g_cases = h5_out.require_group("cases")
        g_case_out = g_cases.require_group(case_out_name)

        # create empty datasets with final shapes (chunked by time)
        dset_sec = g_case_out.create_dataset(
            "section",
            shape=(T, Z, Ny, Nx, 3),
            dtype=OUT_DTYPE,
            chunks=(min(time_chunk, T), 1, Ny, Nx, 3),
            **COMP,
        )
        g_case_out.create_dataset("z_values_section", data=z_vals.astype(META_DTYPE))
        dset_prb = g_case_out.create_dataset(
            "probes",
            shape=(T, Z, Nprobe, 3),
            dtype=OUT_DTYPE,
            chunks=(min(time_chunk, T), 1, Nprobe, 3),
            **COMP,
        )

        # threading across z-layers
        if intra_threads is None or intra_threads <= 0:
            intra_threads = min(8, os.cpu_count() or 1)

        pb = tqdm(total=T, desc=case_out_name, leave=True) if progress else None

        # helpers for fallback interpolation
        _has_gd = False
        try:
            from scipy.interpolate import griddata as _gd  # type: ignore
            _has_gd = True
        except Exception:
            _has_gd = False
        Xg32, Yg32 = Xg.astype(np.float32), Yg.astype(np.float32)

        def _work_one_z(z_id: int, vel_chunk: np.ndarray, Ny: int, Nx: int, probe_xy: np.ndarray):
            idx = z_to_idx[z_id]
            xy  = z_to_xy[z_id]
            A_sec = A_sec_list[z_id]
            A_prb = A_prb_list[z_id]
            tc = vel_chunk.shape[0]
            out_sec_z = np.empty((tc, Ny, Nx, 3), dtype=np.float32)
            out_prb_z = np.empty((tc, Nprobe, 3), dtype=np.float32)

            if (A_sec is not None) and (A_prb is not None):
                for comp in range(3):
                    vals = vel_chunk[:, idx, comp]           # (tc, n_i)
                    grid_all  = (A_sec @ vals.T).T           # (tc, Ny*Nx)
                    probe_all = (A_prb @ vals.T).T           # (tc, Nprobe)
                    out_sec_z[:, :, :, comp] = grid_all.reshape(tc, Ny, Nx)
                    out_prb_z[:, :, comp]    = probe_all
            else:
                tgt_grid = (Xg32, Yg32)
                for tt in range(tc):
                    vals3 = vel_chunk[tt, idx, :]
                    for comp in range(3):
                        val = vals3[:, comp]
                        if _has_gd:
                            arr = _gd(xy, val, tgt_grid, method="linear")
                            nan = np.isnan(arr)
                            if nan.any():
                                arr2 = _gd(xy, val, tgt_grid, method="nearest")
                                arr[nan] = arr2[nan]
                            out_sec_z[tt, :, :, comp] = arr.astype(np.float32)
                            prb = _gd(xy, val, (probe_xy[:, 0], probe_xy[:, 1]), method="linear")
                            nanp = np.isnan(prb)
                            if nanp.any():
                                prb2 = _gd(xy, val, (probe_xy[:, 0], probe_xy[:, 1]), method="nearest")
                                prb[nanp] = prb2[nanp]
                            out_prb_z[tt, :, comp] = prb.astype(np.float32)
                        else:
                            # naive IDW (k = all)
                            pts_grid = np.column_stack([Xg32.ravel(), Yg32.ravel()])
                            diffs = pts_grid[:, None, :] - xy[None, :, :]
                            d2 = np.sum(diffs * diffs, axis=2); d2[d2 == 0] = 1e-12
                            w = 1.0 / (d2 ** (2.0/2.0))
                            out_sec = (w @ val) / np.sum(w, axis=1)
                            out_sec_z[tt, :, :, comp] = out_sec.reshape(Ny, Nx)
                            diffs2 = probe_xy[:, None, :] - xy[None, :, :]
                            d2p = np.sum(diffs2*diffs2, axis=2); d2p[d2p == 0] = 1e-12
                            wp = 1.0 / (d2p ** (2.0/2.0))
                            out_prb_z[tt, :, comp] = (wp @ val) / np.sum(wp, axis=1)

            return z_id, out_sec_z, out_prb_z

        # time streaming
        for t0 in range(0, T, time_chunk):
            t1 = min(T, t0 + time_chunk)
            tc = t1 - t0

            # prefetch this chunk velocities -> (tc, N, 3) float32
            vel_list = []
            for step_idx in range(t0, t1):
                step_name = step_names[step_idx]
                vel = g_case[step_name]["velocity"][...].astype(np.float32, copy=False)
                vel_list.append(vel)
            vel_chunk = np.stack(vel_list, axis=0)  # (tc, N, 3)

            # allocate buffers for this chunk
            buf_sec = np.empty((tc, Z, Ny, Nx, 3), dtype=np.float32)
            buf_prb = np.empty((tc, Z, Nprobe, 3), dtype=np.float32)

            if intra_threads > 1 and Z > 1:
                with ThreadPoolExecutor(max_workers=intra_threads) as tex:
                    futures = [tex.submit(_work_one_z, z, vel_chunk, Ny, Nx, probe_xy) for z in range(Z)]
                    for fut in futures:
                        z_id, sec_z, prb_z = fut.result()
                        buf_sec[:, z_id, ...] = sec_z
                        buf_prb[:, z_id, ...] = prb_z
            else:
                for z in range(Z):
                    z_id, sec_z, prb_z = _work_one_z(z, vel_chunk, Ny, Nx, probe_xy)
                    buf_sec[:, z_id, ...] = sec_z
                    buf_prb[:, z_id, ...] = prb_z

            # zero edges on section (not probes)
            _force_edge_zero(buf_sec)

            # stream write this chunk and flush
            dset_sec[t0:t1, ...] = buf_sec.astype(OUT_DTYPE, copy=False)
            dset_prb[t0:t1, ...] = buf_prb.astype(OUT_DTYPE, copy=False)
            h5_out.flush()

            # free buffers
            del vel_chunk, buf_sec, buf_prb

            if pb: pb.update(tc)

        if pb: pb.close()

        # also save this case's time (useful即使全局 /time 不一致)
        if "time" in g_case_out:
            del g_case_out["time"]
        g_case_out.create_dataset("time", data=time.astype(META_DTYPE))

        return time  # float32

# ---------- Merge driver (single-worker, sequential) ----------
def merge_all(
    input_dir: str,
    output_file: str | None,
    pattern: str = "windfield_all_cases*.h5",
    input_files: Optional[List[str]] = None,
    time_chunk: int = 128,
    probe_x: float = PROBE_X_DEFAULT,
    probe_y_list: Optional[List[float]] = None,
    idw_k: int = 16,
    idw_power: float = 2.0,
    operator: str = "auto",           # "auto" | "linear" | "idw"
    workers: Optional[int] = 1,       # forced to 1
    intra_threads: Optional[int] = None,
    progress: bool = False,
):
    if input_files:
        files = [f if os.path.isabs(f) else os.path.join(input_dir, f) for f in input_files]
    else:
        files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files provided" if input_files else f"No files matching pattern '{pattern}' in {input_dir}"
        )

    if not output_file:
        output_file = os.path.join(input_dir, "dataset_merged_resampled_fp16.h5")
        print(f"[Info] No output path provided. Using default: {output_file}")

    if workers is None or workers != 1:
        print(f"[Info] Forcing workers=1 (single-worker streaming). Was: {workers}")
        workers = 1

    print("Merging & resampling (float16 outputs) in SINGLE-WORKER streaming mode:")
    for f in files:
        print("  -", f)

    xs, ys, _, _ = _grid_xy()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # collect (file, case)
    tasks: List[Tuple[str, str]] = []
    for fpath in files:
        with h5py.File(fpath, "r") as h5_in:
            for case_name in h5_in.keys():
                if isinstance(h5_in[case_name], h5py.Group):
                    tasks.append((fpath, case_name))

    with h5py.File(output_file, "w") as h5_out:
        # write meta once
        meta = h5_out.require_group("meta")
        meta.create_dataset("x_values", data=np.asarray(xs, dtype=META_DTYPE))
        meta.create_dataset("y_values", data=np.asarray(ys, dtype=META_DTYPE))

        grid_info = {
            "x": {"min": float(X_MIN), "max": float(X_MAX), "step": float(X_STEP), "count": int(len(xs))},
            "y": {"min": float(Y_MIN), "max": float(Y_MAX), "step": float(Y_STEP), "count": int(len(ys))},
            "note": "Section edges (x=0/1 or y=0/1.2) are zeroed after interpolation."
        }

        # UTF-8 string dtype for JSON fields
        str_dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("plane_grid_json", data=json.dumps(grid_info), dtype=str_dt)
        meta.create_dataset("probe_x_value", data=np.asarray(PROBE_X_DEFAULT if probe_x is None else probe_x, dtype=META_DTYPE))
        meta.create_dataset("probe_y_values", data=np.asarray(PROBE_Y_DEFAULT if probe_y_list is None else probe_y_list, dtype=META_DTYPE))

        # process cases sequentially
        global_time = None
        global_time_ok = True

        cases_pbar = tqdm(total=len(tasks), desc="cases", leave=True) if progress else None

        for idx, (fpath, case_name) in enumerate(tasks):
            case_id = f"case_{idx:03d}"
            print(f"  > {os.path.basename(fpath)}::{case_name} -> {case_id}")

            time_arr = process_case_streaming(
                h5_out=h5_out,
                case_out_name=case_id,
                fpath_in=fpath,
                case_in_name=case_name,
                time_chunk=time_chunk,
                probe_x=probe_x,
                probe_y_list=probe_y_list,
                idw_k=idw_k,
                idw_power=idw_power,
                operator=operator,
                intra_threads=intra_threads,
                progress=progress,
            )

            # check global time consistency
            if time_arr is not None and time_arr.size > 0:
                if global_time is None:
                    global_time = time_arr
                else:
                    if (global_time.shape != time_arr.shape) or (
                        not np.allclose(global_time, time_arr, rtol=1e-7, atol=1e-10)
                    ):
                        global_time_ok = False

            if cases_pbar: cases_pbar.update(1)
            h5_out.flush()

        if cases_pbar: cases_pbar.close()

        # write global /time if consistent and compute dt for config
        cfg = {
            "dt": None,
            "layout": {"section": "section(T,Z,Ny,Nx,3)", "probes": "probes(T,Z,Nprobe,3)"},
            "dtype": {"section": "float16", "probes": "float16"},
            "z_crop": {"mode": "random_block", "block": 1, "same_for_XY": True},
            "interpolation": operator,
            "chunks": {"section": ["time_chunk", 1, "Ny", "Nx", 3], "probes": ["time_chunk", 1, "Nprobe", 3]},
            "parallel": {"workers_cases": 1},
            "intra_threads": int(intra_threads if intra_threads else 1),
        }

        if global_time_ok and global_time is not None:
            if "time" in h5_out:
                del h5_out["time"]
            h5_out.create_dataset("time", data=global_time.astype(META_DTYPE))
            if len(global_time) >= 2:
                cfg["dt"] = float(np.median(np.diff(global_time)))
            print("Wrote global /time and dt in meta/config_json")

        # write config_json (UTF-8)
        if "config_json" in meta:
            del meta["config_json"]
        meta.create_dataset("config_json", data=json.dumps(cfg), dtype=str_dt)

    print(f"Done. Output written to: {output_file}")

# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-worker streaming: resample x–y per z, add probes; float16 outputs; progress & intra-threads."
    )
    p.add_argument("input_dir", help="Directory containing windfield_all_cases*.h5")
    p.add_argument("output_file", nargs="?", default=None, help="Output HDF5 (default: <input_dir>/dataset_merged_resampled_fp16.h5)")
    p.add_argument("--pattern", default="windfield_all_cases*.h5", help="Glob pattern for inputs")
    p.add_argument("--input-files", nargs="+", default=None, help="Explicit HDF5 files to process (overrides --pattern)")
    p.add_argument("--time-chunk", type=int, default=128, help="Streaming chunk over timesteps (bigger is faster, needs more RAM)")
    p.add_argument("--probe-x", type=float, default=PROBE_X_DEFAULT, help="Probe x (default 0.5)")
    p.add_argument("--probe-y", type=str, default=",".join(str(v) for v in PROBE_Y_DEFAULT), help="Comma-separated probe y list")
    p.add_argument("--idw-k", type=int, default=16, help="IDW neighbors for operator")
    p.add_argument("--idw-power", type=float, default=2.0, help="IDW power for operator")
    p.add_argument("--operator", choices=["auto", "linear", "idw"], default="auto", help="Interpolation operator mode")
    p.add_argument("--workers", type=int, default=1, help="(Forced to 1) Process workers across cases")
    p.add_argument("--intra-threads", type=int, default=None, help="Threads per case over z-layers (default: min(8, CPU))")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    args = p.parse_args()

    if args.workers != 1:
        print(f"[Info] Forcing workers=1 (was {args.workers}).")

    probe_y_list = [float(tok.strip()) for tok in args.probe_y.split(",") if tok.strip()]

    merge_all(
        input_dir=args.input_dir,
        output_file=args.output_file,
        pattern=args.pattern,
        input_files=args.input_files,
        time_chunk=args.time_chunk,
        probe_x=args.probe_x,
        probe_y_list=probe_y_list,
        idw_k=args.idw_k,
        idw_power=args.idw_power,
        operator=args.operator,
        workers=1,  # force single-worker
        intra_threads=args.intra_threads,
        progress=args.progress,
    )

if __name__ == "__main__":
    main()
