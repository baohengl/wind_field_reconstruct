#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect merged windfield HDF5 (new schema), plot a section contour,
and optionally dump probe velocities at a given time & z.

Schema expected:
  /meta/x_values (Nx,), /meta/y_values (Ny,), plane_grid_json (str), ...
  /time (T,)  [optional, only if all cases share identical time]
  /cases/case_###/
      section (T, Z, Ny, Nx, 3)  float16
      z_values_section (Z,)      float32
      probes (T, Z, Nprobe, 3)   float16
      time (T,)                  float32  [per-case time backup]

Examples
--------
# 仅查看文件结构和一个case的维度/精度
python inspect_merged_dataset.py

# 指定case并作图：按时间索引和z索引
python inspect_merged_dataset.py --case case_000 --time-idx 0 --z-idx 0 --plot

# 指定时间值与z值（程序会选最近的index），并导出探头数据为CSV
python inspect_merged_dataset.py --case case_003 --time-val 0.0125 --z-val 0.215 --dump-probes --csv probes_case003_t0p0125_z0p215.csv
"""

import argparse
import os
import json
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# 默认目录与文件
DEFAULT_BASE = "/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data"
DEFAULT_FILE = os.path.join(DEFAULT_BASE, "windfield_interpolation_all_cases.h5")

COMP_MAP = {"u": 0, "v": 1, "w": 2}


def _nearest_index(arr: np.ndarray, value: float) -> int:
    """Return index of nearest value in 1D array."""
    idx = int(np.argmin(np.abs(arr - value)))
    return idx


def summarize_file(h5: h5py.File) -> None:
    print("===== FILE SUMMARY =====")
    # meta
    if "meta" in h5:
        meta = h5["meta"]
        print(" /meta keys:", list(meta.keys()))
        if "x_values" in meta and "y_values" in meta:
            xs = meta["x_values"].shape[0]
            ys = meta["y_values"].shape[0]
            print(f"  x_values: shape={meta['x_values'].shape} dtype={meta['x_values'].dtype} (Nx={xs})")
            print(f"  y_values: shape={meta['y_values'].shape} dtype={meta['y_values'].dtype} (Ny={ys})")
        if "plane_grid_json" in meta:
            try:
                s = meta["plane_grid_json"][()].decode("utf-8") if isinstance(meta["plane_grid_json"][()], bytes) else meta["plane_grid_json"][()]
                grid_info = json.loads(s)
                print(f"  plane_grid_json: {grid_info}")
            except Exception:
                print("  plane_grid_json: <unreadable, but present>")
        if "probe_x_value" in meta:
            print("  probe_x_value:", meta["probe_x_value"][...])
        if "probe_y_values" in meta:
            print("  probe_y_values: shape", meta["probe_y_values"].shape)
    else:
        print(" (no /meta group)")

    # time
    if "time" in h5:
        t = h5["time"][...]
        print(" /time present: shape", t.shape, "dtype", t.dtype)
        if t.size > 0:
            print("  time[0:5] ->", t[:min(5, t.size)])
    else:
        print(" (no global /time)")

    # cases
    if "cases" in h5:
        cases = list(h5["cases"].keys())
        print(f" /cases: {len(cases)} cases ->", cases[:10], "..." if len(cases) > 10 else "")
    else:
        print(" (no /cases group)")
    print("========================\n")


def summarize_case(h5: h5py.File, case_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Print shapes/dtypes for a case; return (time_T, z_values)."""
    if "cases" not in h5 or case_name not in h5["cases"]:
        raise KeyError(f"Case '{case_name}' not found.")
    g = h5["cases"][case_name]
    print(f"===== CASE: {case_name} =====")
    if "section" in g:
        print(" section:", g["section"].shape, g["section"].dtype)
    else:
        print(" section: <missing>")

    if "probes" in g:
        print(" probes :", g["probes"].shape, g["probes"].dtype)
    else:
        print(" probes : <missing>")

    if "z_values_section" in g:
        z_vals = g["z_values_section"][...]
        print(" z_values_section:", z_vals.shape, z_vals.dtype)
    else:
        z_vals = np.array([], dtype=np.float32)
        print(" z_values_section: <missing>")

    # time: prefer global /time; else per-case /cases/case/time
    if "time" in h5:
        time_arr = h5["time"][...]
        print(" time (global):", time_arr.shape, time_arr.dtype)
    elif "time" in g:
        time_arr = g["time"][...]
        print(" time (per-case):", time_arr.shape, time_arr.dtype)
    else:
        time_arr = np.array([], dtype=np.float32)
        print(" time: <missing>")

    print("========================\n")
    return time_arr, z_vals


def plot_section(h5: h5py.File,
                 case_name: str,
                 time_idx: Optional[int],
                 time_val: Optional[float],
                 z_idx: Optional[int],
                 z_val: Optional[float],
                 comp: str = "w",
                 save_dir: Optional[str] = None) -> None:
    """Plot contour for one section (T,Z,Ny,Nx,3)."""
    if comp.lower() not in COMP_MAP:
        raise ValueError("comp must be one of {'u','v','w'}")
    cidx = COMP_MAP[comp.lower()]

    if "cases" not in h5 or case_name not in h5["cases"]:
        raise KeyError(f"Case '{case_name}' not found.")
    g = h5["cases"][case_name]

    # axes from meta
    if "meta" not in h5 or "x_values" not in h5["meta"] or "y_values" not in h5["meta"]:
        raise KeyError("Missing /meta/x_values or /meta/y_values.")
    xs = h5["meta"]["x_values"][...]
    ys = h5["meta"]["y_values"][...]
    Ny, Nx = ys.shape[0], xs.shape[0]

    # time
    if "time" in h5:
        time_arr = h5["time"][...]
    elif "time" in g:
        time_arr = g["time"][...]
    else:
        raise KeyError("No time array found (neither global /time nor /cases/<case>/time).")

    # z
    if "z_values_section" not in g:
        raise KeyError("Missing /cases/<case>/z_values_section")
    z_vals = g["z_values_section"][...]

    # choose indices
    if time_idx is None:
        if time_val is None:
            time_idx = 0
        else:
            time_idx = _nearest_index(time_arr, float(time_val))
    if z_idx is None:
        if z_val is None:
            z_idx = 0
        else:
            z_idx = _nearest_index(z_vals, float(z_val))

    sec = g["section"]  # (T,Z,Ny,Nx,3) float16
    if time_idx < 0 or time_idx >= sec.shape[0]:
        raise IndexError(f"time_idx out of range: {time_idx} (T={sec.shape[0]})")
    if z_idx < 0 or z_idx >= sec.shape[1]:
        raise IndexError(f"z_idx out of range: {z_idx} (Z={sec.shape[1]})")

    # slice -> (Ny,Nx)
    field = sec[time_idx, z_idx, :, :, cidx][...]  # float16
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(X, Y, field, levels=50)
    plt.colorbar(cs, label=f"{comp} (m/s)")
    t_show = time_arr[time_idx] if time_arr.size else time_idx
    z_show = z_vals[z_idx] if z_vals.size else z_idx
    plt.title(f"{case_name} | {comp} | t={t_show:.6g} | z={z_show:.6g}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(h5.filename))
    os.makedirs(save_dir, exist_ok=True)
    out_png = os.path.join(save_dir, f"{case_name}_t{time_idx}_z{z_idx}_{comp}.png")
    plt.savefig(out_png, dpi=150)
    print(f"[Saved] {out_png}")
    plt.show()


def dump_probes(h5: h5py.File,
                case_name: str,
                time_idx: Optional[int],
                time_val: Optional[float],
                z_idx: Optional[int],
                z_val: Optional[float],
                csv_path: Optional[str] = None) -> str:
    """
    Dump probe velocities (u,v,w and |U|) for a given case/time/z.
    Returns the CSV path written.
    """
    if "cases" not in h5 or case_name not in h5["cases"]:
        raise KeyError(f"Case '{case_name}' not found.")
    g = h5["cases"][case_name]

    # time array (global preferred)
    if "time" in h5:
        time_arr = h5["time"][...]
    elif "time" in g:
        time_arr = g["time"][...]
    else:
        raise KeyError("No time array found (neither global /time nor /cases/<case>/time).")

    # z values
    if "z_values_section" not in g:
        raise KeyError("Missing /cases/<case>/z_values_section")
    z_vals = g["z_values_section"][...]

    # choose indices
    if time_idx is None:
        if time_val is None:
            time_idx = 0
        else:
            time_idx = _nearest_index(time_arr, float(time_val))
    if z_idx is None:
        if z_val is None:
            z_idx = 0
        else:
            z_idx = _nearest_index(z_vals, float(z_val))

    probes = g["probes"]  # (T,Z,Nprobe,3) float16
    if time_idx < 0 or time_idx >= probes.shape[0]:
        raise IndexError(f"time_idx out of range: {time_idx} (T={probes.shape[0]})")
    if z_idx < 0 or z_idx >= probes.shape[1]:
        raise IndexError(f"z_idx out of range: {z_idx} (Z={probes.shape[1]})")

    data = probes[time_idx, z_idx, :, :][...].astype(np.float32)  # (Nprobe, 3)
    Np = data.shape[0]
    speed = np.sqrt(np.sum(data**2, axis=1))  # (Nprobe,)

    # probe coordinates from meta
    px = None
    py = None
    if "meta" in h5:
        if "probe_x_value" in h5["meta"]:
            px = float(h5["meta"]["probe_x_value"][()])
        if "probe_y_values" in h5["meta"]:
            py = h5["meta"]["probe_y_values"][...].astype(np.float32)
            if py.shape[0] != Np:
                # 仅提示不一致，不阻断导出
                print(f"[Warn] /meta/probe_y_values count ({py.shape[0]}) != Nprobe ({Np})")

    # 打印到终端（前几条）
    t_show = time_arr[time_idx] if time_arr.size else time_idx
    z_show = z_vals[z_idx] if z_vals.size else z_idx
    print(f"--- Probes @ case={case_name}, t_idx={time_idx} (t={t_show:.6g}), z_idx={z_idx} (z={z_show:.6g}) ---")
    header = f"{'idx':>4}  {'u':>12} {'v':>12} {'w':>12} {'|U|':>12}"
    if px is not None and py is not None and py.shape[0] == Np:
        header = f"{'idx':>4}  {'x':>8} {'y':>10}  {'u':>12} {'v':>12} {'w':>12} {'|U|':>12}"
    print(header)
    for i in range(Np):
        u, v, w = data[i, :]
        mag = speed[i]
        if px is not None and py is not None and py.shape[0] == Np:
            print(f"{i:4d}  {px:8.3f} {py[i]:10.3f}  {u:12.6g} {v:12.6g} {w:12.6g} {mag:12.6g}")
        else:
            print(f"{i:4d}  {u:12.6g} {v:12.6g} {w:12.6g} {mag:12.6g}")

    # 写 CSV
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(h5.filename))
        csv_path = os.path.join(
            base_dir,
            f"{case_name}_probes_t{time_idx}_z{z_idx}.csv"
        )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", encoding="utf-8") as f:
        if px is not None and py is not None and py.shape[0] == Np:
            f.write("probe_idx,x,y,u,v,w,mag\n")
            for i in range(Np):
                u, v, w = data[i, :]
                f.write(f"{i},{px:.6f},{py[i]:.6f},{u:.9g},{v:.9g},{w:.9g},{speed[i]:.9g}\n")
        else:
            f.write("probe_idx,u,v,w,mag\n")
            for i in range(Np):
                u, v, w = data[i, :]
                f.write(f"{i},{u:.9g},{v:.9g},{w:.9g},{speed[i]:.9g}\n")

    print(f"[Saved CSV] {csv_path}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="Inspect merged windfield HDF5 (new schema); plot section; dump probe data.")
    ap.add_argument("--file", default=DEFAULT_FILE, help=f"HDF5 file path (default: {DEFAULT_FILE})")
    ap.add_argument("--case", default="case_000", help="Case name under /cases (e.g., case_000)")

    # 选择时间与z（索引或数值）
    ap.add_argument("--time-idx", type=int, default=None, help="Time index to use (overrides --time-val)")
    ap.add_argument("--time-val", type=float, default=None, help="Time value to use (nearest index will be used)")
    ap.add_argument("--z-idx", type=int, default=None, help="Z index to use (overrides --z-val)")
    ap.add_argument("--z-val", type=float, default=None, help="Z value to use (nearest index will be used)")

    # 绘图
    ap.add_argument("--plot", action="store_true", help="Whether to plot a contour for the chosen section")
    ap.add_argument("--comp", choices=["u", "v", "w"], default="w", help="Velocity component to plot (default: w)")

    # 导出探头
    ap.add_argument("--dump-probes", action="store_true", help="Dump probes (u,v,w,|U|) for chosen time & z")
    ap.add_argument("--csv", type=str, default=None, help="CSV path to save probes (default: same dir as HDF5)")

    # 额外信息
    ap.add_argument("--show-meta", action="store_true", help="Print meta JSON/arrays details")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    with h5py.File(args.file, "r") as h5:
        summarize_file(h5)
        time_arr, z_vals = summarize_case(h5, args.case)

        # 打印精度信息（dtype）
        if "cases" in h5 and args.case in h5["cases"]:
            g = h5["cases"][args.case]
            if "section" in g:
                print(f"[dtype] section: {g['section'].dtype}")
            if "probes" in g:
                print(f"[dtype] probes : {g['probes'].dtype}")

        if args.show_meta and "meta" in h5:
            meta = h5["meta"]
            print("\n--- META DETAIL ---")
            for k in meta.keys():
                v = meta[k]
                if isinstance(v, h5py.Dataset):
                    val = v[()]
                    if isinstance(val, (bytes, bytearray)):
                        try:
                            print(f"{k}: (string) {val.decode('utf-8')}")
                        except Exception:
                            print(f"{k}: (bytes) {val[:64]}...")
                    else:
                        print(f"{k}: shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"{k}: <Group>")

        # 绘图
        if args.plot:
            plot_section(
                h5,
                case_name=args.case,
                time_idx=args.time_idx,
                time_val=args.time_val,
                z_idx=args.z_idx,
                z_val=args.z_val,
                comp=args.comp,
                save_dir=os.path.dirname(os.path.abspath(args.file)),
            )

        # 导出探头
        if args.dump_probes:
            dump_probes(
                h5,
                case_name=args.case,
                time_idx=args.time_idx,
                time_val=args.time_val,
                z_idx=args.z_idx,
                z_val=args.z_val,
                csv_path=args.csv,
            )


if __name__ == "__main__":
    main()
