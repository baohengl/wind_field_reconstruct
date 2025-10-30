#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import h5py
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm.auto import tqdm

DEFAULT_H5 = "/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_interpolation_all_cases.h5"

def _acc_sum_sumsq(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    flat = arr.reshape(-1, 3).astype(np.float64, copy=False)
    s   = flat.sum(axis=0, dtype=np.float64)
    ss  = (flat * flat).sum(axis=0, dtype=np.float64)
    return s, ss, flat.shape[0]

def _process_case(h5_path: str, case_name: str, time_chunk: int,
                  t_stride: int, z_stride: int, xy_stride: int,
                  rdcc_nbytes: int, rdcc_nslots: int, rdcc_w0: float) -> Tuple[np.ndarray, np.ndarray, int, str]:
    total_sum  = np.zeros(3, dtype=np.float64)
    total_sumsq= np.zeros(3, dtype=np.float64)
    total_cnt  = 0

    # 只读 + 大缓存
    with h5py.File(h5_path, "r",
                   rdcc_nbytes=rdcc_nbytes,
                   rdcc_nslots=rdcc_nslots,
                   rdcc_w0=rdcc_w0,
                   swmr=True, libver="latest") as f:
        g = f["cases"][case_name]
        if "section" not in g:
            return total_sum, total_sumsq, total_cnt, case_name

        dset = g["section"]  # (T, Z, Ny, Nx, 3)
        if dset.ndim != 5 or dset.shape[-1] != 3:
            raise RuntimeError(f"{dset.name} shape must be (T,Z,Ny,Nx,3), got {dset.shape}")

        T, Z, Ny, Nx, _ = dset.shape

        # 与文件chunk对齐：时间块
        chunk_t = time_chunk
        if dset.chunks and dset.chunks[0]:
            chunk_t = max(1, min(int(dset.chunks[0]), int(time_chunk)))

        # 子采样切片
        z_idx = np.arange(0, Z, max(1, z_stride))
        y_idx = slice(0, Ny, max(1, xy_stride))
        x_idx = slice(0, Nx, max(1, xy_stride))

        # 按时间顺序块读，减少随机访问
        for t0 in range(0, T, chunk_t * max(1, t_stride)):
            # 这里按 stride 跨块取，避免读取无用块
            t1 = min(T, t0 + chunk_t)
            # 构造本次时间索引：在 [t0, t1) 内再按 t_stride 取子样
            t_sel = np.arange(t0, t1, max(1, t_stride))
            if t_sel.size == 0:
                continue

            # 逐 z 层读取（section 的 chunk 通常 z=1，有效减少读块数）
            for z in z_idx:
                # 读出 (T_sel, 1, Ny', Nx', 3) -> squeeze 到 (T_sel, Ny', Nx', 3)
                block = dset[t_sel, z:z+1, y_idx, x_idx, :].astype(np.float32, copy=False).squeeze(axis=1)
                s, ss, n = _acc_sum_sumsq(block)
                total_sum   += s
                total_sumsq += ss
                total_cnt   += n
                del block

    return total_sum, total_sumsq, total_cnt, case_name

def compute_stats(h5_path: str, workers: int, time_chunk: int,
                  t_stride: int, z_stride: int, xy_stride: int,
                  case_stride: int, rdcc_nbytes: int, rdcc_nslots: int, rdcc_w0: float) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "cases" not in f:
            raise RuntimeError("Missing /cases")
        all_cases = [k for k in f["cases"].keys() if isinstance(f["cases"][k], h5py.Group)]

    # case 级别子采样（如每隔 N 个取一个）
    cases = all_cases[::max(1, case_stride)]
    if not cases:
        raise RuntimeError("No cases selected after case_stride filtering.")

    total_sum  = np.zeros(3, dtype=np.float64)
    total_sumsq= np.zeros(3, dtype=np.float64)
    total_cnt  = 0

    task = partial(_process_case, h5_path,
                   time_chunk=time_chunk,
                   t_stride=t_stride, z_stride=z_stride, xy_stride=xy_stride,
                   rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots, rdcc_w0=rdcc_w0)

    if workers <= 1:
        for cname in tqdm(cases, desc="cases", leave=True):
            s, ss, n, _ = task(cname)
            total_sum   += s
            total_sumsq += ss
            total_cnt   += n
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(task, cname): cname for cname in cases}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"cases (workers={workers})", leave=True):
                s, ss, n, _ = fut.result()
                total_sum   += s
                total_sumsq += ss
                total_cnt   += n

    if total_cnt == 0:
        raise RuntimeError("No samples counted.")

    mean = (total_sum / total_cnt).astype(np.float64)
    ex2  = (total_sumsq / total_cnt).astype(np.float64)
    var  = np.maximum(ex2 - mean * mean, 0.0)
    std  = np.sqrt(var, dtype=np.float64)

    return mean.astype(np.float32), std.astype(np.float32)

def write_meta(h5_path: str, mean: np.ndarray, std: np.ndarray):
    with h5py.File(h5_path, "r+") as f:
        meta = f.require_group("meta")
        for name, arr in [("section_velocity_mean", mean), ("section_velocity_std", std)]:
            if name in meta: del meta[name]
            dset = meta.create_dataset(name, data=arr.astype(np.float32))
            dset.attrs["desc"] = "Global stats over sampled /cases/*/section(T,Z,Ny,Nx,3) for [x,y,z]."
            dset.attrs["note"] = "Subsampled along time/z/xy; float64 accumulation; outputs float32."

def main():
    ap = argparse.ArgumentParser("Fast global mean/std over huge HDF5 with subsampling & parallel reading.")
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8)//2))
    ap.add_argument("--time-chunk", type=int, default=128, help="read block along time; will be clamped to dataset chunk")
    ap.add_argument("--t-stride",  type=int, default=8,   help="subsample along time (e.g., 8 means take every 8th frame)")
    ap.add_argument("--z-stride",  type=int, default=1,   help="subsample along z layers (e.g., 2 means every other layer)")
    ap.add_argument("--xy-stride", type=int, default=2,   help="subsample Ny/Nx by stride (aligned slicing)")
    ap.add_argument("--case-stride", type=int, default=1, help="subsample cases list (e.g., 5 -> take every 5th case)")
    ap.add_argument("--rdcc-nbytes", type=int, default=(1<<28), help="HDF5 raw data chunk cache bytes (e.g., 256MB)")
    ap.add_argument("--rdcc-nslots", type=int, default=(1<<20), help="HDF5 chunk slots")
    ap.add_argument("--rdcc-w0",     type=float, default=0.75,  help="HDF5 cache preemption policy")
    ap.add_argument("--dry-run", action="store_true", help="only compute and print, do not write meta")
    args = ap.parse_args()

    mean, std = compute_stats(
        h5_path=args.h5,
        workers=args.workers,
        time_chunk=args.time_chunk,
        t_stride=args.t_stride,
        z_stride=args.z_stride,
        xy_stride=args.xy_stride,
        case_stride=args.case_stride,
        rdcc_nbytes=args.rdcc_nbytes,
        rdcc_nslots=args.rdcc_nslots,
        rdcc_w0=args.rdcc_w0
    )

    print("=== Global stats (components: x, y, z) ===")
    print(f"Mean: [{mean[0]:.6e}, {mean[1]:.6e}, {mean[2]:.6e}]")
    print(f"Std : [{std[0]:.6e},  {std[1]:.6e},  {std[2]:.6e}]")

    if not args.dry_run:
        write_meta(args.h5, mean, std)
        print(f"Wrote /meta/section_velocity_mean & /meta/section_velocity_std to {args.h5}")

if __name__ == "__main__":
    main()
