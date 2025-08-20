#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concatenate multiple merged HDF5 datasets produced by merge_h5.py / merge_h5_optimized.py.

Each input file is expected to have structure:
  /time (optional)
  /meta/*
  /cases/case_###

The script copies /meta and /time from the first file and appends all
cases into a new sequential numbering under /cases.

Enhancements:
- Default base directory for resolving relative filenames:
  /vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/
- If an input/output path is not absolute, resolve it under the base dir.
- Pretty path resolution logging & early missing-file checks.
"""

import argparse
from typing import List, Optional

import h5py
import numpy as np
import os

# ---- Default base directory ----
DEFAULT_BASE_DIR = "/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/"


def _resolve_under_base(path: str, base: str) -> str:
    """If path is absolute, return as-is.
    If relative (no dir, or not starting with '/'), join with base directory."""
    if os.path.isabs(path):
        return path
    # normalize base
    base = os.path.abspath(base)
    return os.path.join(base, path)


def _meta_allclose(h5_meta_ref: h5py.Group, h5_meta_new: h5py.Group,
                   rtol: float = 1e-7, atol: float = 1e-10) -> None:
    """
    Raise ValueError if essential meta arrays mismatch; ignore cosmetic JSON differences.

    We verify only essential numeric items that affect geometry/layout:
      - x_values (1D)
      - y_values (1D)
      - probe_y_values (1D)  [optional]
      - probe_x_value (scalar) [optional]

    Other entries like plane_grid_json/config_json may differ in formatting or
    run-time parameters; we do not require exact equality.
    """
    essential_float_keys = ["x_values", "y_values"]
    optional_float_keys = ["probe_y_values"]
    optional_scalar_keys = ["probe_x_value"]

    def _check_key(key: str, optional: bool = False):
        in_ref = key in h5_meta_ref
        in_new = key in h5_meta_new
        if not in_ref and not in_new:
            if optional:
                return
            else:
                raise ValueError(f"/meta missing required key '{key}' in both files?")
        if in_ref and not in_new:
            if optional:
                return
            else:
                raise ValueError(f"/meta key '{key}' missing in new file")
        if in_new and not in_ref:
            if optional:
                return
            else:
                raise ValueError(f"/meta key '{key}' missing in reference file")

        a = h5_meta_ref[key][...]
        b = h5_meta_new[key][...]
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise ValueError(f"/meta/{key} mismatch")

    for k in essential_float_keys:
        _check_key(k, optional=False)
    for k in optional_float_keys:
        _check_key(k, optional=True)
    for k in optional_scalar_keys:
        _check_key(k, optional=True)


def concat_merged_h5(inputs: List[str], output: str, base_dir: str = DEFAULT_BASE_DIR) -> None:
    """Concatenate multiple per-node merged HDF5 files into one, resolving relative paths under base_dir."""
    if not inputs:
        raise FileNotFoundError("No input files provided")

    # Resolve output path
    output_resolved = _resolve_under_base(output, base_dir)
    out_dir = os.path.dirname(os.path.abspath(output_resolved))
    os.makedirs(out_dir, exist_ok=True)

    # Resolve input paths
    input_resolved = [_resolve_under_base(p, base_dir) for p in inputs]

    print("[Path resolution]")
    print(f"  Base dir: {os.path.abspath(base_dir)}")
    print(f"  Output  : {output_resolved}")
    print("  Inputs  :")
    for p_in, orig in zip(input_resolved, inputs):
        print(f"    - {orig} -> {p_in}")

    # Early check: all inputs exist
    missing = [p for p in input_resolved if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("The following input files do not exist:\n  " + "\n  ".join(missing))

    with h5py.File(output_resolved, "w") as h5_out:
        cases_out = h5_out.require_group("cases")
        meta_written = False
        time_written = False
        case_idx = 0
        reference_time: Optional[np.ndarray] = None

        for fpath in input_resolved:
            with h5py.File(fpath, "r") as h5_in:
                # ---- /meta ----
                if "meta" in h5_in:
                    if not meta_written:
                        h5_in.copy("meta", h5_out)
                        meta_written = True
                    else:
                        _meta_allclose(h5_out["meta"], h5_in["meta"])

                # ---- /time ----
                if "time" in h5_in:
                    t_new = h5_in["time"][...]
                    if not time_written:
                        h5_in.copy("time", h5_out)
                        reference_time = t_new
                        time_written = True
                    else:
                        if reference_time is None or not np.allclose(reference_time, t_new, rtol=1e-7, atol=1e-10):
                            raise ValueError(f"/time mismatch in {fpath}")

                # ---- /cases ----
                if "cases" not in h5_in:
                    # Allow files with no cases (skip)
                    continue

                for cname, g_case in h5_in["cases"].items():
                    if not isinstance(g_case, h5py.Group):
                        continue
                    new_name = f"case_{case_idx:03d}"
                    h5_in.copy(g_case, cases_out, name=new_name)
                    case_idx += 1

    print(f"Concatenated {case_idx} cases into {output_resolved}")


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate merged windfield HDF5 files (with a default base directory).")
    p.add_argument("output", help="Output HDF5 filename (relative names are resolved under the default base dir)")
    p.add_argument("inputs", nargs="+", help="Input merged HDF5 filenames (relative names are resolved under the default base dir)")
    p.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                   help=f"Base directory for resolving relative paths (default: {DEFAULT_BASE_DIR})")
    args = p.parse_args()

    concat_merged_h5(inputs=args.inputs, output=args.output, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
