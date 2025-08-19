#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concatenate multiple merged HDF5 datasets produced by merge_h5.py.

Each input file is expected to have structure:
  /time (optional)
  /meta/*
  /cases/case_###

The script copies /meta and /time from the first file and appends all
cases into a new sequential numbering under /cases.
"""

import argparse
import os
from typing import List

import h5py
import numpy as np


def concat_merged_h5(inputs: List[str], output: str) -> None:
    """Concatenate multiple per-node merged HDF5 files into one."""
    if not inputs:
        raise FileNotFoundError("No input files provided")

    with h5py.File(output, "w") as h5_out:
        cases_out = h5_out.require_group("cases")
        meta_written = False
        time_written = False
        case_idx = 0
        reference_time: np.ndarray | None = None

        for fpath in inputs:
            with h5py.File(fpath, "r") as h5_in:
                # copy meta once, verify others
                if not meta_written and "meta" in h5_in:
                    h5_in.copy("meta", h5_out)
                    meta_written = True
                elif "meta" in h5_in:
                    for key, ds in h5_in["meta"].items():
                        if not np.array_equal(h5_out["meta"][key][...], ds[...]):
                            raise ValueError(f"/meta mismatch in {fpath}")

                # copy /time once and verify
                if "time" in h5_in:
                    if not time_written:
                        h5_in.copy("time", h5_out)
                        reference_time = h5_in["time"][...]
                        time_written = True
                    else:
                        if not np.array_equal(reference_time, h5_in["time"][...]):
                            raise ValueError(f"/time mismatch in {fpath}")

                # copy cases
                for cname, g_case in h5_in["cases"].items():
                    new_name = f"case_{case_idx:03d}"
                    h5_in.copy(g_case, cases_out, name=new_name)
                    case_idx += 1
    print(f"Concatenated {case_idx} cases into {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate merged windfield HDF5 files")
    p.add_argument("output", help="Output HDF5 file")
    p.add_argument("inputs", nargs="+", help="Input merged HDF5 files")
    args = p.parse_args()

    concat_merged_h5(inputs=args.inputs, output=args.output)


if __name__ == "__main__":
    main()
