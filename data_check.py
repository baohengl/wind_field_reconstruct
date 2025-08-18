"""Utility script to inspect wind field HDF5 files.

This script is derived from the notebook ``data_check.ipynb`` and offers a
command line interface for quickly inspecting the contents of a wind field HDF5
file and optionally plotting a slice of the velocity field.

Example usage::

    python data_check.py --file-path mydata.h5 --case-name 3 --plot

The default parameters reflect the most commonly used dataset and case
information.
"""

from __future__ import annotations

import argparse
import glob
import os


def inspect_file(file_path: str, t_str: str = "0.005") -> None:
    """Print basic information about all cases contained in the HDF5 file.

    Parameters
    ----------
    file_path:
        Path to the HDF5 file to inspect.
    t_str:
        A specific time step to probe for each case.  If the time step does not
        exist in a case, a warning is printed.
    """

    import h5py

    with h5py.File(file_path, "r") as f:
        cases = list(f.keys())
        print("已有 case：", cases)

        for case in cases:
            case_grp = f[case]
            print(f"\n=== Case: {case} ===")
            print(
                "时间序列 dtype:",
                case_grp["time"].dtype,
                "shape:",
                case_grp["time"].shape,
            )
            # Only display the first few time steps for brevity
            print("时间步：", case_grp["time"][:10], "...")

            print(
                "坐标 dtype:",
                case_grp["coords"].dtype,
                "shape:",
                case_grp["coords"].shape,
            )

            if t_str in case_grp:
                print(
                    f"\n{case} -> t = {t_str} 的数据：",
                    list(case_grp[t_str].keys()),
                )
                print(
                    "  velocity shape:",
                    case_grp[t_str]["velocity"].shape,
                    "dtype:",
                    case_grp[t_str]["velocity"].dtype,
                )
            else:
                print(f"⚠️ {case} 中没有时间步 {t_str}")


def inspect_files(file_paths: list[str], t_str: str = "0.005") -> None:
    """Inspect multiple HDF5 files sequentially.

    Parameters
    ----------
    file_paths:
        List of paths to HDF5 files.
    t_str:
        Time step to probe for each case.
    """

    for fp in file_paths:
        print(f"\n##### 文件: {fp} #####")
        inspect_file(fp, t_str=t_str)


def find_case_file(file_paths: list[str], case_name: str) -> str | None:
    """Return the path of the file containing ``case_name``.

    Parameters
    ----------
    file_paths:
        List of HDF5 file paths to search.
    case_name:
        Target case name to look for.
    """

    import h5py

    for fp in file_paths:
        with h5py.File(fp, "r") as f:
            if case_name in f:
                return fp
    return None


def plot_slice(
    file_path: str,
    case_name: str,
    t_target: str = "0.0125",
    z_target: float = 0.215,
    tol: float = 0.01,
) -> None:
    """Plot a slice of the *w* velocity component at a given height.

    Parameters
    ----------
    file_path:
        Path to the HDF5 file to inspect.
    case_name:
        Name of the case within the HDF5 file.
    t_target:
        Time step of the data to visualize.
    z_target:
        Target height for the slice.
    tol:
        Thickness tolerance for selecting cells around ``z_target``.
    """

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    with h5py.File(file_path, "r") as f:
        if case_name not in f:
            raise KeyError(f"❌ HDF5 中不存在 case: {case_name}")
        case_grp = f[case_name]

        if t_target not in case_grp:
            raise KeyError(f"❌ Case {case_name} 中不存在时间步: {t_target}")

        coords = case_grp["coords"][:]
        velocity = case_grp[t_target]["velocity"][:]

    mask = np.abs(coords[:, 2] - z_target) < tol
    coords_slice = coords[mask]
    velocity_slice = velocity[mask]

    if coords_slice.shape[0] == 0:
        raise ValueError(f"⚠️ 在 z={z_target}±{tol} 未找到单元")

    x = coords_slice[:, 0]
    y = coords_slice[:, 1]
    w = velocity_slice[:, 2]

    nx, ny = 200, 200
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Wi = griddata((x, y), w, (Xi, Yi), method="linear")

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(Xi, Yi, Wi, levels=50, cmap="jet")
    plt.colorbar(contour, label="w (m/s)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Case {case_name} | z = {z_target} | t = {t_target} | w 高程图"
    )
    plt.axis("equal")

    import os

    output_dir = "/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"case_{case_name}_z_{z_target}_t_{t_target}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect wind field HDF5 data")
    parser.add_argument(
        "--file-path",
        default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_all_cases.h5",
        help="Path to a single HDF5 file",
    )
    parser.add_argument(
        "--dir",
        help="Directory containing multiple windfield_all_cases*.h5 files",
    )
    parser.add_argument(
        "--pattern",
        default="windfield_all_cases*.h5",
        help="Glob pattern used with --dir (default: windfield_all_cases*.h5)",
    )
    parser.add_argument(
        "--case-name",
        default="3",
        help="Name of the case to plot when --plot is specified",
    )
    parser.add_argument(
        "--time",
        default="0.005",
        help="Time step to query when listing case information",
    )
    parser.add_argument(
        "--t-target",
        default="0.0125",
        help="Time step used for plotting when --plot is given",
    )
    parser.add_argument(
        "--z-target",
        type=float,
        default=0.215,
        help="Target height for the plot slice",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance when selecting cells around z-target",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a w-velocity slice for the selected case",
    )
    args = parser.parse_args()

    if args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching {args.pattern} found in {args.dir}"
            )
        inspect_files(files, t_str=args.time)
        if args.plot:
            case_file = find_case_file(files, args.case_name)
            if case_file is None:
                raise KeyError(
                    f"Case {args.case_name} not found in provided HDF5 files"
                )
            plot_slice(
                case_file,
                args.case_name,
                t_target=args.t_target,
                z_target=args.z_target,
                tol=args.tol,
            )
    else:
        inspect_file(args.file_path, t_str=args.time)
        if args.plot:
            plot_slice(
                args.file_path,
                args.case_name,
                t_target=args.t_target,
                z_target=args.z_target,
                tol=args.tol,
            )


if __name__ == "__main__":
    main()
