import os
import glob
import shutil
import subprocess
import argparse
import shlex


def run_single_case(case_num: int, source_folder: str, base_path: str,
                    output_h5: str, np: int) -> None:
    """Run one OpenFOAM case and collect its data."""
    case_num_name = str(case_num)
    case_path = os.path.join(base_path, case_num_name)

    if os.path.exists(case_path):
        shutil.rmtree(case_path)
    shutil.copytree(source_folder, case_path)

    command = (
        "export TMPDIR=/tmp/mytmp && mkdir -p $TMPDIR && export OMPI_MCA_pml=ob1 "
        "&& source /opt/openfoam11/etc/bashrc && decomposePar "
        f"&& mpirun -np {np} foamRun -parallel"
    )
    subprocess.run(["bash", "-c", command], cwd=case_path, check=True)

    data_script = os.path.join(os.path.dirname(__file__), "data_collection.py")
    subprocess.run(["python", data_script, case_path, output_h5], check=True)

    keep_folders = {"0", "constant", "system"}
    for folder in os.listdir(case_path):
        folder_path = os.path.join(case_path, folder)
        if os.path.isdir(folder_path) and folder not in keep_folders:
            command = (
                f"find {shlex.quote(folder_path)} -mindepth 1 -print0 | "
                f"xargs -0 -P{os.cpu_count() or 1} rm -rf && "
                f"rmdir {shlex.quote(folder_path)}"
            )
            subprocess.run(["bash", "-c", command], check=True)


def _next_available_h5(path: str) -> str:
    """Return a new HDF5 filename with an incremented index if needed.

    If ``path`` exists, or numbered variants ``*_N`` exist, the function
    returns ``path`` with ``_N`` appended where ``N`` is one greater than the
    highest existing index.  Otherwise, ``path`` is returned unchanged.
    """

    base, ext = os.path.splitext(path)
    existing = set()

    if os.path.exists(path):
        existing.add(0)

    pattern = f"{base}_*{ext}"
    for f in glob.glob(pattern):
        name = os.path.splitext(f)[0]
        suffix = name[len(base) + 1 :]
        if suffix.isdigit():
            existing.add(int(suffix))

    if not existing:
        return path

    idx = max(existing) + 1
    return f"{base}_{idx}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run OpenFOAM cases and collect data.")
    parser.add_argument("--start", type=int, required=True, help="Start case number (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End case number (inclusive)")
    parser.add_argument(
        "--source-folder",
        default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/original",
        help="Path to original case folder",
    )
    parser.add_argument(
        "--base-path",
        default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data",
        help="Base path where case copies are stored",
    )
    parser.add_argument(
        "--output-h5",
        default="/vscratch/grp-tengwu/baohengl_0228/reconstruct_wind_field/CFD_data/windfield_all_cases.h5",
        help="Path to aggregated HDF5 file",
    )
    parser.add_argument("--np", type=int, default=16, help="Number of processes for mpirun")
    args = parser.parse_args()

    output_h5 = _next_available_h5(args.output_h5)

    for case_num in range(args.start, args.end + 1):
        run_single_case(case_num, args.source_folder, args.base_path, output_h5, args.np)


if __name__ == "__main__":
    main()
