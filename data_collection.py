import os
import glob
import h5py
import meshio
import numpy as np
import argparse
from collections import defaultdict


def collect_case(case_dir: str, output_h5: str) -> None:
    """Collect VTK data from an OpenFOAM case and append to an HDF5 file.

    Parameters
    ----------
    case_dir: str
        Path to the case directory containing processor* folders.
    output_h5: str
        Path to the aggregated HDF5 file where results are stored.
    """
    case_name = os.path.basename(case_dir)

    all_vtk = sorted(glob.glob(os.path.join(case_dir, "processor*/VTK/*.vtk")))
    if not all_vtk:
        raise FileNotFoundError("未找到任何 processorN/VTK/processorN_*.vtk 文件！")

    files_by_time: dict[str, list[str]] = defaultdict(list)
    for f in all_vtk:
        time_str = f.split("_")[-1].replace(".vtk", "")
        files_by_time[time_str].append(f)

    with h5py.File(output_h5, "a") as h5f:
        if case_name in h5f:
            print(f"⚠️  覆盖已有 case: {case_name}")
            del h5f[case_name]

        case_grp = h5f.create_group(case_name)
        times = sorted(files_by_time.keys(), key=lambda x: float(x))
        time_list = []

        for t in times:
            print(f"▶ 正在处理 {case_name} 时间步 {t} ...")
            part_files = files_by_time[t]

            all_coords = []
            all_velocity = []

            for pf in part_files:
                mesh = meshio.read(pf)
                if "U" not in mesh.cell_data:
                    raise KeyError(f"文件 {pf} 中未找到 Cell Data 'U'")
                points = np.array(mesh.points, dtype=np.float32)

                for i, cell_block in enumerate(mesh.cells):
                    cells = np.array(cell_block.data)
                    velocity_block = np.array(mesh.cell_data["U"][i], dtype=np.float32)
                    cell_centers = points[cells].mean(axis=1)

                    all_coords.append(cell_centers)
                    all_velocity.append(velocity_block)

            all_coords = np.vstack(all_coords)
            all_velocity = np.vstack(all_velocity)

            grp_t = case_grp.create_group(t)
            grp_t.create_dataset("coords", data=all_coords, compression="gzip")
            grp_t.create_dataset("velocity", data=all_velocity, compression="gzip")

            time_list.append(float(t))

        case_grp.create_dataset("time", data=np.array(time_list, dtype=np.float32))

    print(f"✅ {case_name} 已成功写入 {output_h5}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect wind field data from a case directory.")
    parser.add_argument("case_dir", help="Path to case directory")
    parser.add_argument("output_h5", help="Output HDF5 file")
    args = parser.parse_args()

    collect_case(args.case_dir, args.output_h5)


if __name__ == "__main__":
    main()
