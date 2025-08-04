import os
import glob
import h5py
import meshio
import numpy as np
from collections import defaultdict

# ==== 配置 ====
case_dir = "/home/baoheng/OpenFOAM_study/UB_wind_tunnel_simulation_LES/0_test"
output_h5 = "/home/baoheng/OpenFOAM_study/UB_wind_tunnel_simulation_LES/0_test/windfield_all_cases.h5"  # 总数据库文件
case_name = os.path.basename(case_dir)   # 用目录名作为 case ID，例如 "0_test"

# ==== 1. 收集所有 VTK 文件 ====
all_vtk = sorted(glob.glob(os.path.join(case_dir, "processor*/VTK/*.vtk")))
if not all_vtk:
    raise FileNotFoundError("未找到任何 processorN/VTK/processorN_*.vtk 文件！")

# ==== 2. 按时间步组织文件 ====
files_by_time = defaultdict(list)
for f in all_vtk:
    time_str = f.split("_")[-1].replace(".vtk", "")
    files_by_time[time_str].append(f)

# ==== 3. 处理每个时间步 ====
with h5py.File(output_h5, "a") as h5f:   # "a" → 追加模式
    # 如果 case 已存在，则删除重建
    if case_name in h5f:
        print(f"⚠️  覆盖已有 case: {case_name}")
        del h5f[case_name]

    case_grp = h5f.create_group(case_name)  # 创建 case 分组
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

        # 每个时间步存入 case_name/t
        grp_t = case_grp.create_group(t)
        grp_t.create_dataset("coords", data=all_coords, compression="gzip")
        grp_t.create_dataset("velocity", data=all_velocity, compression="gzip")

        time_list.append(float(t))

    # 保存时间序列
    case_grp.create_dataset("time", data=np.array(time_list, dtype=np.float32))

print(f"✅ {case_name} 已成功写入 {output_h5}")
