"""
URDF 连杆参数化缩放脚本 —— 技术可行性验证

演示如何通过代码精确控制 URDF 的大缩放，同时保持：
1. Visual mesh 和 collision shape 的对齐
2. Joint origin 位置自动更新
3. Inertia 参数基于物理模型合理调整

核心思路：
  - 连杆的几何体沿指定轴缩放 → geometry length 和 origin 按比例更新
  - 子关节（child joint）的 origin 按连杆长度变化等比偏移
  - 质量和惯量根据密度不变假设重新计算

适用条件：
  - 原始 URDF 中 visual 和 collision 使用相同的 geometry 和 origin（对齐前提）
  - 几何体是 URDF 原生 primitive（cylinder/box/sphere）或带 scale 属性的 mesh
  - 连杆沿单轴方向排列（如 Z 轴）

如果使用外部 STL mesh：
  - 方案 A：用 <mesh scale="sx sy sz"/> 在 URDF 中缩放（同时缩放 visual 和 collision 的 mesh）
  - 方案 B：用 trimesh 代码修改 STL 顶点坐标
  - 两种方案都能保证 visual/collision 对齐，因为它们使用相同的缩放/修改逻辑
"""

import xml.etree.ElementTree as ET
import os
import math


def compute_cylinder_inertia(mass: float, radius: float, length: float) -> dict:
    """计算圆柱体绕质心的惯量张量（主轴对角）"""
    ixx = mass * (3 * radius**2 + length**2) / 12
    iyy = ixx
    izz = mass * radius**2 / 2
    return {"ixx": ixx, "iyy": iyy, "izz": izz}


def compute_cylinder_mass(radius: float, length: float, density: float = 2700.0) -> float:
    """计算圆柱体质量（默认铝密度 2700 kg/m^3）"""
    return density * math.pi * radius**2 * length


def scale_link_along_axis(
    urdf_tree: ET.ElementTree,
    link_name: str,
    scale_factor: float,
    stretch_axis: int = 2,  # 0=x, 1=y, 2=z
    density: float = 2700.0,
) -> dict:
    """
    沿指定轴缩放连杆的几何体，同时更新：
    1. visual 和 collision 的 geometry（长度）和 origin（质心位置）
    2. inertial 的 origin、mass、inertia
    3. 以该 link 为 parent 的所有子关节的 origin

    参数:
        urdf_tree: URDF 的 ElementTree
        link_name: 要缩放的连杆名
        scale_factor: 缩放因子（>1 为拉伸，<1 为缩短）
        stretch_axis: 缩放轴 (0=x, 1=y, 2=z)
        density: 材料密度，用于重新计算质量和惯量

    返回:
        dict: 包含旧/新参数的摘要信息
    """
    root = urdf_tree.getroot()
    axis_names = ["x", "y", "z"]
    axis_name = axis_names[stretch_axis]

    # 找到目标 link
    target_link = None
    for link in root.findall("link"):
        if link.attrib.get("name") == link_name:
            target_link = link
            break
    if target_link is None:
        raise ValueError(f"Link '{link_name}' not found in URDF")

    summary = {"link": link_name, "axis": axis_name, "scale_factor": scale_factor}

    # ========== 1. 更新 visual 和 collision 的 geometry + origin ==========
    for tag in ["visual", "collision"]:
        element = target_link.find(tag)
        if element is None:
            continue

        geom = element.find("geometry")
        origin = element.find("origin")

        if geom is not None:
            cylinder = geom.find("cylinder")
            box = geom.find("box")
            mesh = geom.find("mesh")

            if cylinder is not None:
                # 圆柱体：缩放 length（假设圆柱沿 Z 轴）
                old_length = float(cylinder.attrib["length"])
                new_length = old_length * scale_factor
                radius = float(cylinder.attrib["radius"])
                cylinder.attrib["length"] = f"{new_length:.6f}"
                summary[f"{tag}_old_length"] = old_length
                summary[f"{tag}_new_length"] = new_length
                summary["radius"] = radius

            elif box is not None:
                # 盒体：缩放对应轴的尺寸
                size = [float(x) for x in box.attrib["size"].split()]
                size[stretch_axis] *= scale_factor
                box.attrib["size"] = " ".join(f"{s:.6f}" for s in size)

            elif mesh is not None:
                # 外部 mesh：通过 scale 属性缩放
                old_scale = mesh.attrib.get("scale", "1 1 1")
                scale_vals = [float(x) for x in old_scale.split()]
                scale_vals[stretch_axis] *= scale_factor
                mesh.attrib["scale"] = " ".join(f"{s:.6f}" for s in scale_vals)

        # 更新 origin：质心位置沿缩放轴按比例偏移
        if origin is not None:
            xyz = [float(x) for x in origin.attrib["xyz"].split()]
            xyz[stretch_axis] *= scale_factor
            origin.attrib["xyz"] = " ".join(f"{x:.6f}" for x in xyz)

    # ========== 2. 更新 inertial ==========
    inertial = target_link.find("inertial")
    if inertial is not None:
        # 更新 inertial origin
        inertial_origin = inertial.find("origin")
        if inertial_origin is not None:
            xyz = [float(x) for x in inertial_origin.attrib["xyz"].split()]
            xyz[stretch_axis] *= scale_factor
            inertial_origin.attrib["xyz"] = " ".join(f"{x:.6f}" for x in xyz)

        # 获取当前几何参数来重新计算质量和惯量
        geom = target_link.find("visual/geometry")
        if geom is not None:
            cylinder = geom.find("cylinder")
            if cylinder is not None:
                new_length = float(cylinder.attrib["length"])
                radius = float(cylinder.attrib["radius"])
                new_mass = compute_cylinder_mass(radius, new_length, density)
                new_inertia = compute_cylinder_inertia(new_mass, radius, new_length)

                mass_elem = inertial.find("mass")
                old_mass = float(mass_elem.attrib["value"])
                mass_elem.attrib["value"] = f"{new_mass:.6f}"

                inertia_elem = inertial.find("inertia")
                for key in ["ixx", "iyy", "izz"]:
                    inertia_elem.attrib[key] = f"{new_inertia[key]:.8f}"
                # 交叉项保持为 0（对称几何体假设）
                for key in ["ixy", "ixz", "iyz"]:
                    inertia_elem.attrib[key] = "0"

                summary["old_mass"] = old_mass
                summary["new_mass"] = new_mass
                summary["new_inertia"] = new_inertia

    # ========== 3. 更新子关节的 origin ==========
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.attrib.get("link") == link_name:
            joint_origin = joint.find("origin")
            if joint_origin is not None:
                xyz = [float(x) for x in joint_origin.attrib["xyz"].split()]
                old_xyz = xyz.copy()
                xyz[stretch_axis] *= scale_factor
                joint_origin.attrib["xyz"] = " ".join(f"{x:.6f}" for x in xyz)
                summary[f"child_joint_{joint.attrib['name']}_old_origin"] = old_xyz
                summary[f"child_joint_{joint.attrib['name']}_new_origin"] = xyz

    return summary


def verify_alignment(urdf_tree: ET.ElementTree) -> list:
    """
    验证 URDF 中所有 link 的 visual 和 collision 是否对齐。

    检查项：
    1. geometry 类型和参数是否一致
    2. origin xyz 和 rpy 是否一致
    """
    root = urdf_tree.getroot()
    results = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "unknown")
        visual = link.find("visual")
        collision = link.find("collision")

        if visual is None or collision is None:
            continue

        # 检查 origin 对齐（数值比较）
        v_origin = visual.find("origin")
        c_origin = collision.find("origin")
        if v_origin is not None and c_origin is not None:
            v_xyz = [float(x) for x in v_origin.attrib.get("xyz", "0 0 0").split()]
            c_xyz = [float(x) for x in c_origin.attrib.get("xyz", "0 0 0").split()]
            v_rpy = [float(x) for x in v_origin.attrib.get("rpy", "0 0 0").split()]
            c_rpy = [float(x) for x in c_origin.attrib.get("rpy", "0 0 0").split()]
            origin_aligned = (
                all(abs(a - b) < 1e-9 for a, b in zip(v_xyz, c_xyz))
                and all(abs(a - b) < 1e-9 for a, b in zip(v_rpy, c_rpy))
            )
        else:
            origin_aligned = (v_origin is None and c_origin is None)

        # 检查 geometry 对齐
        v_geom = visual.find("geometry")
        c_geom = collision.find("geometry")
        geom_aligned = False
        if v_geom is not None and c_geom is not None:
            # 比较几何参数（数值比较，容忍浮点精度差异）
            geom_aligned = True
            for prim_type in ["cylinder", "box", "sphere", "mesh"]:
                v_prim = v_geom.find(prim_type)
                c_prim = c_geom.find(prim_type)
                if (v_prim is None) != (c_prim is None):
                    geom_aligned = False
                    break
                if v_prim is not None and c_prim is not None:
                    for attr in v_prim.attrib:
                        v_val = v_prim.attrib.get(attr, "")
                        c_val = c_prim.attrib.get(attr, "")
                        try:
                            # 数值型属性用浮点比较
                            v_nums = [float(x) for x in v_val.split()]
                            c_nums = [float(x) for x in c_val.split()]
                            if len(v_nums) != len(c_nums):
                                geom_aligned = False
                            elif any(abs(a - b) > 1e-9 for a, b in zip(v_nums, c_nums)):
                                geom_aligned = False
                        except ValueError:
                            # 字符串属性用精确匹配
                            if v_val != c_val:
                                geom_aligned = False

        results.append({
            "link": link_name,
            "origin_aligned": origin_aligned,
            "geometry_aligned": geom_aligned,
            "overall": origin_aligned and geom_aligned,
        })

    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "original_2r.urdf")
    output_path = os.path.join(script_dir, "scaled_2r.urdf")

    # 加载原始 URDF
    tree = ET.parse(input_path)

    print("=" * 60)
    print("原始 URDF 对齐验证")
    print("=" * 60)
    alignment = verify_alignment(tree)
    for item in alignment:
        status = "✓ 对齐" if item["overall"] else "✗ 未对齐"
        print(f"  {item['link']}: {status}")

    # 缩放 link1：沿 Z 轴拉伸 1.5 倍
    print("\n" + "=" * 60)
    print("缩放 link1：沿 Z 轴拉伸 1.5 倍")
    print("=" * 60)
    summary = scale_link_along_axis(tree, "link1", scale_factor=1.5, stretch_axis=2)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.8f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("缩放后 URDF 对齐验证")
    print("=" * 60)
    alignment = verify_alignment(tree)
    for item in alignment:
        status = "✓ 对齐" if item["overall"] else "✗ 未对齐"
        print(f"  {item['link']}: {status}")

    # 保存
    tree.write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"\n已保存缩放后的 URDF: {output_path}")

    # 打印关键对比
    print("\n" + "=" * 60)
    print("参数对比")
    print("=" * 60)
    print(f"  link1 长度:  0.3m  →  {0.3 * 1.5:.3f}m  (×1.5)")
    print(f"  link1 质心:  z=0.15  →  z={0.15 * 1.5:.4f}  (随长度等比)")
    print(f"  link1 质量:  {summary.get('old_mass', 0):.4f} kg  →  {summary.get('new_mass', 0):.4f} kg  (密度不变)")
    print(f"  joint2 位置: z=0.3  →  z={0.3 * 1.5:.3f}  (在 link1 末端)")


if __name__ == "__main__":
    main()
