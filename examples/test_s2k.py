"""
.s2k 解析器测试
===============
读取示例 .s2k 文件，与手工构建的同一模型对比求解结果，
验证：节点/杆件/截面/材料/支座/荷载/单位换算全部正确。

运行：python examples/test_s2k.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.s2k_parser import parse_s2k
from src.model import FrameModel
from src.fem import solve


def build_reference_model() -> FrameModel:
    """与 sample_frame.s2k 等价的手工模型（单位 SI）。"""
    m = FrameModel()
    m.add_material("STEEL", E=2.0e11, nu=0.3, density=7850.0)
    m.add_section("SECT1", A=0.02, Iy=1e-5, Iz=4e-5, J=2e-6, Wy=2e-4, Wz=4e-4)
    m.add_node(0, 0, 0)
    m.add_node(0, 3, 0)
    m.add_node(6, 3, 0)
    m.add_node(6, 0, 0)
    m.add_member(1, 2, "SECT1", "STEEL")
    m.add_member(2, 3, "SECT1", "STEEL")
    m.add_member(3, 4, "SECT1", "STEEL")
    m.add_support(1, (True,) * 6)
    m.add_support(4, (True,) * 6)
    m.add_nodal_load(2, fx=20e3, fy=-30e3)
    m.add_nodal_load(3, fy=-30e3)
    m.add_member_load(2, wy=-5000.0)
    return m


if __name__ == "__main__":
    s2k_path = os.path.join(os.path.dirname(__file__), "sample_frame.s2k")
    model = parse_s2k(s2k_path)
    ref = build_reference_model()

    print(f"节点数: {model.num_nodes} vs 参考 {ref.num_nodes}")
    print(f"杆件数: {model.num_members} vs 参考 {ref.num_members}")
    print(f"材料:   {list(model.materials)}")
    print(f"截面:   {list(model.sections)}")
    print(f"支座数: {len(model.supports)}")
    print(f"节点荷载: {len(model.nodal_loads)} 条")
    print(f"杆件均布荷载: {len(model.member_loads)} 条")

    # 单位换算检查
    assert abs(model.materials["STEEL"].E - 2e11) < 1.0, \
        f"弹性模量换算错误: {model.materials['STEEL'].E}"
    assert abs(model.sections["SECT1"].A - 0.02) < 1e-9
    assert abs(model.materials["STEEL"].density - 7850.0) < 50, \
        f"密度换算错误: {model.materials['STEEL'].density}"

    r1 = solve(model)
    r2 = solve(ref)

    U1 = r1.U.reshape(-1, 6)
    U2 = r2.U.reshape(-1, 6)
    max_diff = np.max(np.abs(U1 - U2))
    print(f"\n解析模型与手工模型最大位移差: {max_diff:.3e} m")
    assert max_diff < 1e-8, "解析结果与参考模型不一致"

    for nid in sorted(model.nodes):
        d = r1.node_displacement(nid)
        print(f"  节点{nid}: u={d[0]*1000:9.4f}mm v={d[1]*1000:9.4f}mm "
              f"w={d[2]*1000:9.4f}mm rz={d[5]*1000:8.3f}mrad")

    print("\n.s2k 解析与求解测试通过 ✔")
