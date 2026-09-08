"""
有限元内核验证算例
==================
用经典解析解对拍，验证求解器精度：

1. 悬臂梁端部集中力  ->  δ = PL³/(3EI)        （弯曲）
2. 悬臂梁端部弯矩    ->  θ = ML/(EI)           （弯曲转角）
3. 简支梁跨中集中力  ->  δ = PL³/(48EI)        （弯曲）
4. 简支梁均布荷载    ->  δ = 5wL⁴/(384EI)      （弯曲 + 等效节点荷载）
5. 轴向拉杆          ->  δ = PL/(EA)           （轴向）
6. 纯扭杆            ->  θ = TL/(GJ)           （扭转）
7. 平面刚架平衡      ->  反力与荷载自平衡        （整体平衡）

运行：python examples/verify_solver.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.model import FrameModel
from src.fem import solve, member_internal_forces


def build_material_section(model: FrameModel):
    model.add_material("steel", E=2.0e11, nu=0.3)
    # 矩形截面 0.1m x 0.2m（z 方向 0.2 为主轴方向）
    model.add_section("rect", A=0.02, Iy=1.0e-5, Iz=4.0e-5, J=2.0e-6)


def test_cantilever_tip_load():
    """悬臂梁端部集中力：δ = PL³/(3EI_z)。"""
    m = FrameModel()
    build_material_section(m)
    m.add_node(0, 0, 0)
    m.add_node(4, 0, 0)                    # 沿 x 轴，L=4m
    m.add_member(1, 2, "rect", "steel")
    m.add_support(1, (True,) * 6)          # 固端
    m.add_nodal_load(2, fy=-10e3)          # 端部 -y 方向 10kN

    r = solve(m)
    d = r.node_displacement(2)
    delta_num = d[1]                        # v 方向位移

    P, L, E, I = 10e3, 4.0, 2.0e11, 4.0e-5
    # 荷载为 -y 方向，位移亦为 -y
    delta_ana = -P * L ** 3 / (3 * E * I)
    theta_ana = -P * L ** 2 / (2 * E * I)

    err = abs(delta_num - delta_ana) / abs(delta_ana) * 100
    print(f"[1] 悬臂梁端部集中力")
    print(f"    解析 δ = {delta_ana*1000:.6f} mm, 数值 δ = {delta_num*1000:.6f} mm, 误差 {err:.4f}%")
    print(f"    解析 θ = {theta_ana:.6f} rad, 数值 θ = {d[5]:.6f} rad")
    assert err < 1e-6, "悬臂梁位移误差过大"


def test_cantilever_tip_moment():
    """悬臂梁端部弯矩：θ = ML/(EI_z)。"""
    m = FrameModel()
    build_material_section(m)
    m.add_node(0, 0, 0)
    m.add_node(4, 0, 0)
    m.add_member(1, 2, "rect", "steel")
    m.add_support(1, (True,) * 6)
    m.add_nodal_load(2, mz=50e3)           # 端部绕 z 弯矩 50 kN·m

    r = solve(m)
    d = r.node_displacement(2)
    M, L, E, I = 50e3, 4.0, 2.0e11, 4.0e-5
    theta_ana = M * L / (E * I)
    err = abs(d[5] - theta_ana) / abs(theta_ana) * 100
    print(f"[2] 悬臂梁端部弯矩")
    print(f"    解析 θ = {theta_ana:.6f} rad, 数值 θ = {d[5]:.6f} rad, 误差 {err:.4f}%")
    assert err < 1e-6, "悬臂梁转角误差过大"


def test_simply_supported_mid_load():
    """简支梁跨中集中力：δ = PL³/(48EI_z)。"""
    m = FrameModel()
    build_material_section(m)
    # 3 个节点、2 个单元，荷载作用在跨中
    for x in [0, 3, 6]:
        m.add_node(x, 0, 0)
    for i in range(1, 3):
        m.add_member(i, i + 1, "rect", "steel")
    # 简支：两端固定平动，释放转动；左端固定扭转以消除绕轴刚体位移
    m.add_support(1, (True, True, True, True, False, False))
    m.add_support(3, (True, True, True, False, False, False))
    m.add_nodal_load(2, fy=-20e3)          # 跨中 -y 方向 20kN

    r = solve(m)
    d = r.node_displacement(2)
    P, L, E, I = 20e3, 6.0, 2.0e11, 4.0e-5
    delta_ana = -P * L ** 3 / (48 * E * I)     # 荷载 -y，位移 -y
    err = abs(d[1] - delta_ana) / abs(delta_ana) * 100
    print(f"[3] 简支梁跨中集中力")
    print(f"    解析 δ = {delta_ana*1000:.6f} mm, 数值 δ = {d[1]*1000:.6f} mm, 误差 {err:.4f}%")
    assert err < 1e-4, "简支梁位移误差过大"


def test_simply_supported_uniform_load():
    """简支梁均布荷载：δ = 5wL⁴/(384EI_z)（验证等效节点荷载）。"""
    m = FrameModel()
    build_material_section(m)
    n_el = 8
    L = 6.0
    for k in range(n_el + 1):
        m.add_node(k * L / n_el, 0, 0)
    for i in range(1, n_el + 1):
        m.add_member(i, i + 1, "rect", "steel")
    m.add_support(1, (True, True, True, True, False, False))
    m.add_support(n_el + 1, (True, True, True, False, False, False))
    # 全梁均布荷载 -y 方向 1000 N/m（施加在每个单元上）
    for mid in m.members:
        m.add_member_load(mid, wy=-1000.0)

    r = solve(m)
    mid_node = n_el // 2 + 1
    d = r.node_displacement(mid_node)
    w, E, I = 1000.0, 2.0e11, 4.0e-5
    delta_ana = -5 * w * L ** 4 / (384 * E * I)   # 均布荷载 -y，位移 -y
    err = abs(d[1] - delta_ana) / abs(delta_ana) * 100
    print(f"[4] 简支梁均布荷载")
    print(f"    解析 δ = {delta_ana*1000:.6f} mm, 数值 δ = {d[1]*1000:.6f} mm, 误差 {err:.4f}%")
    assert err < 0.1, "简支梁均布荷载位移误差过大"


def test_axial_bar():
    """轴向拉杆：δ = PL/(EA)。"""
    m = FrameModel()
    build_material_section(m)
    m.add_node(0, 0, 0)
    m.add_node(2, 0, 0)
    m.add_member(1, 2, "rect", "steel")
    m.add_support(1, (True,) * 6)
    m.add_nodal_load(2, fx=100e3)          # +x 方向 100kN

    r = solve(m)
    d = r.node_displacement(2)
    P, L, E, A = 100e3, 2.0, 2.0e11, 0.02
    delta_ana = P * L / (E * A)
    err = abs(d[0] - delta_ana) / abs(delta_ana) * 100
    print(f"[5] 轴向拉杆")
    print(f"    解析 δ = {delta_ana*1000:.6f} mm, 数值 δ = {d[0]*1000:.6f} mm, 误差 {err:.4f}%")
    assert err < 1e-6, "轴向拉杆位移误差过大"


def test_torsion_bar():
    """纯扭杆：θ = TL/(GJ)。"""
    m = FrameModel()
    build_material_section(m)
    m.add_node(0, 0, 0)
    m.add_node(2, 0, 0)
    m.add_member(1, 2, "rect", "steel")
    m.add_support(1, (True,) * 6)
    m.add_nodal_load(2, mx=10e3)           # 绕 x 扭矩 10 kN·m

    r = solve(m)
    d = r.node_displacement(2)
    T, L, G, J = 10e3, 2.0, 2.0e11 / (2 * 1.3), 2.0e-6
    theta_ana = T * L / (G * J)
    err = abs(d[3] - theta_ana) / abs(theta_ana) * 100
    print(f"[6] 纯扭杆")
    print(f"    解析 θ = {theta_ana:.6f} rad, 数值 θ = {d[3]:.6f} rad, 误差 {err:.4f}%")
    assert err < 1e-6, "扭转位移误差过大"


def test_equilibrium():
    """平面门式刚架整体平衡：总反力 + 总荷载（含等效节点荷载）= 0。"""
    m = FrameModel()
    build_material_section(m)
    # 门式刚架：两柱 + 一梁
    m.add_node(0, 0, 0)      # 1 左下
    m.add_node(0, 3, 0)      # 2 左上
    m.add_node(6, 3, 0)      # 3 右上
    m.add_node(6, 0, 0)      # 4 右下
    m.add_member(1, 2, "rect", "steel")   # 左柱
    m.add_member(2, 3, "rect", "steel")   # 梁
    m.add_member(3, 4, "rect", "steel")   # 右柱
    m.add_support(1, (True,) * 6)
    m.add_support(4, (True,) * 6)
    # 水平力 + 竖向力 + 梁上均布荷载
    m.add_nodal_load(2, fx=20e3, fy=-30e3)
    m.add_nodal_load(3, fy=-30e3)
    m.add_member_load(2, wy=-5000.0)      # 梁均布 5kN/m

    r = solve(m)
    reac = r.support_reactions()

    # 严格整体平衡：绕原点取矩
    #   力平衡：ΣR + ΣF = 0（F 含等效节点荷载）
    #   矩平衡：Σ(r×R + M_R) + Σ(r×F + M_F) = 0（绕原点）
    from src.fem import assemble_load_vector, equivalent_nodal_loads
    from src.model import MemberUniformLoad

    def vec_moment(r_, f_):
        """r × f 的力矩向量。"""
        return np.cross(r_, np.array(f_[:3]))

    total_R = np.zeros(6)
    mom_R = np.zeros(3)
    for nd, fx, fy, fz, mx, my, mz in reac:
        total_R += [fx, fy, fz, mx, my, mz]
        mom_R += vec_moment(m.nodes[nd].coord, [fx, fy, fz]) + [mx, my, mz]

    total_F = np.zeros(6)
    mom_F = np.zeros(3)
    for nl in m.nodal_loads:
        f = [nl.fx, nl.fy, nl.fz, nl.mx, nl.my, nl.mz]
        total_F += f
        mom_F += vec_moment(m.nodes[nl.node].coord, f)

    for mem in m.members.values():
        eq = equivalent_nodal_loads(m, mem)          # 12 维全局等效节点荷载
        for k, nd in enumerate((mem.node_i, mem.node_j)):
            f6 = eq[k*6:k*6+6]
            total_F += f6
            mom_F += vec_moment(m.nodes[nd].coord, f6)

    print(f"[7] 平面刚架平衡检查")
    print(f"    力平衡残差 ΣR+ΣF = {total_R + total_F}")
    print(f"    力矩平衡残差     = {mom_R + mom_F}")
    assert np.allclose(total_R[:3], -total_F[:3], atol=1e-3), "刚架力不平衡"
    assert np.allclose(mom_R, -mom_F, atol=1e-3), "刚架力矩不平衡"
    print("    力与力矩平衡均通过 ✓")


if __name__ == "__main__":
    tests = [
        test_cantilever_tip_load,
        test_cantilever_tip_moment,
        test_simply_supported_mid_load,
        test_simply_supported_uniform_load,
        test_axial_bar,
        test_torsion_bar,
        test_equilibrium,
    ]
    for t in tests:
        t()
        print("-" * 50)
    print("\n全部验证算例通过 ✔")
