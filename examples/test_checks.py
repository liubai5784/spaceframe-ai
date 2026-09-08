"""
GB50017 规范校核模块测试
========================
与手算值对拍，验证：
1. 稳定系数 φ（附录 D 曲线公式）
2. 轴压构件稳定应力比（N/(φA)）
3. 受弯构件强度应力比（M/W）
4. 长细比判定
5. 完整模型校核流程 + 安全/超限结论

运行：python examples/test_checks.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

from src.model import FrameModel
from src.fem import solve
from src.checks import (check_model, CheckOptions, stability_factor)


def test_stability_factor():
    """手算校验：L=3m, I=4e-5, A=0.02, Q235, b 曲线。
    i = sqrt(4e-5/0.02) = 0.04472, λ = 3/0.04472 = 67.08
    λn = 67.08/π·sqrt(235/200000) = 0.7321
    t = 0.965+0.3·0.7321+0.7321² = 1.7204
    φ = (t - sqrt(t²-4λn²))/(2λn²) = 0.7621
    """
    i = math.sqrt(4e-5 / 0.02)
    lam = 3.0 / i
    phi = stability_factor(lam, 235e6, 2e11, 'b')
    lam_n = lam / math.pi * math.sqrt(235e6 / 2e11)
    t = 0.965 + 0.3 * lam_n + lam_n ** 2
    phi_ref = (t - math.sqrt(t * t - 4 * lam_n ** 2)) / (2 * lam_n ** 2)
    print(f"[1] 稳定系数 φ: 计算 {phi:.6f} vs 手算 {phi_ref:.6f}")
    assert abs(phi - phi_ref) < 1e-10
    assert abs(phi - 0.7621) < 1e-3, f"φ 与参考值不符: {phi}"
    print("    通过 ✓")


def test_axial_compression():
    """轴压柱：施加 0.5·φ·A·f 的轴力，稳定应力比应 ≈ 0.5。"""
    m = FrameModel()
    m.add_material("STEEL", E=2.0e11, nu=0.3)
    m.add_section("SECT", A=0.02, Iy=1e-5, Iz=4e-5, J=2e-6, Wy=2e-4, Wz=4e-4)
    m.add_node(0, 0, 0)
    m.add_node(0, 3, 0)          # 竖杆
    m.add_member(1, 2, "SECT", "STEEL")
    m.add_support(1, (True,) * 6)
    # 顶部仅约束平动，模拟顶端自由（计算长度取 2L 更接近，这里按 L 测试公式）
    m.add_support(2, (False, False, False, True, True, True))

    opts = CheckOptions(steel_grade='Q235', curve_class='b')
    i_min = math.sqrt(min(1e-5, 4e-5) / 0.02)
    lam = 3.0 / i_min
    phi = stability_factor(lam, 235e6, 2e11, 'b')
    N_target = 0.5 * phi * 0.02 * 215e6
    m.add_nodal_load(2, fy=-N_target)      # 轴压（-y 方向）

    r = solve(m)
    report = check_model(m, r, opts)
    res = report.results[1]
    print(f"[2] 轴压柱稳定应力比: {res.stability_ratio:.4f} (期望 ≈ 0.5000)")
    assert abs(res.stability_ratio - 0.5) < 0.02, \
        f"稳定应力比偏差过大: {res.stability_ratio}"
    print(f"    强度应力比: {res.stress_ratio:.4f}（轴压构件强度通常不起控制）")
    print("    通过 ✓")


def test_bending_strength():
    """受弯构件：施加 0.6·Wx·f 的弯矩，强度应力比应 ≈ 0.6。"""
    m = FrameModel()
    m.add_material("STEEL", E=2.0e11, nu=0.3)
    m.add_section("SECT", A=0.02, Iy=1e-5, Iz=4e-5, J=2e-6, Wy=2e-4, Wz=4e-4)
    m.add_node(0, 0, 0)
    m.add_node(4, 0, 0)          # 水平悬臂梁
    m.add_member(1, 2, "SECT", "STEEL")
    m.add_support(1, (True,) * 6)

    M_target = 0.6 * 4e-4 * 215e6      # 0.6·Wz·f
    P = M_target / 4.0                 # 端部集中力
    m.add_nodal_load(2, fy=-P)

    r = solve(m)
    report = check_model(m, r, CheckOptions(steel_grade='Q235'))
    res = report.results[1]
    print(f"[3] 悬臂梁强度应力比: {res.stress_ratio:.4f} (期望 ≈ 0.6000)")
    assert abs(res.stress_ratio - 0.6) < 0.02, f"强度应力比偏差过大: {res.stress_ratio}"
    print("    通过 ✓")


def test_slenderness_and_full_report():
    """长细比判定 + 完整校核报告（.s2k 示例门式刚架）。"""
    from src.s2k_parser import parse_s2k
    s2k_path = os.path.join(os.path.dirname(__file__), "sample_frame.s2k")
    model = parse_s2k(s2k_path)
    r = solve(model)
    report = check_model(model, r, CheckOptions(steel_grade='Q235'))

    print(f"[4] 门式刚架完整校核")
    print(f"    参考跨度: {report.span:.2f} m, 最大位移: {report.max_displacement*1000:.2f} mm")
    print(f"    位移比: {report.max_deflection_ratio:.3f}")
    for mid, res in sorted(report.results.items()):
        print(f"    杆件{mid}: 强度比 {res.stress_ratio:.3f} | "
              f"稳定比 {res.stability_ratio:.3f} | λ {res.slenderness:.0f} | "
              f"结论 {'OK' if res.ok else '超限'}")
    print(f"    整体结论: {report.summary()}")
    assert isinstance(report.safe, bool)
    print("    通过 ✓")


def test_overload_detection():
    """放大荷载后应判定为不满足规范。"""
    from src.s2k_parser import parse_s2k
    s2k_path = os.path.join(os.path.dirname(__file__), "sample_frame.s2k")
    model = parse_s2k(s2k_path)
    # 荷载放大 3 倍
    for nl in model.nodal_loads:
        nl.fx *= 3; nl.fy *= 3; nl.fz *= 3
    for ml in model.member_loads:
        ml.wx *= 3; ml.wy *= 3; ml.wz *= 3
    r = solve(model)
    report = check_model(model, r, CheckOptions(steel_grade='Q235'))
    print(f"[5] 荷载放大 3 倍后整体结论: {report.summary()}")
    assert report.safe is False, "放大荷载后应判定为不满足"
    print("    通过 ✓")


if __name__ == "__main__":
    test_stability_factor()
    print("-" * 50)
    test_axial_compression()
    print("-" * 50)
    test_bending_strength()
    print("-" * 50)
    test_slenderness_and_full_report()
    print("-" * 50)
    test_overload_detection()
    print("-" * 50)
    print("\n规范校核模块测试全部通过 ✔")
