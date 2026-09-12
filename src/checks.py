"""
钢结构规范校核模块（GB50017-2017）
===================================
对有限元分析结果按《钢结构设计标准》GB50017-2017 进行构件校核：

1. 强度校核：拉/压弯构件 N/A + Mx/Wx + My/Wy ≤ f （第 8.1.1 条简化）
2. 整体稳定校核：
   - 轴压构件：N/(φA) ≤ f （第 7.2.1 条）
   - 压弯构件平面内：N/(φx·A) + βmx·Mx/[γx·Wx(1-0.8N/N'Ex)] ≤ f （第 8.2.1 条）
   - 压弯构件平面外：N/(φy·A) + η·βtx·Mx/(φb·Wx) ≤ f （第 8.2.2 条）
   稳定系数 φ 按附录 D 曲线公式计算（a/b/c/d 类截面）
3. 刚度校核：
   - 长细比：受压构件 ≤ 150，受拉构件 ≤ 400（第 7.4.6 条）
   - 结构最大位移 ≤ L/400（默认，可配置，参考附录 B 挠度限值）

说明：
- 内力取沿杆件采样（含跨中）的绝对最大值
- 等效弯矩系数 β、截面塑性发展系数 γx 等采用保守取值，详见 CheckOptions
- 本模块用于课程项目教学演示，工程应用请以正式设计文件为准
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .model import FrameModel
from .fem import AnalysisResult, member_extreme_forces


# ---------------------------------------------------------------------------
# 材料（GB/T 1591 / GB50017-2017 表 4.4.1，厚度 ≤ 16mm）
# ---------------------------------------------------------------------------

STEEL_GRADES: Dict[str, dict] = {
    # 名称: {'fy': 屈服强度 Pa, 'f': 抗拉/抗压/抗弯设计强度 Pa}
    'Q235': {'fy': 235e6, 'f': 215e6},
    'Q345': {'fy': 345e6, 'f': 305e6},
    'Q355': {'fy': 355e6, 'f': 305e6},
    'Q390': {'fy': 390e6, 'f': 330e6},
    'Q420': {'fy': 420e6, 'f': 360e6},
}

# 稳定曲线系数（GB50017-2017 附录 D 表 D.0.1）
CURVE_COEFFS = {
    'a': {'alpha1': 0.41, 'alpha2': 0.986, 'alpha3': 0.152},
    'b': {'alpha1': 0.65, 'alpha2': 0.965, 'alpha3': 0.300},
    'c': {'alpha1': 0.73, 'alpha2': 0.906, 'alpha3': 0.595},
    'd': {'alpha1': 1.35, 'alpha2': 0.868, 'alpha3': 0.915},
}


def stability_factor(lam: float, fy: float, E: float, curve: str = 'b') -> float:
    """轴心受压构件整体稳定系数 φ（GB50017-2017 附录 D）。

    lam : 长细比（无量纲）
    """
    coeff = CURVE_COEFFS[curve]
    lam_n = lam / math.pi * math.sqrt(fy / E)          # 正则化长细比
    if lam_n <= 0.215:
        return 1.0 - coeff['alpha1'] * lam_n * lam_n
    a2, a3 = coeff['alpha2'], coeff['alpha3']
    t = a2 + a3 * lam_n + lam_n * lam_n
    return (t - math.sqrt(t * t - 4 * lam_n * lam_n)) / (2 * lam_n * lam_n)


# ---------------------------------------------------------------------------
# 校核选项
# ---------------------------------------------------------------------------

@dataclass
class CheckOptions:
    steel_grade: str = 'Q235'              # 钢材牌号
    curve_class: str = 'b'                 # 稳定曲线类别
    deflection_limit_denom: float = 400.0  # 挠度限值分母（L/400）
    slenderness_limit_comp: float = 150.0  # 受压构件容许长细比
    slenderness_limit_tens: float = 400.0  # 受拉构件容许长细比
    effective_length_factor: float = 1.0   # 计算长度系数 μ（默认按两端铰接）
    gamma_x: float = 1.05                  # 截面塑性发展系数 γx
    beta_mx: float = 1.0                   # 等效弯矩系数（保守取值）
    beta_tx: float = 1.0                   # 平面外等效弯矩系数（保守取值）
    eta: float = 1.0                       # 截面影响系数（保守取值）
    phi_b: float = 1.0                     # 受弯构件整体稳定系数（保守取值）
    ref_span: float | None = None          # 参考跨度（用于挠度比）；None=最长杆件


# ---------------------------------------------------------------------------
# 单根杆件校核结果
# ---------------------------------------------------------------------------

@dataclass
class MemberCheckResult:
    member_id: int
    forces: dict                              # 极值内力
    slenderness: float                        # 最大长细比
    stress_ratio: float                       # 强度应力比
    stability_ratio: float                    # 稳定应力比（取平面内/外较大）
    combined_ratio: float                     # 综合应力比 = max(强度, 稳定)
    deflection_ratio: float                   # 挠度比（≤1 满足）
    ok: bool                                  # 是否全部满足
    messages: List[str] = field(default_factory=list)   # 校核说明

    @property
    def max_ratio(self) -> float:
        return max(self.stress_ratio, self.stability_ratio, self.deflection_ratio)


# ---------------------------------------------------------------------------
# 整体报告
# ---------------------------------------------------------------------------

@dataclass
class CheckReport:
    model: FrameModel
    options: CheckOptions
    results: Dict[int, MemberCheckResult]
    max_displacement: float
    max_deflection_ratio: float
    span: float

    @property
    def safe(self) -> bool:
        return all(r.ok for r in self.results.values()) and self.max_deflection_ratio <= 1.0

    @property
    def worst_member(self) -> int | None:
        if not self.results:
            return None
        return max(self.results, key=lambda k: self.results[k].max_ratio)

    def summary(self) -> str:
        """一句话安全结论（供 Agent 使用）。"""
        if self.safe:
            return (f"结构安全：所有构件满足 GB50017-2017 校核要求，"
                    f"最大综合应力比 {max(r.max_ratio for r in self.results.values()):.3f}"
                    f"，最大位移比 {self.max_deflection_ratio:.3f}")
        worst = self.results[self.worst_member]
        return (f"结构不满足规范要求：最不利构件为 {self.worst_member} 号，"
                f"综合应力比 {worst.max_ratio:.3f}"
                f"{'（超限）' if worst.max_ratio > 1 else ''}；"
                f"最大位移比 {self.max_deflection_ratio:.3f}"
                f"{'（超限）' if self.max_deflection_ratio > 1 else ''}")


# ---------------------------------------------------------------------------
# 校核主流程
# ---------------------------------------------------------------------------

def check_model(model: FrameModel, result: AnalysisResult,
                options: CheckOptions | None = None) -> CheckReport:
    """对分析结果执行 GB50017 构件校核。"""
    options = options or CheckOptions()
    grade = STEEL_GRADES.get(options.steel_grade)
    if grade is None:
        raise ValueError(f"不支持的钢材牌号: {options.steel_grade}（可选 {list(STEEL_GRADES)}）")
    f_design = grade['f']
    fy = grade['fy']

    # 参考跨度（挠度比）
    if options.ref_span:
        span = options.ref_span
    else:
        span = max((model.member_length(m.id) for m in model.members.values()), default=1.0)
    def_limit = span / options.deflection_limit_denom

    # 结构最大位移（平动）
    U = result.U.reshape(-1, 6)[:, :3]
    max_disp = float(max(math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) for u in U)) if len(U) else 0.0
    deflection_ratio = max_disp / def_limit if def_limit > 0 else 0.0

    results: Dict[int, MemberCheckResult] = {}
    for mid, mem in model.members.items():
        sec = model.sections[mem.section]
        mat = model.materials[mem.material]
        L = model.member_length(mid)

        forces = member_extreme_forces(model, mem, result.member_forces[mid])
        N = forces['N']                    # 绝对值（强度计算用）
        Ns = forces.get('N_signed', N)     # 带符号：拉为正、压为负
        is_comp = Ns < -1e-6               # 是否受压
        Mx = forces['Mx']
        My = forces['My']
        Mz = forces['Mz']

        messages = []
        # ---------------- 长细比 ----------------
        i_min = math.sqrt(min(sec.Iy, sec.Iz) / sec.A) if sec.A > 0 else 1e-12
        lam = options.effective_length_factor * L / i_min
        sl_limit = (options.slenderness_limit_comp if is_comp
                    else options.slenderness_limit_tens)
        sl_ok = lam <= sl_limit

        # ---------------- 强度校核（8.1.1） ----------------
        Wx = max(sec.Wz, 1e-12)            # 绕局部 z（对应 SAP2000 3 轴）
        Wy = max(sec.Wy, 1e-12)
        stress = N / sec.A + Mz / Wx + My / Wy
        stress_ratio = stress / f_design

        # ---------------- 整体稳定（压弯构件，8.2.1 / 8.2.2） ----------------
        stability_ratio = 0.0
        if is_comp:                       # 仅受压构件需要稳定校核
            E = mat.E
            lam_y = options.effective_length_factor * L / math.sqrt(sec.Iy / sec.A)
            lam_z = options.effective_length_factor * L / math.sqrt(sec.Iz / sec.A)
            phi_y = stability_factor(lam_y, fy, E, options.curve_class)
            phi_z = stability_factor(lam_z, fy, E, options.curve_class)

            # 平面内（绕强轴 z 失稳）：N/(φz·A) + β·M/(γ·W·(1-0.8N/N'Ez))
            N_ez = math.pi**2 * E * sec.Iz / (1.1 * lam_z**2) if lam_z > 0 else 1e30
            term1 = N / (phi_z * sec.A)
            term2 = (options.beta_mx * Mz /
                     (options.gamma_x * Wx * max(1 - 0.8 * N / N_ez, 0.2)))
            in_plane = (term1 + term2) / f_design

            # 平面外（绕弱轴 y 失稳）：N/(φy·A) + η·β·M/(φb·Wx)
            out_plane = (N / (phi_y * sec.A)
                         + options.eta * options.beta_tx * Mz
                         / (options.phi_b * Wx)) / f_design

            stability_ratio = max(in_plane, out_plane)
            messages.append(
                f"λy={lam_y:.0f}(φy={phi_y:.3f}) λz={lam_z:.0f}(φz={phi_z:.3f})")

        combined = max(stress_ratio, stability_ratio)

        # ---------------- 判定 ----------------
        ok = (sl_ok and combined <= 1.0)
        if not sl_ok:
            messages.append(f"长细比 λ={lam:.0f} > {sl_limit:.0f}")

        results[mid] = MemberCheckResult(
            member_id=mid, forces=forces,
            slenderness=lam, stress_ratio=stress_ratio,
            stability_ratio=stability_ratio, combined_ratio=combined,
            deflection_ratio=0.0, ok=ok, messages=messages,
        )

    return CheckReport(model, options, results, max_disp,
                       deflection_ratio, span)
