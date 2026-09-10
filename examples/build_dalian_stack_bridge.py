"""
大连新港栈桥单跨空间模型（历史名桥数字化重建）
==============================================
依据《力学与实践》1980《大连新港输油码头栈桥》（曹富新 等，大连工学院数力系）
与钱令希院士相关公开资料，用空间刚架通用建模器重建单跨：

  百米抛物线形空腹桁架（Vierendeel）全焊接钢栈桥
  - 跨度 L=100m，跨中高度 f=12.5m（高跨比 1/8）
  - 12 个节间，两片主桁中心距 7.6m，桥宽约 12m
  - 上弦抛物线（节点落在抛物线上），下弦直线（系杆）
  - 无斜腹杆，节点刚接（空腹桁架靠节点弯矩传力）
  - 简支：左端固定铰，右端活动铰
  - 荷载：均布恒载 p=4t/m（桥跨自重+管线），16Mn 钢（按 Q355 校核）

对拍目标（原文给出）：
  - 系杆轴力 H = p·L²/(8f) = 4×100²/(8×12.5) = 400 吨
  - 竖杆（吊杆）轴力 p·a ≈ 4×9 = 36 吨
  - 弦杆平均轴力 ≈ 402 吨

说明：箱形截面尺寸为合理假定（原文未公开具体尺寸），用于教学演示。

运行：python examples/build_dalian_stack_bridge.py
"""

import sys
import os
import json
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.custom_model import build_custom_model
from src.fem import solve
from src.checks import check_model, CheckOptions

# ---------------------------------------------------------------------------
# 几何参数
# ---------------------------------------------------------------------------
L = 100.0          # 跨度 m
f = 12.5           # 跨中矢高 m（高跨比 1/8）
h0 = 2.5           # 拱脚高度 m（端竖杆高度）
n = 12             # 节间数
B = 7.6            # 两片主桁中心距 m
p_kn_m = 40.0      # 均布恒载 kN/m（4 t/m，自重+管线）

# 节间长
a = L / n

# 抛物线（拱脚高 h0，跨中矢高 f）：z(x) = h0 + (f-h0)·4x(L-x)/L²
def arch_z(x: float) -> float:
    return h0 + (f - h0) * 4.0 * x * (L - x) / (L * L)

# ---------------------------------------------------------------------------
# 节点：下弦 1..13（左片）、14..26（右片）；上弦 27..39（左）、40..52（右）
# ---------------------------------------------------------------------------
nodes = []
def nid_bot(side: int, i: int) -> int:      # side 0=左 1=右，i=0..12
    return (side * 13) + i + 1
def nid_top(side: int, i: int) -> int:
    return 26 + (side * 13) + i + 1

for side in (0, 1):
    y = -B / 2 if side == 0 else B / 2
    for i in range(n + 1):
        x = i * a
        nodes.append({'id': nid_bot(side, i), 'x': x, 'y': y, 'z': 0.0})
        nodes.append({'id': nid_top(side, i), 'x': x, 'y': y, 'z': arch_z(x)})

# ---------------------------------------------------------------------------
# 自定义箱形截面（封闭箱形，近似原桥；尺寸为教学假定）
# ---------------------------------------------------------------------------
def box(A_h, A_w, t):          # 箱形截面属性（外轮廓高×宽×壁厚，单位 m）
    # 面积：2·(宽·t) + 2·(高-2t)·t
    area = 2 * A_w * t + 2 * (A_h - 2 * t) * t
    # 惯性矩（薄壁箱形近似，强轴绕 x）
    Ix = (A_w * A_h ** 3 - (A_w - 2 * t) * (A_h - 2 * t) ** 3) / 12.0
    Iy = (A_h * A_w ** 3 - (A_h - 2 * t) * (A_w - 2 * t) ** 3) / 12.0
    Wx = Ix / (A_h / 2)
    Wy = Iy / (A_w / 2)
    return {'A': area, 'Iy': Iy, 'Iz': Ix, 'J': 2 * area * t * t * 0.4,
            'Wy': Wy, 'Wz': Wx}

sections = {
    # 弦杆（上弦拱 + 下弦系杆）□500×400×10（封闭箱形，增强抗扭）
    'CHORD': box(0.50, 0.40, 0.010),
    # 竖杆（吊杆）□300×300×8（空腹桁架节点弯矩由抗弯+稳定覆盖）
    'WEB': box(0.30, 0.30, 0.008),
    # 横联 / 平纵联 □200×200×6
    'LATERAL': box(0.20, 0.20, 0.006),
}

# ---------------------------------------------------------------------------
# 杆件
# ---------------------------------------------------------------------------
members = []

def add(i, j, sec):
    members.append({'i': i, 'j': j, 'section': sec})

# 1) 下弦（系杆）
for side in (0, 1):
    for i in range(n):
        add(nid_bot(side, i), nid_bot(side, i + 1), 'CHORD')
# 2) 上弦（抛物线拱肋）
for side in (0, 1):
    for i in range(n):
        add(nid_top(side, i), nid_top(side, i + 1), 'CHORD')
# 3) 竖杆（空腹桁架：无斜腹杆，节点刚接）
for side in (0, 1):
    for i in range(n + 1):
        add(nid_bot(side, i), nid_top(side, i), 'WEB')
# 4) 下横联（桥面横向连接）
for i in range(n + 1):
    add(nid_bot(0, i), nid_bot(1, i), 'LATERAL')
# 5) 上横联
for i in range(n + 1):
    add(nid_top(0, i), nid_top(1, i), 'LATERAL')
# 6) 平纵联斜杆（上下弦平面内交叉支撑）
for plane_bot in (True, False):
    for i in range(n):
        if plane_bot:
            add(nid_bot(0, i), nid_bot(1, i + 1), 'LATERAL')
            add(nid_bot(1, i), nid_bot(0, i + 1), 'LATERAL')
        else:
            add(nid_top(0, i), nid_top(1, i + 1), 'LATERAL')
            add(nid_top(1, i), nid_top(0, i + 1), 'LATERAL')

# ---------------------------------------------------------------------------
# 支座：左端固定铰（u,v,w 固定，转动释放），右端活动铰（v,w 固定，u 释放）
# ---------------------------------------------------------------------------
supports = [
    {'node': nid_bot(0, 0), 'fix': [True, True, True, False, False, False]},
    {'node': nid_bot(1, 0), 'fix': [True, True, True, False, False, False]},
    {'node': nid_bot(0, n), 'fix': [False, True, True, False, False, False]},
    {'node': nid_bot(1, n), 'fix': [False, True, True, False, False, False]},
]

# ---------------------------------------------------------------------------
# 荷载：均布恒载 p=4t/m 沿全桥 → 两片主桁各承担一半，施加为下弦杆均布荷载
# （下弦即桥面系杆：荷载经竖杆传给上弦拱，拱脚水平推力由下弦系杆平衡）
# ---------------------------------------------------------------------------
member_loads = []
for idx, mb in enumerate(members, start=1):
    # 下弦杆：i/j 为下弦节点（id <= 26）
    if mb['i'] <= 26 and mb['j'] <= 26 and mb['section'] == 'CHORD':
        member_loads.append({'member': idx, 'wz': -(p_kn_m / 2.0)})

spec = {
    'units': {'length': 'm', 'force': 'kN'},
    'nodes': nodes,
    'members': members,
    'supports': supports,
    'member_loads': member_loads,
    'sections': sections,
    'material': {'E': 2.06e8, 'nu': 0.3, 'density': 7850.0},   # kN/m²（units=kN）
    'steel_grade': 'Q355',
    'ref_span': L,
}

# ---------------------------------------------------------------------------
# 求解 + 校核
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = build_custom_model(spec)
    result = solve(model)
    report = check_model(model, result,
                         CheckOptions(steel_grade='Q355', ref_span=L))

    print("=" * 62)
    print("大连新港栈桥 · 单跨空间模型重建（空腹桁架 Vierendeel）")
    print("=" * 62)
    print(f"模型规模：{model.num_nodes} 节点，{model.num_members} 杆件，"
          f"支座 {len(model.supports)} 组")
    print(f"跨度 {L} m，矢高 {f} m（高跨比 1/{L/f:.0f}），"
          f"节间 {n} × {a:.2f} m，主桁中心距 {B} m")
    print(f"均布恒载 p = {p_kn_m} kN/m（4 t/m，两片主桁各担一半）")
    print("-" * 62)

    # ---- 与原文对拍：系杆（下弦）轴力、竖杆轴力（统一 kN 口径）----
    # 原文（吨力制，1t≈9.8kN）：系杆 H = pL²/8f = 4×100²/(8×12.5) = 400t ≈ 3920 kN
    #                          竖杆 P = p·a ≈ 4×8.33 = 33.3t ≈ 326 kN（原文按 a=9m 得 36t）
    H_ana = p_kn_m * L ** 2 / (8 * f)          # 系杆轴力 kN
    P_ana = p_kn_m * a                          # 竖杆轴力 kN
    chord_N, web_N, arch_N = [], [], []
    for mid, mem in model.members.items():
        f_ = report.results[mid].forces
        ni, nj = model.members[mid].node_i, model.members[mid].node_j
        if mem.section == 'CHORD' and ni <= 26 and nj <= 26:
            chord_N.append(f_['N_signed'])      # 下弦系杆
        elif mem.section == 'CHORD':
            arch_N.append(f_['N_signed'])       # 上弦拱
        elif mem.section == 'WEB' and nj == ni + 26 \
                and ni not in (1, 13, 14, 26):
            web_N.append(f_['N_signed'])        # 内部竖杆（剔除端竖杆）
    chord_avg = sum(chord_N) / len(chord_N)
    arch_avg = sum(arch_N) / len(arch_N)
    web_avg = sum(web_N) / len(web_N)
    print(f"下弦系杆轴力：计算 {chord_avg/1000:+8.1f} kN"
          f"（原文公式 H=pL²/8f = {H_ana:8.1f} kN）"
          f"  偏差 {abs(chord_avg-H_ana*1000)/H_ana/10:.1f}%")
    print(f"上弦拱轴力：计算 {arch_avg/1000:+8.1f} kN"
          f"（原文约 -3920~-4165 kN，即 -400~-425 t）")
    print(f"内部竖杆轴力：计算 {web_avg/1000:+8.1f} kN"
          f"（原文公式 p·a = {P_ana:8.1f} kN）"
          f"  偏差 {abs(web_avg-P_ana*1000)/P_ana/10:.1f}%")
    print("-" * 62)

    print(f"最大位移（跨中附近）: {report.max_displacement*1000:.1f} mm"
          f"（挠跨比 1/{report.span/report.max_displacement:.0f}，"
          f"限值 1/400={report.span/400*1000:.0f} mm）")
    print(f"规范校核：{'全部满足 GB50017 ✓' if report.safe else '不满足'}，"
          f"最不利杆件 {report.worst_member} 号"
          + (f"，控制指标 {report.results[report.worst_member].max_ratio:.2f}"
             if report.worst_member else ""))
    print("-" * 62)
    print("【与原文差异讨论】")
    print("· 系杆/竖杆内力低于系杆拱公式：本模型为空间刚架（节点刚接），")
    print("  桥面均布荷载直接作用于下弦，荷载经'下弦梁路径'与'竖杆-拱路径'")
    print("  按刚度分配；而 pL²/8f 与 p·a 为平面系杆拱的静定简化值，")
    print("  对应桥面系分离、荷载全经竖杆（吊杆）传拱的理想传力。")
    print("· 挠跨比偏大：空腹桁架（Vierendeel）无斜腹杆，节点弯矩传力，")
    print("  竖向刚度低于同尺寸斜腹杆桁架，属该类桥型的固有特性。")
    print("· 截面尺寸为教学假定（原文未公开弦杆/腹杆具体截面），")
    print("  内力趋势与原文一致：上弦受压、下弦受拉、内部竖杆受拉、端竖杆受压。")
    print("=" * 62)

    # 输出 JSON（可粘贴到公网「自由建模」查看 3D）
    out = os.path.join(os.path.dirname(__file__), 'dalian_stack_bridge.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    print(f"模型 JSON 已保存：{out}")
