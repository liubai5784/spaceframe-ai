"""
参数化空间刚架生成器
====================
按矩形网格生成规则的空间刚架结构：

        ●────●────●
       /│   /│   /│
      ●────●────● │
      │ │  │ │  │ │
      │ ●──│─●──│─●
      │/   │/   │/
      ●────●────●

- 平面网格 nx × ny 个节点，层数 nz 层（默认 1 层）
- 每层：X 向梁、Y 向梁
- 每层网格交点：竖向柱（底层柱从地面到第一层）
- 柱底默认固接（可配置）
- 顶部可施加竖向均布面荷载（换算为节点力）或杆件均布荷载
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .model import FrameModel, Node
from . import sections_db


@dataclass
class FrameParams:
    Lx: float = 6.0          # X 方向总跨度 m
    Ly: float = 4.0          # Y 方向总跨度 m
    Lz: float = 3.0          # 层高 m
    nx: int = 2              # X 向跨数
    ny: int = 2              # Y 向跨数
    nz: int = 1              # 层数
    column_section: str = 'HW200'
    beam_section: str = 'HN300'
    material: str = 'STEEL'
    E: float = 2.06e11       # 弹性模量 Pa（钢材默认 2.06e11）
    load_kn_m2: float = 0.0  # 顶部均布面荷载 kN/m²（向下为正）
    steel_grade: str = 'Q355'


def build_frame(p: FrameParams) -> FrameModel:
    """按参数生成空间刚架模型。"""
    m = FrameModel()
    m.add_material(p.material, E=p.E, nu=0.3, density=7850.0)

    col = sections_db.section_properties(p.column_section)
    beam = sections_db.section_properties(p.beam_section)
    if col is None:
        raise ValueError(f"未知柱截面: {p.column_section}")
    if beam is None:
        raise ValueError(f"未知梁截面: {p.beam_section}")
    m.add_section(col['name'], col['A'], col['Iy'], col['Iz'],
                  col['J'], col['Wy'], col['Wz'])
    m.add_section(beam['name'], beam['A'], beam['Iy'], beam['Iz'],
                  beam['J'], beam['Wy'], beam['Wz'])

    # ---- 节点编号：层 z 从 0 到 nz，网格 (ix, iy) ----
    # node_id = (iz*(nx+1) + ix)*(ny+1) + iy + 1
    def node_id(iz: int, ix: int, iy: int) -> int:
        return ((iz * (p.nx + 1) + ix) * (p.ny + 1) + iy + 1)

    dx, dy, dz = p.Lx / p.nx, p.Ly / p.ny, p.Lz / max(p.nz, 1)
    for iz in range(p.nz + 1):
        for ix in range(p.nx + 1):
            for iy in range(p.ny + 1):
                m.nodes[node_id(iz, ix, iy)] = Node(
                    node_id(iz, ix, iy), ix * dx, iy * dy, iz * dz)

    # ---- 杆件 ----
    # 柱：每层每网格交点
    for iz in range(p.nz):
        for ix in range(p.nx + 1):
            for iy in range(p.ny + 1):
                m.add_member(node_id(iz, ix, iy), node_id(iz + 1, ix, iy),
                             col['name'], p.material)
    # X 向梁：每层每行
    for iz in range(1, p.nz + 1):
        for iy in range(p.ny + 1):
            for ix in range(p.nx):
                m.add_member(node_id(iz, ix, iy), node_id(iz, ix + 1, iy),
                             beam['name'], p.material)
    # Y 向梁：每层每列
    for iz in range(1, p.nz + 1):
        for ix in range(p.nx + 1):
            for iy in range(p.ny):
                m.add_member(node_id(iz, ix, iy), node_id(iz, ix, iy + 1),
                             beam['name'], p.material)

    # ---- 支座：柱底固接 ----
    for ix in range(p.nx + 1):
        for iy in range(p.ny + 1):
            m.add_support(node_id(0, ix, iy), (True,) * 6)

    # ---- 荷载：顶部均布面荷载 -> 各顶部节点（按面积分配） ----
    if p.load_kn_m2 > 0:
        w = p.load_kn_m2 * 1e3                    # kN/m² -> N/m²
        top = p.nz
        for ix in range(p.nx + 1):
            for iy in range(p.ny + 1):
                ax = dx / 2.0 if (ix == 0 or ix == p.nx) else dx
                ay = dy / 2.0 if (iy == 0 or iy == p.ny) else dy
                m.add_nodal_load(node_id(top, ix, iy), fz=-w * ax * ay)

    return m


def node_xy_id(model: FrameModel, nid: int) -> Tuple[int, int, int]:
    """反查节点所在 (iz, ix, iy)（用于可视化/结果定位，仅适用于规则网格）。"""
    n = model.nodes[nid]
    return (round(n.z), round(n.x), round(n.y))


# ============================================================
# 空间桁架桥（下承式简支桁架桥）
# ============================================================

@dataclass
class TrussBridgeParams:
    L: float = 24.0            # 桥长 m
    W: float = 5.0             # 桥宽 m
    H: float = 3.0             # 桥高（桁高）m
    n_panels: int = 6          # 节间数（沿桥长）
    chord_section: str = 'HW200'   # 弦杆截面
    web_section: str = 'HW150'     # 腹杆/竖杆/横撑截面
    material: str = 'STEEL'
    E: float = 2.06e11
    load_kn_m2: float = 0.0    # 桥面均布荷载 kN/m²
    steel_grade: str = 'Q355'


def build_truss_bridge(p: TrussBridgeParams) -> FrameModel:
    """生成下承式空间简支桁架桥（用刚架单元等效，荷载在节点上，杆件以轴力为主）。"""
    m = FrameModel()
    m.add_material(p.material, E=p.E, nu=0.3, density=7850.0)

    chord = sections_db.section_properties(p.chord_section)
    web = sections_db.section_properties(p.web_section)
    if chord is None:
        raise ValueError(f"未知弦杆截面: {p.chord_section}")
    if web is None:
        raise ValueError(f"未知腹杆截面: {p.web_section}")
    for s in (chord, web):
        m.add_section(s['name'], s['A'], s['Iy'], s['Iz'],
                      s['J'], s['Wy'], s['Wz'])

    n = max(int(p.n_panels), 1)
    dx = p.L / n
    # ---- 节点：每截面 4 个（下左/下右/上左/上右） ----
    def nid(i, k):
        return 4 * i + k            # k: 1下左 2下右 3上左 4上右
    for i in range(n + 1):
        x = i * dx
        m.nodes[nid(i, 1)] = Node(nid(i, 1), x, -p.W / 2, 0.0)
        m.nodes[nid(i, 2)] = Node(nid(i, 2), x,  p.W / 2, 0.0)
        m.nodes[nid(i, 3)] = Node(nid(i, 3), x, -p.W / 2, p.H)
        m.nodes[nid(i, 4)] = Node(nid(i, 4), x,  p.W / 2, p.H)

    def add(a, b, sec):
        m.add_member(a, b, sec, p.material)

    for i in range(n):
        # 上下弦杆（弦杆截面）
        add(nid(i, 1), nid(i + 1, 1), chord['name'])
        add(nid(i, 2), nid(i + 1, 2), chord['name'])
        add(nid(i, 3), nid(i + 1, 3), chord['name'])
        add(nid(i, 4), nid(i + 1, 4), chord['name'])
        # 斜腹杆（按剪力方向跨中对称布置，使斜杆受拉 = Pratt 式）
        if i < (n - 1) / 2:
            add(nid(i, 3), nid(i + 1, 1), web['name'])   # 左半：\ 方向
            add(nid(i, 4), nid(i + 1, 2), web['name'])
        else:
            add(nid(i, 1), nid(i + 1, 3), web['name'])   # 右半：/ 方向
            add(nid(i, 2), nid(i + 1, 4), web['name'])
        # 平纵联斜杆（顶/底各一根，增强空间整体性；用弦杆截面保证支撑刚度）
        add(nid(i, 1), nid(i + 1, 2), chord['name'])
        add(nid(i, 3), nid(i + 1, 4), chord['name'])

    for i in range(n + 1):
        # 竖杆
        add(nid(i, 1), nid(i, 3), web['name'])
        add(nid(i, 2), nid(i, 4), web['name'])
        # 横向支撑（下/上横杆，弦杆截面）
        add(nid(i, 1), nid(i, 2), chord['name'])
        add(nid(i, 3), nid(i, 4), chord['name'])

    # ---- 简支支座：左端固定铰，右端活动铰（桁架节点释放转动） ----
    m.add_support(nid(0, 1), (True, True, True, False, False, False))
    m.add_support(nid(0, 2), (True, True, True, False, False, False))
    m.add_support(nid(n, 1), (False, True, True, False, False, False))
    m.add_support(nid(n, 2), (False, True, True, False, False, False))

    # ---- 桥面荷载（下承式 -> 下弦节点，按分担面积） ----
    if p.load_kn_m2 > 0:
        w = p.load_kn_m2 * 1e3
        for i in range(n + 1):
            ax = dx / 2.0 if (i == 0 or i == n) else dx
            fz = -w * ax * (p.W / 2.0)     # 每侧分担一半桥宽
            m.add_nodal_load(nid(i, 1), fz=fz)
            m.add_nodal_load(nid(i, 2), fz=fz)

    return m
