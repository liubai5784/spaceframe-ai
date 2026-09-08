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
