"""
空间刚架有限元 —— 数据模型
============================
定义结构分析所需的核心数据结构：
节点、材料、截面、杆件、支座、荷载。

约定（重要）：
- 每个节点 6 个自由度：u, v, w, rx, ry, rz
  （3 个平动 + 3 个转动，对应全局 X, Y, Z 方向）
- 杆件局部坐标：x 轴沿杆轴线（从 i 节点指向 j 节点）；
  y、z 轴为截面的两个主轴方向（默认 z 轴尽量靠近全局 Z 轴，可用 gamma 角绕 x 轴旋转）
- 所有物理量采用国际单位：m, N, Pa, kg/m^3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 基础对象
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """节点：全局坐标 (x, y, z)，单位 m。"""
    id: int
    x: float
    y: float
    z: float

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Material:
    """材料：弹性模量 E、泊松比 nu、密度（用于自重/重量统计）。"""
    name: str
    E: float          # Pa
    nu: float = 0.3
    density: float = 7850.0   # kg/m^3，钢材默认

    @property
    def G(self) -> float:
        """剪切模量 G = E / [2(1+nu)]，Pa。"""
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class Section:
    """截面属性（基于局部坐标主轴）。

    A   : 截面积, m^2
    Iy  : 绕局部 y 轴的惯性矩, m^4
    Iz  : 绕局部 z 轴的惯性矩, m^4
    J   : 扭转常数, m^4
    Wy  : 绕局部 y 轴的截面模量, m^3
    Wz  : 绕局部 z 轴的截面模量, m^3
    """
    name: str
    A: float
    Iy: float
    Iz: float
    J: float = 0.0
    Wy: float = 0.0
    Wz: float = 0.0


@dataclass
class Member:
    """杆件单元：连接 node_i -> node_j，引用截面与材料。"""
    id: int
    node_i: int
    node_j: int
    section: str          # Section.name
    material: str         # Material.name
    gamma: float = 0.0    # 绕局部 x 轴的附加旋转角（弧度）


@dataclass
class Support:
    """支座约束：对节点 6 个自由度分别固定或释放。"""
    node: int
    fix: Tuple[bool, bool, bool, bool, bool, bool]  # u,v,w,rx,ry,rz


@dataclass
class NodalLoad:
    """节点荷载（全局坐标）：力 N，弯矩 N·m。"""
    node: int
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0


@dataclass
class MemberUniformLoad:
    """杆件均布荷载（局部坐标，单位 N/m 或 N·m/m）。

    wx : 沿局部 x 轴（轴向）
    wy : 沿局部 y 轴
    wz : 沿局部 z 轴
    """
    member: int
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0


# ---------------------------------------------------------------------------
# 结构模型
# ---------------------------------------------------------------------------

@dataclass
class FrameModel:
    """完整的空间刚架模型。"""

    nodes: Dict[int, Node] = field(default_factory=dict)
    members: Dict[int, Member] = field(default_factory=dict)
    materials: Dict[str, Material] = field(default_factory=dict)
    sections: Dict[str, Section] = field(default_factory=dict)
    supports: List[Support] = field(default_factory=list)
    nodal_loads: List[NodalLoad] = field(default_factory=list)
    member_loads: List[MemberUniformLoad] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def add_node(self, x: float, y: float, z: float) -> int:
        nid = max(self.nodes.keys(), default=0) + 1
        self.nodes[nid] = Node(nid, x, y, z)
        return nid

    def add_member(self, i: int, j: int, section: str,
                   material: str, gamma: float = 0.0) -> int:
        mid = max(self.members.keys(), default=0) + 1
        self.members[mid] = Member(mid, i, j, section, material, gamma)
        return mid

    def add_material(self, name: str, E: float, nu: float = 0.3,
                     density: float = 7850.0) -> Material:
        m = Material(name, E, nu, density)
        self.materials[name] = m
        return m

    def add_section(self, name: str, A: float, Iy: float, Iz: float,
                    J: float = 0.0, Wy: float = 0.0, Wz: float = 0.0) -> Section:
        s = Section(name, A, Iy, Iz, J, Wy, Wz)
        self.sections[name] = s
        return s

    def add_support(self, node: int,
                    fix: Tuple[bool, bool, bool, bool, bool, bool]) -> None:
        self.supports.append(Support(node, fix))

    def add_nodal_load(self, node: int, fx=0.0, fy=0.0, fz=0.0,
                       mx=0.0, my=0.0, mz=0.0) -> None:
        self.nodal_loads.append(NodalLoad(node, fx, fy, fz, mx, my, mz))

    def add_member_load(self, member: int, wx=0.0, wy=0.0, wz=0.0) -> None:
        self.member_loads.append(MemberUniformLoad(member, wx, wy, wz))

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_members(self) -> int:
        return len(self.members)

    @property
    def num_dofs(self) -> int:
        return 6 * len(self.nodes)

    def member_length(self, mid: int) -> float:
        m = self.members[mid]
        i = self.nodes[m.node_i]
        j = self.nodes[m.node_j]
        dx, dy, dz = j.x - i.x, j.y - i.y, j.z - i.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def total_weight(self) -> float:
        """结构总重（kg）：杆件体积 × 材料密度。"""
        w = 0.0
        for m in self.members.values():
            sec = self.sections[m.section]
            mat = self.materials[m.material]
            w += sec.A * self.member_length(m.id) * mat.density
        return w
