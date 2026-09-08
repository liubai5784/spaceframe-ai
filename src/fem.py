"""
空间刚架有限元内核
==================
采用经典的矩阵位移法（直接刚度法）：

1. 单元局部刚度矩阵 k_e（12×12，每节点 6 自由度）
2. 坐标变换：局部 -> 全局
3. 组装全局刚度矩阵 K
4. 处理边界条件（支座约束）
5. 求解 KU = F 得到节点位移
6. 回代求支座反力与杆端内力

单元局部自由度排列（索引 0..11）：
   [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]
    0   1   2   3    4    5    6   7   8   9   10   11

杆件局部坐标约定：
  x 轴沿杆轴线（i -> j）；z 轴尽量靠近全局 Z 轴（杆件接近竖直时用全局 Y 作参考）；
  y 轴由右手法则确定；gamma 为绕局部 x 轴的附加旋转角。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .model import FrameModel


# ---------------------------------------------------------------------------
# 1. 单元局部刚度矩阵（12×12）
# ---------------------------------------------------------------------------

def local_stiffness(L: float, E: float, G: float, A: float,
                    Iy: float, Iz: float, J: float) -> np.ndarray:
    """空间刚架单元在局部坐标系下的 12×12 刚度矩阵。"""
    L2, L3 = L * L, L * L * L
    k = np.zeros((12, 12))

    # --- 轴向 (u1, u2: 索引 0, 6) ---
    ea = E * A / L
    k[0, 0] = k[6, 6] = ea
    k[0, 6] = k[6, 0] = -ea

    # --- 扭转 (rx1, rx2: 索引 3, 9) ---
    gj = G * J / L
    k[3, 3] = k[9, 9] = gj
    k[3, 9] = k[9, 3] = -gj

    # --- 绕局部 z 轴弯曲（x-y 平面内：v1, rz1, v2, rz2 = 索引 1, 5, 7, 11）---
    eiz = E * Iz
    k[1, 1] = k[7, 7] = 12 * eiz / L3
    k[1, 5] = k[5, 1] = 6 * eiz / L2
    k[1, 7] = k[7, 1] = -12 * eiz / L3
    k[1, 11] = k[11, 1] = 6 * eiz / L2
    k[5, 5] = k[11, 11] = 4 * eiz / L
    k[5, 7] = k[7, 5] = -6 * eiz / L2
    k[5, 11] = k[11, 5] = 2 * eiz / L
    k[7, 11] = k[11, 7] = -6 * eiz / L2

    # --- 绕局部 y 轴弯曲（x-z 平面内：w1, ry1, w2, ry2 = 索引 2, 4, 8, 10）---
    eiy = E * Iy
    k[2, 2] = k[8, 8] = 12 * eiy / L3
    k[2, 4] = k[4, 2] = -6 * eiy / L2
    k[2, 8] = k[8, 2] = -12 * eiy / L3
    k[2, 10] = k[10, 2] = -6 * eiy / L2
    k[4, 4] = k[10, 10] = 4 * eiy / L
    k[4, 8] = k[8, 4] = 6 * eiy / L2
    k[4, 10] = k[10, 4] = 2 * eiy / L
    k[8, 10] = k[10, 8] = 6 * eiy / L2

    return k


# ---------------------------------------------------------------------------
# 2. 坐标变换
# ---------------------------------------------------------------------------

def rotation_matrix(model: FrameModel, member, gamma: float = 0.0) -> np.ndarray:
    """计算杆件的 3×3 方向余弦矩阵 R（局部坐标 -> 全局坐标）。

    返回 R，使得 全局向量 = R @ 局部向量；R 的列即局部基向量在全局坐标中的分量。
    """
    i = model.nodes[member.node_i]
    j = model.nodes[member.node_j]
    dx, dy, dz = j.x - i.x, j.y - i.y, j.z - i.z
    L = (dx * dx + dy * dy + dz * dz) ** 0.5
    if L < 1e-12:
        raise ValueError(f"杆件 {member.id} 长度为零")

    ex = np.array([dx, dy, dz]) / L                       # 局部 x 轴

    # 参考向量：默认全局 Z；杆件接近平行 Z 时改用全局 Y
    if abs(ex[2]) > 0.999:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    y_axis = np.cross(ref, ex)                            # 与 ex 垂直且在水平面内
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(ex, y_axis)                         # 右手系
    z_axis = z_axis / np.linalg.norm(z_axis)

    # gamma：绕局部 x 轴的附加旋转
    c, s = np.cos(gamma), np.sin(gamma)
    yg = c * y_axis + s * z_axis
    zg = -s * y_axis + c * z_axis

    return np.column_stack([ex, yg, zg])


def transformation_matrix(R: np.ndarray) -> np.ndarray:
    """由 3×3 方向余弦矩阵生成 12×12 变换矩阵 T = diag(R, R, R, R)。"""
    T = np.zeros((12, 12))
    for b in range(4):
        T[b*3:b*3+3, b*3:b*3+3] = R
    return T


def global_stiffness(model: FrameModel, member) -> np.ndarray:
    """杆件在全局坐标系下的 12×12 刚度矩阵 k_g = T k_l Tᵀ。"""
    sec = model.sections[member.section]
    mat = model.materials[member.material]
    L = model.member_length(member.id)
    k_l = local_stiffness(L, mat.E, mat.G, sec.A, sec.Iy, sec.Iz, sec.J)
    R = rotation_matrix(model, member, member.gamma)
    T = transformation_matrix(R)
    return T @ k_l @ T.T


# ---------------------------------------------------------------------------
# 3. 杆件均布荷载
# ---------------------------------------------------------------------------

def _member_uniform_loads(model: FrameModel, member) -> Tuple[float, float, float]:
    """返回杆件的累计均布荷载 (wx, wy, wz)（局部坐标，N/m）。"""
    wx = wy = wz = 0.0
    for ml in model.member_loads:
        if ml.member == member.id:
            wx += ml.wx
            wy += ml.wy
            wz += ml.wz
    return wx, wy, wz


def fixed_end_forces(model: FrameModel, member) -> np.ndarray:
    """杆件均布荷载在两端固支时产生的固端力向量 f⁰（局部坐标，12 维）。

    f⁰ 定义为“节点对杆件施加的杆端力”（即等效节点荷载的相反数），
    用于叠加到杆端总内力：f_end = k·d + f⁰。
    """
    wx, wy, wz = _member_uniform_loads(model, member)
    L = model.member_length(member.id)
    f = np.zeros(12)

    # 轴向均布 wx -> 两端轴力（压为正的平衡力）
    if wx != 0.0:
        f[0] += -wx * L / 2.0
        f[6] += -wx * L / 2.0

    # 局部 y 方向均布 wy -> v 方向剪力 + 绕 z 弯矩（x-y 平面）
    # 一致荷载向量: P_eq = w[L/2, +L²/12, L/2, -L²/12]（[v1,rz1,v2,rz2]）
    # 固端力 f⁰ = -P_eq
    if wy != 0.0:
        f[1] += -wy * L / 2.0
        f[5] += -wy * L * L / 12.0
        f[7] += -wy * L / 2.0
        f[11] += +wy * L * L / 12.0

    # 局部 z 方向均布 wz -> w 方向剪力 + 绕 y 弯矩（x-z 平面）
    # 一致荷载向量: P_eq = w[L/2, -L²/12, L/2, +L²/12]（[w1,ry1,w2,ry2]）
    if wz != 0.0:
        f[2] += -wz * L / 2.0
        f[4] += +wz * L * L / 12.0
        f[8] += -wz * L / 2.0
        f[10] += -wz * L * L / 12.0

    return f


def equivalent_nodal_loads(model: FrameModel, member) -> np.ndarray:
    """杆件均布荷载的等效节点荷载向量（全局坐标，12 维），用于组装到 F。"""
    R = rotation_matrix(model, member, member.gamma)
    T = transformation_matrix(R)
    return T @ (-fixed_end_forces(model, member))


# ---------------------------------------------------------------------------
# 4. 组装与求解
# ---------------------------------------------------------------------------

def _node_dof_indices(model: FrameModel, node_id: int) -> np.ndarray:
    base = 6 * (node_id - 1)                # 节点 id 从 1 开始
    return np.arange(base, base + 6)


def _member_dof_indices(model: FrameModel, member) -> np.ndarray:
    return np.concatenate([_node_dof_indices(model, member.node_i),
                           _node_dof_indices(model, member.node_j)])


def assemble_global_stiffness(model: FrameModel) -> np.ndarray:
    """组装全局刚度矩阵 K（6N × 6N）。"""
    n = model.num_dofs
    K = np.zeros((n, n))
    for m in model.members.values():
        kg = global_stiffness(model, m)
        idx = _member_dof_indices(model, m)
        K[np.ix_(idx, idx)] += kg
    return K


def assemble_load_vector(model: FrameModel) -> np.ndarray:
    """组装全局荷载向量 F（节点荷载 + 均布荷载等效节点荷载）。"""
    n = model.num_dofs
    F = np.zeros(n)
    for nl in model.nodal_loads:
        dofs = _node_dof_indices(model, nl.node)
        F[dofs] += [nl.fx, nl.fy, nl.fz, nl.mx, nl.my, nl.mz]
    for m in model.members.values():
        idx = _member_dof_indices(model, m)
        F[idx] += equivalent_nodal_loads(model, m)
    return F


def _build_constraint(model: FrameModel) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (约束自由度, 自由自由度) 索引数组。"""
    constrained = set()
    for s in model.supports:
        base = 6 * (s.node - 1)
        for k, fixed in enumerate(s.fix):
            if fixed:
                constrained.add(base + k)
    free = [i for i in range(model.num_dofs) if i not in constrained]
    return np.array(sorted(constrained), dtype=int), np.array(free, dtype=int)


# ---------------------------------------------------------------------------
# 5. 结果对象
# ---------------------------------------------------------------------------

class AnalysisResult:
    """有限元分析结果。"""

    def __init__(self, model: FrameModel, U: np.ndarray,
                 R: np.ndarray, member_forces: Dict[int, np.ndarray]):
        self.model = model
        self.U = U                            # 全部节点位移 (6N,)
        self.R = R                            # 全部节点反力（仅支座处有意义）(6N,)
        self.member_forces = member_forces    # {mid: 局部坐标杆端总力 (12,)}

    def node_displacement(self, node_id: int) -> np.ndarray:
        base = 6 * (node_id - 1)
        return self.U[base:base + 6]

    def max_displacement(self) -> float:
        """最大节点平动位移幅值（m）。"""
        U = self.U.reshape(-1, 6)[:, :3]
        return float(np.max(np.linalg.norm(U, axis=1)))

    def support_reactions(self) -> list:
        """支座反力列表 [(node, fx, fy, fz, mx, my, mz), ...]"""
        out = []
        for s in self.model.supports:
            base = 6 * (s.node - 1)
            out.append((s.node, *self.R[base:base + 6]))
        return out


def solve(model: FrameModel) -> AnalysisResult:
    """求解 KU = F，返回位移、反力与杆端内力。"""
    K = assemble_global_stiffness(model)
    F = assemble_load_vector(model)

    constrained, free = _build_constraint(model)
    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    Uf = np.linalg.solve(Kff, Ff)

    U = np.zeros(model.num_dofs)
    U[free] = Uf

    R = K @ U - F                          # 支座反力（约束自由度上有意义）

    # 杆端总内力（局部坐标）：f = k·d + f⁰
    member_forces = {}
    for m in model.members.values():
        sec = model.sections[m.section]
        mat = model.materials[m.material]
        L = model.member_length(m.id)
        k_l = local_stiffness(L, mat.E, mat.G, sec.A, sec.Iy, sec.Iz, sec.J)
        Rm = rotation_matrix(model, m, m.gamma)
        T = transformation_matrix(Rm)
        d_global = U[_member_dof_indices(model, m)]
        d_local = T.T @ d_global
        f_local = k_l @ d_local + fixed_end_forces(model, m)
        member_forces[m.id] = f_local

    return AnalysisResult(model, U, R, member_forces)


# ---------------------------------------------------------------------------
# 6. 内力提取
# ---------------------------------------------------------------------------

def member_section_forces(model: FrameModel, member,
                          f_local: np.ndarray, x: float) -> np.ndarray:
    """杆件在距 i 端距离 x（局部坐标，m）处的内力向量。

    返回 [N, Vy, Vz, Mx, My, Mz]（局部坐标）：
      N  轴力（拉为正）；Vy/Vz 剪力；Mx 扭矩；My/Mz 弯矩。

    由左段平衡导出：
      N(x)  = -f[0] - wx·x
      Vy(x) = -f[1] - wy·x
      Vz(x) = -f[2] - wz·x
      Mx(x) = -f[3]
      My(x) = -f[4] + x·f[2] + wz·x²/2
      Mz(x) = -f[5] + x·f[1] + wy·x²/2
    其中 f = f_local 为“节点作用于杆件”的杆端力（k·d + f⁰）。
    """
    wx, wy, wz = _member_uniform_loads(model, member)
    return np.array([
        -f_local[0] - wx * x,
        -f_local[1] - wy * x,
        -f_local[2] - wz * x,
        -f_local[3],
        -f_local[4] + x * f_local[2] + wz * x * x / 2.0,
        -f_local[5] + x * f_local[1] + wy * x * x / 2.0,
    ])


def member_extreme_forces(model: FrameModel, member,
                          f_local: np.ndarray, n_samples: int = 21) -> dict:
    """沿杆件采样，返回各内力的最大值（绝对值）字典。

    返回 {'N': float, 'Vy': float, 'Vz': float, 'Mx': float,
          'My': float, 'Mz': float}，均为绝对值极大值。
    """
    L = model.member_length(member.id)
    xs = np.linspace(0.0, L, n_samples)
    forces = np.array([member_section_forces(model, member, f_local, x)
                       for x in xs])
    names = ['N', 'Vy', 'Vz', 'Mx', 'My', 'Mz']
    return {n: float(np.max(np.abs(forces[:, i]))) for i, n in enumerate(names)}


def member_internal_forces(f_local: np.ndarray) -> dict:
    """把 12 维杆端力整理为易读的杆端内力字典（i 端与 j 端）。

    返回 {N, Vy, Vz, Mx, My, Mz}，每项为 (i端值, j端值)，
    采用与 member_section_forces 一致的正负约定。
    """
    return {
        'N':   (-f_local[0], f_local[6]),
        'Vy':  (-f_local[1], f_local[7]),
        'Vz':  (-f_local[2], f_local[8]),
        'Mx':  (-f_local[3], f_local[9]),
        'My':  (-f_local[4], f_local[10]),
        'Mz':  (-f_local[5], f_local[11]),
    }
