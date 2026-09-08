"""
常用热轧 H 型钢截面库（GB/T 11263-2017）
=========================================
供 Agent 与建模使用：截面名 -> (A, Iy, Iz, J, Wy, Wz)，单位 SI（m², m⁴, m³）。

数值来源：GB/T 11263-2017 标准截面表（部分常见规格）。
- Iz/I22 为绕截面强轴（腹板方向，对应程序局部 z / SAP2000 3 轴）
- Iy/I23 为绕截面弱轴
- 扭转常数 J 按板件公式 J = (2·bf·tf³ + (h-2·tf)·tw³)/3 近似计算
- 翼缘/腹板尺寸 h×bf×tw×tf 单位为 mm

说明：正式工程设计请以设计手册或厂家样本为准。
"""

from __future__ import annotations

from typing import Dict, Tuple

# name: (h, bf, tw, tf)  单位 mm
_H_DIMS: Dict[str, Tuple[float, float, float, float]] = {
    'HW150': (150, 150, 7, 10),
    'HW200': (200, 200, 8, 12),
    'HW250': (250, 250, 9, 14),
    'HW300': (300, 300, 10, 15),
    'HM294': (294, 200, 8, 12),
    'HM340': (340, 250, 9, 14),
    'HM440': (440, 300, 11, 18),
    'HN300': (300, 150, 6.5, 9),
    'HN350': (350, 175, 7, 11),
    'HN400': (400, 200, 8, 13),
    'HN500': (500, 200, 10, 16),
    'HN600': (600, 200, 11, 17),
}

# 标准截面表（GB/T 11263-2017）：A(cm²), Ix(cm⁴), Iy(cm⁴), Wx(cm³), Wy(cm³)
_STD = {
    'HW150': (39.65, 1620, 563, 216, 75.1),
    'HW200': (63.53, 4720, 1600, 472, 160),
    'HW250': (91.43, 10700, 3650, 867, 292),
    'HW300': (118.45, 20300, 6750, 1360, 450),
    'HM294': (71.05, 11100, 3400, 755, 340),
    'HM340': (99.53, 21200, 3650, 1250, 292),
    'HM440': (153.9, 56100, 8110, 2550, 541),
    'HN300': (46.78, 6820, 508, 455, 67.7),
    'HN350': (62.91, 13600, 984, 779, 112),
    'HN400': (83.37, 22800, 1740, 1140, 174),
    'HN500': (112.25, 46800, 2140, 1870, 214),
    'HN600': (131.4, 72500, 2450, 2420, 245),
}


def _torsion_j(h_mm: float, bf_mm: float, tw_mm: float, tf_mm: float) -> float:
    """H 型钢扭转常数近似：J ≈ (2·bf·tf³ + (h-2·tf)·tw³)/3，单位 m⁴。"""
    mm = 1e-3
    return ((2 * bf_mm * tf_mm**3 + (h_mm - 2 * tf_mm) * tw_mm**3) / 3.0) * mm**4


def section_properties(name: str) -> dict | None:
    """返回截面 SI 属性字典，未收录时返回 None。"""
    name = name.upper().replace('-', '').replace('×', 'x').replace('X', 'x')
    key = name[:5] if name[:5] in _H_DIMS else name[:4]
    if key not in _H_DIMS:
        return None
    A_cm2, Ix_cm4, Iy_cm4, Wx_cm3, Wy_cm3 = _STD[key]
    h, bf, tw, tf = _H_DIMS[key]
    return {
        'name': key,
        'A': A_cm2 * 1e-4,                 # cm² -> m²
        'Iy': Iy_cm4 * 1e-8,               # cm⁴ -> m⁴（弱轴）
        'Iz': Ix_cm4 * 1e-8,               # cm⁴ -> m⁴（强轴）
        'J': _torsion_j(h, bf, tw, tf),
        'Wy': Wy_cm3 * 1e-6,               # cm³ -> m³（弱轴）
        'Wz': Wx_cm3 * 1e-6,               # cm³ -> m³（强轴）
    }


def all_sections() -> Dict[str, dict]:
    """返回全部收录截面的 SI 属性。"""
    return {k: section_properties(k) for k in _H_DIMS}


# 别名（方便自然语言引用）
ALIASES = {
    'H150': 'HW150', '150': 'HW150',
    'H200': 'HW200', '200': 'HW200',
    'H250': 'HW250', '250': 'HW250',
    'H300': 'HW300', '300': 'HW300',
    'H350': 'HN350', '350': 'HN350',
    'H400': 'HN400', '400': 'HN400',
    'H500': 'HN500', '500': 'HN500',
}


def resolve(name: str) -> str:
    """把常见称呼（H200/HN200 等）归一化为截面库键名。"""
    n = name.upper().replace('-', '').replace(' ', '')
    n = n.replace('×', 'X').replace('X', 'x')
    if n in _H_DIMS:
        return n
    if n.startswith('H') and n[1:] in ALIASES:
        return ALIASES[n[1:]]
    for k, v in ALIASES.items():
        if n.endswith(k):
            return v
    return n if n in _H_DIMS else name
