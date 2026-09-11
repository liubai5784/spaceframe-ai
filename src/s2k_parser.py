"""
SAP2000 .s2k 文件解析器
========================
将 SAP2000 导出的文本模型文件（.s2k / .$2k）转换为程序内部的 FrameModel。

支持的表（兼容新旧版本命名）：
- PROGRAM CONTROL                  （单位）
- JOINT COORDINATES                （节点坐标）
- JOINT RESTRAINT ASSIGNMENTS      （支座约束）
- CONNECTIVITY - FRAME             （杆件连接）
- FRAME SECTION ASSIGNMENTS        （杆件-截面 分配）
- FRAME PROPERTY ASSIGNMENTS       （旧版杆件属性分配）
- MATERIAL PROPERTIES              （材料）
- FRAME SECTIONS                   （旧版截面定义）
- FRAME SECTION PROPERTY DEFINITIONS（新版截面定义）
- LOAD PATTERNS                    （荷载工况，含自重系数）
- JOINT FORCES                     （节点力）
- JOINT LOADS                      （旧版节点力表）
- FRAME DISTRIBUTED LOADS          （杆件均布/梯形荷载）
- FRAME POINT LOADS                （杆件集中荷载）

约定：
- 杆件局部轴：SAP2000 的 1/2/3 轴 对应本程序的 x/y/z
  （I22 -> Iy，I33 -> Iz；S22 -> Wy，S33 -> Wz）
- 默认只解析 GLOBAL 坐标系；LOCAL 坐标系节点给出警告
- 均布荷载支持均匀分布；非均匀（梯形）荷载解析但发出警告并取平均近似
- 多个荷载工况（LoadPat）默认叠加
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import FrameModel

GRAVITY = 9.80665          # m/s^2

# 单位换算：力 / 长度
UNITS_MAP = {
    # 格式: "kN, m, C" -> (force_scale, length_scale)
    ("kN", "m"): (1e3, 1.0),
    ("kN", "cm"): (1e3, 1e-2),
    ("N", "mm"): (1.0, 1e-3),
    ("N", "m"): (1.0, 1.0),
    ("kgf", "m"): (9.80665, 1.0),
    ("kip", "in"): (4448.2216, 0.0254),
    ("kip", "ft"): (4448.2216, 0.3048),
    ("lb", "in"): (4.4482216, 0.0254),
    ("lb", "ft"): (4.4482216, 0.3048),
    ("tf", "m"): (9806.65, 1.0),
}


class S2KError(Exception):
    """.s2k 解析错误。"""


def _parse_table_sections(text: str) -> Dict[str, List[Dict[str, str]]]:
    """把 s2k 文本解析为 {表名: [{字段: 值}, ...]}。

    行结构：
      - 注释行以 // 开头
      - TABLE:  "表名" 开启一个新表
      - 数据行：Field=Value Field=Value ...（空格分隔）
    """
    tables: Dict[str, List[Dict[str, str]]] = {}
    current = None
    # 匹配 Field=Value：Value 可以是带引号字符串（含空格）或非空白 token
    pair_re = re.compile(r'([A-Za-z][A-Za-z0-9_]*)=(?:"([^"]*)"|(\S+))')

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('//'):
            continue
        if line.upper().startswith('TABLE:'):
            m = re.search(r'"([^"]+)"', line)
            name = m.group(1).strip().upper() if m else line[6:].strip().upper()
            current = name
            tables.setdefault(current, [])
            continue
        if current is None:
            continue

        row: Dict[str, str] = {}
        for m in pair_re.finditer(line):
            row[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
        if row:
            tables[current].append(row)
    return tables


def _fmt(val) -> float:
    """解析数值字段（容忍逗号分隔的整数如 1,000 与缺省值）。"""
    return float(str(val).replace(',', ''))


def _bool(val: str) -> bool:
    return val.strip().upper() in ('YES', 'TRUE', '1', 'Y')


def _parse_units(tables: Dict[str, List[Dict[str, str]]]) -> Tuple[float, float]:
    """从 PROGRAM CONTROL 读取单位，返回 (force_scale, length_scale)。"""
    rows = tables.get('PROGRAM CONTROL', [])
    if not rows:
        return 1e3, 1.0                 # SAP2000 默认 kN, m
    units = rows[0].get('CurrUnits', '"kN, m, C"').strip('"')
    parts = [p.strip() for p in units.split(',')]
    if len(parts) >= 2:
        key = (parts[0], parts[1])
        if key in UNITS_MAP:
            return UNITS_MAP[key]
    raise S2KError(f"无法识别的单位系统: {units}")


def _map_load_dir(direction: str) -> Tuple[str, int]:
    """把 SAP2000 荷载方向字符串映射为 (局部/全局, 轴索引 0/1/2)。

    例：Local 2 -> ('local', 1)；Global Z -> ('global', 2)。
    """
    d = direction.strip().strip('"').lower()
    if d.startswith('local'):
        return 'local', int(d[-1]) - 1
    if d.startswith('global'):
        axis = d[-1]
        return 'global', {'x': 0, 'y': 1, 'z': 2}[axis]
    raise S2KError(f"无法识别的荷载方向: {direction}")


def parse_s2k(path: str, load_patterns: List[str] | None = None,
              include_self_weight: bool = True) -> FrameModel:
    """解析 .s2k 文件，返回 FrameModel。

    参数
    ----
    path              : .s2k 文件路径
    load_patterns     : 需要导入的荷载工况名列表；None 表示全部
    include_self_weight: 是否计入自重（按 LOAD PATTERNS 的自重系数）
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    tables = _parse_table_sections(text)
    force_scale, length_scale = _parse_units(tables)

    model = FrameModel()
    warnings: List[str] = []

    # 坐标单位换算（长度）
    L = length_scale
    F = force_scale
    E_scale = F / (L * L)               # 应力/弹性模量单位 -> Pa
    rho_scale = F / (L * L * L)         # 重度单位 -> N/m^3

    # ---------------------------------------------------------------
    # 1. 节点
    # ---------------------------------------------------------------
    joint_rows = tables.get('JOINT COORDINATES', [])
    for row in joint_rows:
        nid = int(row['Joint'])
        coord_sys = row.get('CoordSys', 'GLOBAL')
        if coord_sys.strip().upper() != 'GLOBAL':
            warnings.append(f"节点 {nid} 使用 {coord_sys} 坐标系，已按 GLOBAL 处理")
        from .model import Node
        model.nodes[nid] = Node(
            nid,
            _fmt(row['XorR']) * L,
            _fmt(row['YorR']) * L,
            _fmt(row['ZorR']) * L,
        )

    # ---------------------------------------------------------------
    # 2. 支座
    # ---------------------------------------------------------------
    restraint_rows = tables.get('JOINT RESTRAINT ASSIGNMENTS', [])
    for row in restraint_rows:
        nid = int(row['Joint'])
        if nid not in model.nodes:
            raise S2KError(f"支座引用不存在的节点 {nid}")
        fix = tuple(_bool(row.get(k, 'No')) for k in ('U1', 'U2', 'U3', 'R1', 'R2', 'R3'))
        model.add_support(nid, fix)

    # ---------------------------------------------------------------
    # 3. 材料
    # ---------------------------------------------------------------
    mat_rows = tables.get('MATERIAL PROPERTIES', [])
    for row in mat_rows:
        name = row['Material']
        E = _fmt(row['E1']) * E_scale
        nu = _fmt(row.get('U12', 0.3))
        # 密度：优先 WeightPerVol，其次 A1（质量密度），再次 A2/A3
        rho = 7850.0
        if 'WeightPerVol' in row:
            rho = _fmt(row['WeightPerVol']) * rho_scale / GRAVITY
        elif 'A1' in row:
            rho = _fmt(row['A1']) * rho_scale / GRAVITY
        model.add_material(name, E, nu, rho)

    # ---------------------------------------------------------------
    # 4. 截面
    # ---------------------------------------------------------------
    sec_rows = tables.get('FRAME SECTION PROPERTY DEFINITIONS', []) or \
        tables.get('FRAME SECTIONS', [])
    for row in sec_rows:
        name = row.get('SectionName') or row.get('PropName')
        if not name:
            continue
        if 'Area' not in row:
            warnings.append(f"截面 {name} 缺少 Area，跳过（可能是自动设计截面）")
            continue
        A = _fmt(row['Area']) * L * L
        Iy = _fmt(row.get('I22', 0.0)) * L ** 4
        Iz = _fmt(row.get('I33', 0.0)) * L ** 4
        J = _fmt(row.get('TorsConst', 0.0)) * L ** 4
        Wy = _fmt(row.get('S22', 0.0)) * L ** 3
        Wz = _fmt(row.get('S33', 0.0)) * L ** 3
        model.add_section(name, A, Iy, Iz, J, Wy, Wz)

    # ---------------------------------------------------------------
    # 5. 杆件
    # ---------------------------------------------------------------
    conn_rows = tables.get('CONNECTIVITY - FRAME', [])
    assign_rows = tables.get('FRAME SECTION ASSIGNMENTS', []) or \
        tables.get('FRAME PROPERTY ASSIGNMENTS', [])
    # 截面分配表：Frame -> (SectionName, MatProp)
    assign: Dict[str, Tuple[str, str]] = {}
    for row in assign_rows:
        fname = row['Frame']
        assign[fname] = (row.get('SectionName') or row.get('PropName'),
                         row.get('MatProp', ''))

    for row in conn_rows:
        fname = row['Frame']
        i = int(row['JointI'])
        j = int(row['JointJ'])
        if i not in model.nodes or j not in model.nodes:
            raise S2KError(f"杆件 {fname} 引用不存在的节点")
        sec_name, mat_name = assign.get(fname, (None, None))
        if sec_name is None or sec_name not in model.sections:
            # 尝试直接从连接表行里读（某些导出格式）
            sec_name = row.get('Section', sec_name)
        if sec_name is None or sec_name not in model.sections:
            raise S2KError(f"杆件 {fname} 未找到截面定义: {sec_name}")
        if not mat_name or mat_name not in model.materials:
            # 取截面行里指定的材料，或第一个材料
            for srow in sec_rows:
                if (srow.get('SectionName') or srow.get('PropName')) == sec_name:
                    mat_name = srow.get('MatProp', mat_name)
                    break
            if not mat_name or mat_name not in model.materials:
                if model.materials:
                    mat_name = next(iter(model.materials))
                else:
                    raise S2KError(f"杆件 {fname} 未找到材料定义")
        model.add_member(i, j, sec_name, mat_name)

    # ---------------------------------------------------------------
    # 6. 荷载
    # ---------------------------------------------------------------
    def want_pattern(p: str) -> bool:
        return load_patterns is None or p in load_patterns

    # 自重系数
    if include_self_weight:
        for row in tables.get('LOAD PATTERNS', []):
            mult = _fmt(row.get('SelfWeightMult', 0.0))
            if mult == 0.0 or not want_pattern(row.get('LoadPat', '')):
                continue
            for mid, mem in model.members.items():
                sec = model.sections[mem.section]
                mat = model.materials[mem.material]
                w = sec.A * mat.density * GRAVITY * mult      # N/m
                model.add_member_load(mid, wz=-w)             # 局部 z 近似全局 -Z

    # 节点力
    jf_rows = tables.get('JOINT FORCES', []) or tables.get('JOINT LOADS', [])
    for row in jf_rows:
        if not want_pattern(row.get('LoadPat', '')):
            continue
        nid = int(row['Joint'])
        if nid not in model.nodes:
            raise S2KError(f"节点荷载引用不存在的节点 {nid}")
        model.add_nodal_load(
            nid,
            fx=_fmt(row.get('FX', 0)) * F,
            fy=_fmt(row.get('FY', 0)) * F,
            fz=_fmt(row.get('FZ', 0)) * F,
            mx=_fmt(row.get('MX', 0)) * F * L,
            my=_fmt(row.get('MY', 0)) * F * L,
            mz=_fmt(row.get('MZ', 0)) * F * L,
        )

    # 杆件均布荷载
    for row in tables.get('FRAME DISTRIBUTED LOADS', []):
        if not want_pattern(row.get('LoadPat', '')):
            continue
        fname = row['Frame']
        mid = None
        for mk in model.members:
            if str(mk) == str(fname):
                mid = mk
                break
        if mid is None:
            try:
                mid = int(fname)
            except ValueError:
                pass
        if mid is None or mid not in model.members:
            raise S2KError(f"分布荷载引用不存在的杆件 {fname}")
        load_type = row.get('Type', 'Force').strip().lower()
        frame, axis = _map_load_dir(row.get('Dir', 'Local 1'))
        v1 = _fmt(row.get('Val1', 0)) * F
        v2 = _fmt(row.get('Val2', v1)) * F
        d1 = _fmt(row.get('Dist1', 0))
        d2 = _fmt(row.get('Dist2', 1))
        if d1 != 0.0 or d2 != 1.0 or abs(v1 - v2) > 1e-12:
            warnings.append(
                f"杆件 {fname} 荷载为非均匀分布（{d1}-{d2}, {v1}-{v2}），"
                f"已按均匀分布 {v1} 近似")

        if load_type == 'force':
            if frame == 'local':
                vec = [0.0, 0.0, 0.0]
                vec[axis] = v1
                model.add_member_load(mid, wx=vec[0], wy=vec[1], wz=vec[2])
            else:
                # 全局方向荷载 -> 转换到局部坐标
                from .fem import rotation_matrix
                gvec = np.zeros(3)
                gvec[axis] = v1
                lvec = rotation_matrix(model, model.members[mid]).T @ gvec
                model.add_member_load(mid, wx=lvec[0], wy=lvec[1], wz=lvec[2])
        elif load_type == 'moment':
            warnings.append(f"杆件 {fname} 分布弯矩荷载暂不支持，已跳过")
        else:
            warnings.append(f"杆件 {fname} 荷载类型 {load_type} 暂不支持，已跳过")

    # 杆件集中荷载
    for row in tables.get('FRAME POINT LOADS', []):
        if not want_pattern(row.get('LoadPat', '')):
            continue
        mid = int(row['Frame'])
        if mid not in model.members:
            raise S2KError(f"集中荷载引用不存在的杆件 {mid}")
        warnings.append(
            f"杆件 {mid} 上的集中荷载暂不支持（当前仅支持节点力与杆件均布荷载），已跳过")

    if warnings:
        print("=== .s2k 解析警告 ===")
        for w in warnings:
            print(" -", w)

    return model
