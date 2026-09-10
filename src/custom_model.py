"""
通用空间刚架建模器（任意构型）
==============================
让"任意构型"都能进入有限元计算，不再依赖固定模板（规则刚架 / 桁架桥）。

用户（或 LLM）只需给出通用模型描述 JSON，程序即完成：
  建模 -> 求解 -> GB50017 校核 -> 结果报告

通用模型描述（单位默认 SI：m, N, N·m, N/m；可通过 units 切换）：

    {
      "nodes":    [ {"id": 1, "x": 0, "y": 0, "z": 0}, ... ],
      "members":  [ {"i": 1, "j": 2, "section": "HW200"}, ... ],
      "supports": [ {"node": 1, "fix": [true,true,true,true,true,true]}, ... ],
      "nodal_loads":  [ {"node": 2, "fx": 0, "fy": 0, "fz": -100000}, ... ],
      "member_loads": [ {"member": 1, "wz": -5000}, ... ],
      "sections": {"MY_SEC": {"A": .., "Iy": .., "Iz": .., "J": .., "Wy": .., "Wz": ..}},
      "material": {"E": 2.06e11, "nu": 0.3, "density": 7850},
      "steel_grade": "Q355",
      "ref_span": 6.0,
      "units": {"length": "m", "force": "N"}        # length: m|mm, force: N|kN
    }

容错规则（对 LLM 输出友好）：
- nodes.id 可省略，程序自动按 1..N 编号
- members 的端点字段可用 i/j、node_i/node_j、n1/n2，或直接写 "1-2"
- supports.fix 可省略（默认全固定），也接受 "固定"/"铰接" 字符串
- section 优先查自定义 sections，其次查截面库（支持 H200 等别名）
- 校验失败抛出带中文说明的 ValueError，方便 LLM 修正后重新调用
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .model import FrameModel, Material
from . import sections_db


# ---------------------------------------------------------------------------
# 数值容错
# ---------------------------------------------------------------------------

def _num(v: Any, name: str, units: Dict[str, float]) -> float:
    """把任意可转数字的值转 float；非法时报中文错误。"""
    try:
        return float(v) * units.get(name, 1.0)
    except (TypeError, ValueError):
        raise ValueError(f"字段 '{name}' 应为数值，收到: {v!r}")


def _unit_scales(spec: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """解析 units 配置，返回长度/力换算系数。

    length: m(1) 或 mm(1e-3)；force: N(1) 或 kN(1e3)。
    """
    u = spec.get('units') or {}
    length = str(u.get('length', 'm')).strip().lower()
    force = str(u.get('force', 'N')).strip().lower()
    if length not in ('m', 'mm'):
        raise ValueError(f"不支持的长度单位: {length}（可选 m / mm）")
    if force not in ('n', 'kn', 'kN'):
        raise ValueError(f"不支持的力单位: {force}（可选 N / kN）")
    l_scale = 1e-3 if length == 'mm' else 1.0
    f_scale = 1e3 if force in ('kn', 'kN') else 1.0
    # 各字段使用的换算系数
    u_len = {'x': l_scale, 'y': l_scale, 'z': l_scale,
             'fx': f_scale, 'fy': f_scale, 'fz': f_scale,
             'mx': f_scale * l_scale, 'my': f_scale * l_scale,
             'mz': f_scale * l_scale,
             'wx': f_scale / l_scale, 'wy': f_scale / l_scale,
             'wz': f_scale / l_scale,
             'A': l_scale ** 2, 'Iy': l_scale ** 4, 'Iz': l_scale ** 4,
             'J': l_scale ** 4, 'Wy': l_scale ** 3, 'Wz': l_scale ** 3,
             'E': f_scale / l_scale ** 2}
    return u_len, u_len


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def _to_bool_list(v: Any, node: int) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """把 fix 转成长度为 6 的布尔元组。"""
    if v is None:
        return (True,) * 6
    if isinstance(v, str):
        s = v.strip()
        if s in ('固定', '固', 'fix', 'fixed', 'FIXED'):
            return (True,) * 6
        if s in ('铰接', '铰', 'pin', 'PIN'):
            return (True, True, True, False, False, False)
    if isinstance(v, (list, tuple)):
        if len(v) != 6:
            raise ValueError(f"节点 {node} 的 fix 应为 6 个布尔值，收到 {len(v)} 个")
        out = []
        for b in v:
            if isinstance(b, bool):
                out.append(b)
            elif b in (0, 1):
                out.append(bool(b))
            elif isinstance(b, str) and b.strip().lower() in ('true', '1'):
                out.append(True)
            elif isinstance(b, str) and b.strip().lower() in ('false', '0'):
                out.append(False)
            else:
                raise ValueError(f"节点 {node} 的 fix 含非法值: {b!r}")
        return tuple(out)
    raise ValueError(f"节点 {node} 的 fix 无法识别: {v!r}")


def _member_endpoints(mem: Dict[str, Any], u_len: Dict[str, float]) -> Tuple[int, int]:
    """提取杆件两端节点 id，支持 i/j、node_i/node_j、n1/n2、"1-2" 字符串。"""
    if isinstance(mem, str):
        m = re.fullmatch(r'\s*(\d+)\s*[-–—]\s*(\d+)\s*', mem)
        if m:
            return int(m.group(1)), int(m.group(2))
        raise ValueError(f"无法解析杆件描述: {mem!r}")
    if not isinstance(mem, dict):
        raise ValueError(f"杆件描述应为对象或 '1-2' 字符串，收到: {mem!r}")

    # 直接数字
    for k in ('i', 'node_i', 'n1', 'a'):
        if k in mem:
            return _node_id(mem[k], '杆件端点'), _node_id(mem.get('j', mem.get('node_j', mem.get('n2', mem.get('b')))), '杆件端点')
    # 字符串 "1-2"
    for k in ('ends', 'connect', 'nodes'):
        if k in mem and isinstance(mem[k], str):
            m = re.fullmatch(r'\s*(\d+)\s*[-–—]\s*(\d+)\s*', mem[k])
            if m:
                return int(m.group(1)), int(m.group(2))
    raise ValueError(f"杆件缺少端点（需 i/j、node_i/node_j、n1/n2 或 '1-2'）: {mem!r}")


def _node_id(v: Any, ctx: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{ctx} 的节点号应为整数，收到: {v!r}")


# ---------------------------------------------------------------------------
# 主入口：通用 spec -> FrameModel
# ---------------------------------------------------------------------------

def build_custom_model(spec: Dict[str, Any]) -> FrameModel:
    """从通用 JSON 描述构建任意空间刚架模型。

    校验失败时抛出 ValueError（中文信息），供上层回显 / 让 LLM 修正。
    """
    if not isinstance(spec, dict):
        raise ValueError("模型描述应为 JSON 对象")

    u_len, _ = _unit_scales(spec)
    m = FrameModel()

    # ---------------- 材料 ----------------
    mat = spec.get('material') or {}
    # 默认 E 为 SI 值（Pa = N/m²）；仅当用户显式给出 E 时才按 units 换算
    E = _num(mat['E'], 'E', u_len) if 'E' in mat else 2.06e11
    nu = _num(mat.get('nu', 0.3), 'nu', {})
    density = _num(mat.get('density', 7850.0), 'density', {})
    if E <= 0:
        raise ValueError(f"弹性模量 E 必须 > 0，收到 {E}")
    m.add_material('STEEL', E=E, nu=nu, density=density)

    # ---------------- 自定义截面 ----------------
    custom_sections: Dict[str, Dict[str, float]] = {}
    for name, sec in (spec.get('sections') or {}).items():
        if not isinstance(sec, dict):
            raise ValueError(f"截面 {name} 描述应为对象")
        cs = {'A': _num(sec.get('A'), 'A', u_len),
              'Iy': _num(sec.get('Iy'), 'Iy', u_len),
              'Iz': _num(sec.get('Iz'), 'Iz', u_len),
              'J': _num(sec.get('J', 0.0), 'J', u_len),
              'Wy': _num(sec.get('Wy', 0.0), 'Wy', u_len),
              'Wz': _num(sec.get('Wz', 0.0), 'Wz', u_len)}
        if cs['A'] <= 0 or cs['Iy'] <= 0 or cs['Iz'] <= 0:
            raise ValueError(f"截面 {name} 的 A/Iy/Iz 必须 > 0")
        custom_sections[str(name).upper()] = cs

    # ---------------- 节点 ----------------
    raw_nodes = spec.get('nodes')
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("缺少 nodes：请至少提供 2 个节点（如 [{\"x\":0,\"y\":0,\"z\":0}, ...]）")
    node_ids: List[int] = []
    auto_id = 1
    for nd in raw_nodes:
        if not isinstance(nd, dict):
            raise ValueError(f"节点描述应为对象，收到: {nd!r}")
        nid = nd.get('id', auto_id)
        nid = _node_id(nid, '节点')
        x = _num(nd.get('x', 0.0), 'x', u_len)
        y = _num(nd.get('y', 0.0), 'y', u_len)
        z = _num(nd.get('z', 0.0), 'z', u_len)
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ValueError(f"节点 {nid} 的坐标必须是有限数值")
        if nid in m.nodes:
            raise ValueError(f"节点 {nid} 重复定义")
        from .model import Node
        m.nodes[nid] = Node(nid, x, y, z)
        node_ids.append(nid)
        auto_id = max(auto_id, nid + 1)
    if len(node_ids) < 2:
        raise ValueError("节点数至少 2 个才能构成结构")

    # ---------------- 杆件 ----------------
    raw_members = spec.get('members')
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("缺少 members：请至少提供 1 根杆件（如 [{\"i\":1,\"j\":2,\"section\":\"HW200\"}]）")
    connected: Dict[int, int] = {nid: 0 for nid in m.nodes}
    for mb in raw_members:
        i, j = _member_endpoints(mb, u_len)
        if i not in m.nodes:
            raise ValueError(f"杆件端点节点 {i} 不存在（可用节点: {sorted(m.nodes)}）")
        if j not in m.nodes:
            raise ValueError(f"杆件端点节点 {j} 不存在（可用节点: {sorted(m.nodes)}）")
        if i == j:
            raise ValueError(f"杆件 {i}-{j} 两端为同一节点")
        ni, nj = m.nodes[i], m.nodes[j]
        dx, dy, dz = nj.x - ni.x, nj.y - ni.y, nj.z - ni.z
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L < 1e-9:
            raise ValueError(f"杆件 {i}-{j} 长度为零")
        # 截面：自定义优先，其次截面库
        sec_name_raw = mb.get('section', mb.get('sec', ''))
        if isinstance(sec_name_raw, (int, float)):
            sec_name_raw = f"H{int(sec_name_raw)}"
        raw_key = str(sec_name_raw).upper().replace('-', '').replace(' ', '')
        if raw_key in custom_sections:
            cs = custom_sections[raw_key]
            m.add_section(raw_key, cs['A'], cs['Iy'], cs['Iz'],
                          cs['J'], cs['Wy'], cs['Wz'])
            final_sec = raw_key
        else:
            # 截面库键名归一化（X->x 以匹配别名，如 HX300 -> Hx300）
            sec_key = raw_key.replace('×', 'X').replace('X', 'x')
            final_sec = sections_db.resolve(sec_key)
            sp = sections_db.section_properties(final_sec) if final_sec else None
            if sp is None:
                avail = '、'.join(sorted(sections_db._H_DIMS.keys()))
                raise ValueError(
                    f"未知截面: {sec_name_raw}（可用截面: {avail}，或通过 sections 定义自定义截面）")
            m.add_section(sp['name'], sp['A'], sp['Iy'], sp['Iz'],
                          sp['J'], sp['Wy'], sp['Wz'])
        gamma = _num(mb.get('gamma', 0.0), 'gamma', {})
        m.add_member(i, j, final_sec, 'STEEL', gamma=gamma)
        connected[i] += 1
        connected[j] += 1

    # ---------------- 支座 ----------------
    raw_supports = spec.get('supports') or []
    if not isinstance(raw_supports, list):
        raise ValueError("supports 应为列表")
    for sp in raw_supports:
        if not isinstance(sp, dict):
            raise ValueError(f"支座描述应为对象，收到: {sp!r}")
        nid = _node_id(sp.get('node'), '支座')
        if nid not in m.nodes:
            raise ValueError(f"支座节点 {nid} 不存在")
        fix = _to_bool_list(sp.get('fix'), nid)
        m.add_support(nid, fix)

    # ---------------- 荷载 ----------------
    raw_loads = spec.get('nodal_loads') or []
    if not isinstance(raw_loads, list):
        raise ValueError("nodal_loads 应为列表")
    for nl in raw_loads:
        if not isinstance(nl, dict):
            raise ValueError(f"节点荷载描述应为对象，收到: {nl!r}")
        nid = _node_id(nl.get('node'), '节点荷载')
        if nid not in m.nodes:
            raise ValueError(f"节点荷载作用的节点 {nid} 不存在")
        m.add_nodal_load(
            nid,
            fx=_num(nl.get('fx', 0.0), 'fx', u_len),
            fy=_num(nl.get('fy', 0.0), 'fy', u_len),
            fz=_num(nl.get('fz', 0.0), 'fz', u_len),
            mx=_num(nl.get('mx', 0.0), 'mx', u_len),
            my=_num(nl.get('my', 0.0), 'my', u_len),
            mz=_num(nl.get('mz', 0.0), 'mz', u_len),
        )

    raw_ml = spec.get('member_loads') or []
    if not isinstance(raw_ml, list):
        raise ValueError("member_loads 应为列表")
    for ml in raw_ml:
        if not isinstance(ml, dict):
            raise ValueError(f"杆件均布荷载描述应为对象，收到: {ml!r}")
        mid = _node_id(ml.get('member'), '杆件均布荷载')
        if mid not in m.members:
            raise ValueError(f"杆件均布荷载作用的杆件 {mid} 不存在")
        m.add_member_load(
            mid,
            wx=_num(ml.get('wx', 0.0), 'wx', u_len),
            wy=_num(ml.get('wy', 0.0), 'wy', u_len),
            wz=_num(ml.get('wz', 0.0), 'wz', u_len),
        )

    # ---------------- 全局合理性检查 ----------------
    # 1) 至少 1 个约束自由度（否则刚体位移，刚度矩阵奇异）
    n_fix = sum(1 for s in m.supports for f in s.fix if f)
    if n_fix == 0:
        raise ValueError("缺少支座：请至少约束一个节点（如 {\"node\":1,\"fix\":[true]*6}）")

    # 2) 无约束节点必须被杆件连接（否则该节点自由度奇异）
    supported = {s.node for s in m.supports}
    for nid, cnt in connected.items():
        if cnt == 0 and nid not in supported:
            raise ValueError(
                f"节点 {nid} 既没有连接任何杆件也没有支座，无法计算"
                f"（请给它加杆件或支座，或删除该节点）")

    # 3) 截面重复添加去重：add_section 以字典赋值，同名列覆盖，无碍
    return m


# ---------------------------------------------------------------------------
# 文本兜底解析（无 LLM 时支持直接粘贴节点/杆件表）
# ---------------------------------------------------------------------------

def parse_custom_text(text: str) -> Dict[str, Any]:
    """从自由文本解析通用模型描述（规则解析，无需 LLM）。

    支持格式：
        节点:
          1 0 0 0
          2 6 0 0
        杆件:
          1-2 HW200
          2-3 HN300
        支座:
          1 固定
          3 铰接
        荷载:
          3 FZ=-20000
    也支持用分号分隔的紧凑写法。单位默认 m/N，可写 "单位: mm kN"。
    """
    spec: Dict[str, Any] = {'nodes': [], 'members': [], 'supports': [],
                            'nodal_loads': []}
    section: Optional[str] = None
    lines = [ln.strip() for ln in text.replace('；', ';').splitlines() if ln.strip()]
    # 支持分号分隔的紧凑写法
    expanded: List[str] = []
    for ln in lines:
        for part in ln.split(';'):
            part = part.strip()
            if part:
                expanded.append(part)

    # 单位行
    m = re.search(r'单位\s*[:：]\s*(\S+)\s+(\S+)', text)
    if m:
        spec['units'] = {'length': m.group(1), 'force': m.group(2)}

    cn_num = {'一': 1, '两': 2, '二': 2, '三': 3, '四': 4, '五': 5,
              '六': 6, '七': 7, '八': 8, '九': 9}

    def _coord(v: str) -> float:
        return float(v)

    for ln in expanded:
        low = ln.lower()
        if low.startswith('节点') or low.startswith('node'):
            section = 'nodes'
            continue
        if low.startswith('杆件') or low.startswith('member') or low.startswith('单元'):
            section = 'members'
            continue
        if low.startswith('支座') or low.startswith('support') or low.startswith('约束'):
            section = 'supports'
            continue
        if low.startswith('荷载') or low.startswith('load'):
            section = 'loads'
            continue
        if low.startswith('单位') or low.startswith('unit'):
            continue
        if not section:
            raise ValueError(f"无法识别行（请先写 节点:/杆件:/支座:/荷载: 分节）: {ln!r}")

        if section == 'nodes':
            parts = ln.replace(',', ' ').split()
            if len(parts) < 3:
                raise ValueError(f"节点行应为 'id x y z'，收到: {ln!r}")
            if len(parts) == 3:
                # 无 id：自动编号
                spec['nodes'].append({'x': _coord(parts[0]), 'y': _coord(parts[1]),
                                      'z': _coord(parts[2])})
            else:
                spec['nodes'].append({'id': int(parts[0]), 'x': _coord(parts[1]),
                                      'y': _coord(parts[2]), 'z': _coord(parts[3])})
        elif section == 'members':
            parts = ln.split()
            ends = parts[0]
            sec = parts[1] if len(parts) > 1 else 'HW200'
            mm = re.fullmatch(r'(\d+)[-–—](\d+)', ends)
            if not mm:
                raise ValueError(f"杆件行应为 'i-j 截面'，收到: {ln!r}")
            spec['members'].append({'i': int(mm.group(1)), 'j': int(mm.group(2)),
                                    'section': sec})
        elif section == 'supports':
            parts = ln.split()
            nid = int(parts[0])
            fix = parts[1] if len(parts) > 1 else '固定'
            spec['supports'].append({'node': nid, 'fix': fix})
        elif section == 'loads':
            parts = ln.replace(',', ' ').split()
            if not parts:
                continue
            nid = int(parts[0])
            load: Dict[str, Any] = {'node': nid}
            for tok in parts[1:]:
                mm = re.fullmatch(r'([fFmM][xyzXYZ])\s*=\s*([+-]?[\d.eE]+)', tok)
                if mm:
                    load[mm.group(1).lower()] = float(mm.group(2))
                else:
                    raise ValueError(f"荷载行应为 '节点号 FZ=-20000'，收到: {ln!r}")
            spec['nodal_loads'].append(load)

    if not spec['nodes']:
        raise ValueError("未解析到任何节点，请按 '节点:' 分节提供节点表")
    if not spec['members']:
        raise ValueError("未解析到任何杆件，请按 '杆件:' 分节提供杆件表")
    return spec


# ---------------------------------------------------------------------------
# 示例（前端 / 文档 / 测试用）
# ---------------------------------------------------------------------------

def example_specs() -> Dict[str, Dict[str, Any]]:
    """内置通用构型示例。"""
    # 四角锥塔架（顶部集中荷载 + 底部四角固定）
    h, b = 6.0, 2.0
    tower = {
        'units': {'length': 'm', 'force': 'N'},
        'nodes': [
            {'id': 1, 'x': -b, 'y': -b, 'z': 0.0},
            {'id': 2, 'x':  b, 'y': -b, 'z': 0.0},
            {'id': 3, 'x':  b, 'y':  b, 'z': 0.0},
            {'id': 4, 'x': -b, 'y':  b, 'z': 0.0},
            {'id': 5, 'x': 0.0, 'y': 0.0, 'z': h},
        ],
        'members': [
            {'i': 1, 'j': 2, 'section': 'HW200'},
            {'i': 2, 'j': 3, 'section': 'HW200'},
            {'i': 3, 'j': 4, 'section': 'HW200'},
            {'i': 4, 'j': 1, 'section': 'HW200'},
            {'i': 1, 'j': 5, 'section': 'HW150'},
            {'i': 2, 'j': 5, 'section': 'HW150'},
            {'i': 3, 'j': 5, 'section': 'HW150'},
            {'i': 4, 'j': 5, 'section': 'HW150'},
        ],
        'supports': [{'node': n, 'fix': [True] * 6} for n in (1, 2, 3, 4)],
        'nodal_loads': [{'node': 5, 'fz': -200000.0}],
        'steel_grade': 'Q355',
        'ref_span': 6.0,
    }

    # 悬挑 L 形平面框架（演示不规则平面）
    lframe = {
        'units': {'length': 'm', 'force': 'kN'},
        'nodes': [
            {'id': 1, 'x': 0, 'y': 0, 'z': 0},
            {'id': 2, 'x': 6, 'y': 0, 'z': 0},
            {'id': 3, 'x': 6, 'y': 3, 'z': 0},
            {'id': 4, 'x': 9, 'y': 3, 'z': 0},
            {'id': 5, 'x': 0, 'y': 0, 'z': 4},
            {'id': 6, 'x': 6, 'y': 0, 'z': 4},
            {'id': 7, 'x': 6, 'y': 3, 'z': 4},
            {'id': 8, 'x': 9, 'y': 3, 'z': 4},
        ],
        'members': [
            {'i': 1, 'j': 2, 'section': 'HW200'},
            {'i': 2, 'j': 3, 'section': 'HN300'},
            {'i': 3, 'j': 4, 'section': 'HN300'},
            {'i': 5, 'j': 6, 'section': 'HW200'},
            {'i': 6, 'j': 7, 'section': 'HN300'},
            {'i': 7, 'j': 8, 'section': 'HN300'},
            {'i': 1, 'j': 5, 'section': 'HW200'},
            {'i': 2, 'j': 6, 'section': 'HW200'},
            {'i': 3, 'j': 7, 'section': 'HW200'},
            {'i': 4, 'j': 8, 'section': 'HW200'},
        ],
        'supports': [{'node': n, 'fix': [True] * 6} for n in (1, 2, 3, 4)],
        'nodal_loads': [{'node': 4, 'fz': -50.0}, {'node': 8, 'fz': -50.0}],
        'steel_grade': 'Q355',
        'ref_span': 6.0,
    }
    return {'塔架': tower, 'L形框架': lframe}


def spec_summary(spec: Dict[str, Any]) -> str:
    """模型规模概述（用于回复展示）。"""
    n_nodes = len(spec.get('nodes') or [])
    n_mem = len(spec.get('members') or [])
    n_sup = len(spec.get('supports') or [])
    n_ld = len(spec.get('nodal_loads') or []) + len(spec.get('member_loads') or [])
    return (f"{n_nodes} 个节点、{n_mem} 根杆件、{n_sup} 组支座、{n_ld} 组荷载")
