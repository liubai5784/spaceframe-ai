"""
LLM Agent 层（与大语言模型结合的核心）
======================================
Agent 的工作方式：

1. 用户用自然语言描述结构需求（如"5m×4m×3m 两层刚架，柱 H200，梁 HN300，
   顶部荷载 20kN/m²"）
2. Agent 调用 LLM（Function Calling）把需求解析为结构化参数
3. 程序用参数化模板/有限元内核完成建模与计算（LLM 不直接算数）
4. Agent 再把计算结果交给 LLM 生成中文解释报告（安全判定、最危险构件、规范条款）

关键设计（答辩要点）：
- LLM 只负责"自然语言 <-> 结构化参数"与"结果解释"，数值 100% 由有限元内核计算
- 所有 LLM 输出经过程序侧校验（参数范围/截面库/物理合理性），防止幻觉
- 支持多种后端：豆包(火山方舟)/DeepSeek/OpenAI 兼容接口；无 Key 时降级为规则解析

配置方式（环境变量）：
  LLM_API_KEY   API Key
  LLM_BASE_URL  接口地址（默认 https://ark.cn-beijing.volces.com/api/v3）
  LLM_MODEL     模型名（默认 doubao-seed-1-6-250615）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .model import FrameModel
from .fem import solve, AnalysisResult
from .checks import check_model, CheckOptions, CheckReport
from .templates import (FrameParams, build_frame,
                        TrussBridgeParams, build_truss_bridge)
from .custom_model import build_custom_model, parse_custom_text, spec_summary
from . import sections_db


# ---------------------------------------------------------------------------
# 1. LLM 客户端（OpenAI 兼容）
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI 兼容接口客户端（豆包/DeepSeek/通义/本地 Ollama 均可）。

    无 API Key 时 __call__ 返回 None，Agent 降级为规则解析。
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
        self.base_url = base_url or os.environ.get(
            'LLM_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
        self.model = model or os.environ.get('LLM_MODEL', 'doubao-seed-1-6-250615')

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None,
             temperature: float = 0.2) -> Optional[dict]:
        """调用 chat completions，返回完整响应（含 tool_calls 时原样返回）。"""
        if not self.available:
            return None
        try:
            import httpx
        except ImportError:
            return None
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
        }
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        try:
            resp = httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={'Authorization': f'Bearer {self.api_key}',
                         'Content-Type': 'application/json'},
                json=payload, timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:                      # 网络/鉴权失败时降级
            print(f"[Agent] LLM 调用失败，降级为规则解析: {e}")
            return None


# ---------------------------------------------------------------------------
# 2. 规则解析（无 LLM 时的降级路径，同时用于测试）
# ---------------------------------------------------------------------------

def parse_params_rules(text: str) -> FrameParams:
    """从中文自然语言中提取刚架参数（正则规则解析，无需 LLM）。

    支持：跨度/层高/层数/跨数/截面/荷载。缺省用默认值。
    """
    p = FrameParams()
    s = text.strip()

    # 尺寸：N×N×N（m）→ Lx×Ly×Lz
    m3 = re.search(r'([\d.]+)\s*[x×X]\s*([\d.]+)\s*[x×X]\s*([\d.]+)', s)
    if m3:
        p.Lx, p.Ly, p.Lz = float(m3.group(1)), float(m3.group(2)), float(m3.group(3))

    # 层数（支持中文数字：一层/两层/三层）
    _CN_NUM = {'一': 1, '两': 2, '二': 2, '三': 3, '四': 4,
               '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    m = re.search(r'([0-9两一二三四五六七八九])\s*层', s)
    if m:
        t = m.group(1)
        p.nz = int(t) if t.isdigit() else _CN_NUM[t]
    # 跨数 X×Y
    m = re.search(r'(\d+)\s*[x×X]\s*(\d+)\s*跨', s)
    if m:
        p.nx, p.ny = int(m.group(1)), int(m.group(2))

    # 截面：柱/梁
    m = re.search(r'柱[^，。,;；\s]*?[Hh]?(\d{3})', s)
    if m:
        p.column_section = sections_db.resolve(f"H{m.group(1)}")
    m = re.search(r'梁[^，。,;；\s]*?[Hh]?(\d{3})', s)
    if m:
        p.beam_section = sections_db.resolve(f"H{m.group(1)}")

    # 荷载：kN/m²（向下）
    m = re.search(r'([\d.]+)\s*(?:kN|千牛)[/／]?m²?', s)
    if m:
        p.load_kn_m2 = abs(float(m.group(1)))

    # 钢材牌号
    m = re.search(r'[Qq](\d{3})', s)
    if m:
        p.steel_grade = f"Q{m.group(1)}"

    return p


def parse_bridge_rules(text: str) -> TrussBridgeParams:
    """从中文自然语言提取桁架桥参数（正则规则解析）。

    支持：桥长/桥宽/桥高/节间数/弦杆腹杆截面/桥面荷载/钢材。
    """
    p = TrussBridgeParams()
    s = text.strip()

    # 桥长："长24m" 或 "24m长"（优先带"长"字的）
    m = re.search(r'长\s*([\d.]+)\s*m', s) or re.search(r'([\d.]+)\s*m\s*长', s)
    if m:
        p.L = float(m.group(1))
    # 桥宽
    m = re.search(r'宽\s*([\d.]+)\s*m', s)
    if m:
        p.W = float(m.group(1))
    # 桥高/桁高
    m = re.search(r'(?:桥高|桁高|高)\s*([\d.]+)\s*m', s)
    if m:
        p.H = float(m.group(1))
    # 节间数（支持中文数字）
    _CN = {'一':1,'两':2,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    m = re.search(r'([0-9两一二三四五六七八九])\s*个?\s*节间', s)
    if m:
        t = m.group(1)
        p.n_panels = int(t) if t.isdigit() else _CN[t]
    # 弦杆 / 腹杆截面
    m = re.search(r'弦杆[^，。,;；\s]*?[Hh]?(\d{3})', s)
    if m:
        p.chord_section = sections_db.resolve(f"H{m.group(1)}")
    m = re.search(r'腹杆[^，。,;；\s]*?[Hh]?(\d{3})', s)
    if m:
        p.web_section = sections_db.resolve(f"H{m.group(1)}")
    # 未区分弦腹杆时，统一截面（"HW200"直接出现）
    if not re.search(r'弦杆|腹杆', s):
        m = re.search(r'\b(H[WwMmNn]?\d{3})\b', s)
        if m:
            r = sections_db.resolve(m.group(1))
            p.chord_section = p.web_section = r
    # 桥面荷载
    m = re.search(r'([\d.]+)\s*(?:kN|千牛)[/／]?m²?', s)
    if m:
        p.load_kn_m2 = abs(float(m.group(1)))
    # 钢材
    m = re.search(r'[Qq](\d{3})', s)
    if m:
        p.steel_grade = f"Q{m.group(1)}"
    return p


# ---------------------------------------------------------------------------
# 3. Agent 主体
# ---------------------------------------------------------------------------

class SpaceFrameAgent:
    """空间刚架智能计算 Agent。

    用法：
        agent = SpaceFrameAgent(llm=LLMClient())
        reply = agent.ask("设计一个 5×4×3m 的两层刚架，柱 H200，顶部荷载 20kN/m²")
    """

    # 供 LLM 调用的工具定义（Function Calling）
    TOOLS = [{
        'type': 'function',
        'function': {
            'name': 'build_space_frame',
            'description': '根据用户描述构建规则矩形网格空间刚架（等跨等层高）并完成有限元分析与规范校核',
            'parameters': {
                'type': 'object',
                'properties': {
                    'Lx': {'type': 'number', 'description': 'X方向总跨度(m)'},
                    'Ly': {'type': 'number', 'description': 'Y方向总跨度(m)'},
                    'Lz': {'type': 'number', 'description': '层高(m)'},
                    'nx': {'type': 'integer', 'description': 'X向跨数'},
                    'ny': {'type': 'integer', 'description': 'Y向跨数'},
                    'nz': {'type': 'integer', 'description': '层数'},
                    'column_section': {'type': 'string',
                                       'description': '柱截面（如HW200/HN300）'},
                    'beam_section': {'type': 'string',
                                     'description': '梁截面（如HN300）'},
                    'load_kn_m2': {'type': 'number',
                                   'description': '顶部均布荷载(kN/m²，向下为正)'},
                    'steel_grade': {'type': 'string',
                                    'description': '钢材牌号（Q235/Q355/Q390/Q420）'},
                },
                'required': [],
            },
        },
    }]

    # 通用任意构型工具：节点/杆件/支座/荷载自由定义
    CUSTOM_TOOL = {
        'type': 'function',
        'function': {
            'name': 'build_custom_model',
            'description': (
                '根据用户描述构建【任意构型】的空间刚架（不限于规则框架/桁架桥），'
                '并完成有限元分析与规范校核。'
                '当用户描述的结构无法用 build_space_frame 的参数化规则网格表达时'
                '（如塔架、悬挑、不规则平面、空间桁架、构筑物等），必须使用本工具。'
                '你需要把结构离散为节点（坐标）与杆件（连接关系），单位一律用 SI：'
                '坐标 m、力 N、弯矩 N·m、均布荷载 N/m。'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'nodes': {
                        'type': 'array',
                        'description': '节点列表：[{"id":1,"x":0,"y":0,"z":0},...]。'
                                       'id 可省略（自动编号）；z 为高度方向。',
                        'items': {'type': 'object',
                                  'properties': {
                                      'id': {'type': 'integer'},
                                      'x': {'type': 'number'},
                                      'y': {'type': 'number'},
                                      'z': {'type': 'number'}},
                                  'required': ['x', 'y', 'z']}},
                    'members': {
                        'type': 'array',
                        'description': '杆件列表：[{"i":1,"j":2,"section":"HW200"},...]。'
                                       'section 为截面库名（HW150/HW200/HW250/HW300/'
                                       'HM294/HM340/HM440/HN250/HN300/HN350/HN400/HN500/HN600）。',
                        'items': {'type': 'object',
                                  'properties': {
                                      'i': {'type': 'integer'},
                                      'j': {'type': 'integer'},
                                      'section': {'type': 'string'}},
                                  'required': ['i', 'j', 'section']}},
                    'supports': {
                        'type': 'array',
                        'description': '支座列表（可选）：[{"node":1,"fix":'
                                       '[true,true,true,true,true,true]}]。'
                                       'fix 为 6 个自由度(u,v,w,rx,ry,rz)是否固定；'
                                       '省略 fix 表示全固定。至少需要一个支座。',
                        'items': {'type': 'object',
                                  'properties': {
                                      'node': {'type': 'integer'},
                                      'fix': {'type': 'array',
                                              'items': {'type': 'boolean'}}},
                                  'required': ['node']}},
                    'nodal_loads': {
                        'type': 'array',
                        'description': '节点荷载（可选）：[{"node":2,"fx":0,"fy":0,'
                                       '"fz":-100000}]，力 N、弯矩 N·m，向下为负。',
                        'items': {'type': 'object',
                                  'properties': {
                                      'node': {'type': 'integer'},
                                      'fx': {'type': 'number'}, 'fy': {'type': 'number'},
                                      'fz': {'type': 'number'}, 'mx': {'type': 'number'},
                                      'my': {'type': 'number'}, 'mz': {'type': 'number'}},
                                  'required': ['node']}},
                    'member_loads': {
                        'type': 'array',
                        'description': '杆件均布荷载（可选，局部坐标 N/m）：'
                                       '[{"member":1,"wz":-5000}]，wz 为沿杆件局部 z 方向。',
                        'items': {'type': 'object',
                                  'properties': {
                                      'member': {'type': 'integer'},
                                      'wx': {'type': 'number'}, 'wy': {'type': 'number'},
                                      'wz': {'type': 'number'}},
                                  'required': ['member']}},
                    'sections': {
                        'type': 'object',
                        'description': '自定义截面（可选，SI 单位）：'
                                       '{"SEC1":{"A":..,"Iy":..,"Iz":..,"J":..,"Wy":..,"Wz":..}}。'
                                       'A 截面积 m²，Iy/Iz 惯性矩 m⁴，J 扭转常数 m⁴，'
                                       'Wy/Wz 截面模量 m³。',
                        'additionalProperties': {'type': 'object'}},
                    'steel_grade': {
                        'type': 'string',
                        'description': '钢材牌号（Q235/Q355/Q390/Q420，默认 Q355）'},
                    'ref_span': {
                        'type': 'number',
                        'description': '挠度参考跨度 m（可选；缺省取最长杆件）'},
                },
                'required': ['nodes', 'members'],
            },
        },
    }

    SYSTEM_PROMPT = (
        "你是空间刚架结构智能计算助手。根据用户描述选择建模工具：\n"
        "1) 规则矩形网格刚架（等跨等层高的框架）→ build_space_frame，"
        "参数不完整时合理补全（默认 6×4×3m、柱 HW200、梁 HN300、单层、Q355）；\n"
        "2) 桁架桥 → build_truss_bridge（桥长24m、桥宽5m、桁高3m、6个节间、"
        "弦杆HW200、腹杆HW150、Q355）；\n"
        "3) 其他任意构型（塔架、悬挑、不规则平面、任意空间结构）→ build_custom_model，"
        "把结构离散为节点坐标与杆件连接，单位 SI。\n"
        "分析完成后，请用中文向用户解释：结构是否满足 GB50017-2017 要求、"
        "最大位移、最危险构件及其应力比，并给出直观结论。"
        "如果工具调用返回 error，请根据错误信息修正参数后重新调用。"
    )

    def __init__(self, llm: Optional[LLMClient] = None,
                 check_options: Optional[CheckOptions] = None):
        self.llm = llm or LLMClient()
        self.check_options = check_options
        self.last_model: Optional[FrameModel] = None
        self.last_result: Optional[AnalysisResult] = None
        self.last_report: Optional[CheckReport] = None

    # ---------------- 工具执行 ----------------
    def _execute_build(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行 build_space_frame：建模型 -> 求解 -> 校核。"""
        p = FrameParams(**{k: v for k, v in args.items()
                           if k in FrameParams.__dataclass_fields__})
        model = build_frame(p)
        result = solve(model)
        report = check_model(model, result, self.check_options or CheckOptions(
            steel_grade=p.steel_grade))
        self.last_model, self.last_result, self.last_report = model, result, report

        # 返回给 LLM 的结构化摘要
        members = []
        for mid, r in sorted(report.results.items()):
            members.append({
                'id': mid,
                'stress_ratio': round(r.stress_ratio, 3),
                'stability_ratio': round(r.stability_ratio, 3),
                'slenderness': round(r.slenderness, 1),
                'ok': r.ok,
            })
        return {
            'model': {
                'nodes': model.num_nodes, 'members': model.num_members,
                'span': round(report.span, 2),
            },
            'max_displacement_mm': round(report.max_displacement * 1000, 2),
            'deflection_ratio': round(report.max_deflection_ratio, 3),
            'safe': report.safe,
            'worst_member': report.worst_member,
            'worst_ratio': round(report.results[report.worst_member].max_ratio, 3)
            if report.worst_member else None,
            'members': members,
        }

    def _execute_custom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行 build_custom_model：任意构型建模 -> 求解 -> 校核。

        支持三种入参形态：
        - args 本身即 spec（LLM 直接输出模型对象）
        - args['spec'] 为 spec
        - args['spec'] 为 JSON 字符串
        """
        spec = args.get('spec', args)
        if isinstance(spec, str):
            spec = json.loads(spec)
        if not isinstance(spec, dict) or 'nodes' not in spec:
            raise ValueError(
                "build_custom_model 参数应为模型描述对象（含 nodes/members），"
                "如 {\"nodes\":[{\"id\":1,\"x\":0,\"y\":0,\"z\":0},...],"
                "\"members\":[{\"i\":1,\"j\":2,\"section\":\"HW200\"},...]}")
        model = build_custom_model(spec)
        result = solve(model)
        steel = spec.get('steel_grade', 'Q355')
        opts = CheckOptions(steel_grade=steel, ref_span=spec.get('ref_span'))
        report = check_model(model, result, opts)
        self.last_model, self.last_result, self.last_report = model, result, report

        members = []
        for mid, r in sorted(report.results.items()):
            members.append({
                'id': mid,
                'stress_ratio': round(r.stress_ratio, 3),
                'stability_ratio': round(r.stability_ratio, 3),
                'slenderness': round(r.slenderness, 1),
                'ok': r.ok,
            })
        return {
            'model': {
                'nodes': model.num_nodes, 'members': model.num_members,
                'span': round(report.span, 2),
                'summary': spec_summary(spec),
            },
            'max_displacement_mm': round(report.max_displacement * 1000, 2),
            'deflection_ratio': round(report.max_deflection_ratio, 3),
            'safe': report.safe,
            'worst_member': report.worst_member,
            'worst_ratio': round(report.results[report.worst_member].max_ratio, 3)
            if report.worst_member else None,
            'members': members,
        }

    # ---------------- 主流程 ----------------
    def ask(self, user_text: str) -> str:
        """处理用户自然语言请求，返回 Agent 的最终回复。

        路径：桁架桥关键词 -> LLM(三工具) -> 规则刚架 -> 自由文本建模
        """
        self._tool_result_cache = {}
        # 桁架桥分支
        if ('桁架桥' in user_text or '桁架' in user_text and '桥' in user_text
                or 'truss' in user_text.lower()):
            return self._ask_bridge(user_text)

        # 方案 A：LLM Function Calling（含 build_custom_model 任意构型）
        if self.llm.available:
            reply = self._ask_with_llm(user_text)
            if reply:
                return reply
            print("[Agent] LLM 链路异常，规则解析兜底")

        # 方案 B：规则解析兜底（无 Key 或 LLM 失败）
        # B2 优先：文本带"节点:/杆件:"分节标记 -> 自由文本建模
        looks_like_table = (('节点:' in user_text or '节点：' in user_text)
                            and ('杆件' in user_text or '单元' in user_text))
        if looks_like_table:
            try:
                spec = parse_custom_text(user_text)
                tool_result = self._execute_custom({'spec': spec})
                return self._format_tool_report(tool_result)
            except ValueError as e:
                return (f"未能解析你提供的节点/杆件表：{e}\n"
                        f"格式示例：\n节点:\n1 0 0 0\n2 6 0 0\n杆件:\n1-2 HW200\n"
                        f"支座:\n1 固定\n荷载:\n2 FZ=-20000")
        # B1: 规则刚架模板
        try:
            params = parse_params_rules(user_text)
            tool_result = self._execute_build(vars(params))
            return self._format_tool_report(tool_result)
        except ValueError:
            pass
        return (f"未能理解你的描述。你可以：\n"
                f"· 描述规则刚架，如\u201c做一个 8×6×4m 的两层刚架，柱 HW300，梁 HN400，顶部荷载 20kN/m²\u201d\n"
                f"· 描述桁架桥，如\u201c做一座 24m 跨的桁架桥\u201d\n"
                f"· 描述任意构型，如\u201c一个 6m 高的四角锥塔架，底部四角固定，顶部受 200kN 竖向荷载\u201d\n"
                f"· 或直接在\u2018自由建模\u2019面板粘贴节点/杆件表")

    # ---------------- 桁架桥流程 ----------------
    BRIDGE_TOOLS = [{
        'type': 'function',
        'function': {
            'name': 'build_truss_bridge',
            'description': '根据用户描述构建下承式空间简支桁架桥并完成有限元分析与规范校核',
            'parameters': {
                'type': 'object',
                'properties': {
                    'L': {'type': 'number', 'description': '桥长(m)'},
                    'W': {'type': 'number', 'description': '桥宽(m)'},
                    'H': {'type': 'number', 'description': '桥高/桁高(m)'},
                    'n_panels': {'type': 'integer', 'description': '节间数'},
                    'chord_section': {'type': 'string',
                                      'description': '弦杆截面（如HW200）'},
                    'web_section': {'type': 'string',
                                    'description': '腹杆截面（如HW150）'},
                    'load_kn_m2': {'type': 'number',
                                   'description': '桥面均布荷载(kN/m²)'},
                    'steel_grade': {'type': 'string',
                                    'description': '钢材牌号'},
                },
                'required': [],
            },
        },
    }]

    BRIDGE_SYSTEM_PROMPT = (
        "你是桁架桥结构智能计算助手。用户描述不完整时合理补全"
        "（默认桥长24m、桥宽5m、桁高3m、6个节间、弦杆HW200、腹杆HW150、Q355），"
        "然后调用 build_truss_bridge 工具完成分析。"
        "分析完成后用中文解释：是否满足 GB50017-2017、最大位移（下挠）、"
        "最危险杆件（弦杆/腹杆）及其应力比，并给出直观结论。"
    )

    def _execute_bridge(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行 build_truss_bridge：建模 -> 求解 -> 校核。"""
        p = TrussBridgeParams(**{k: v for k, v in args.items()
                                 if k in TrussBridgeParams.__dataclass_fields__})
        model = build_truss_bridge(p)
        result = solve(model)
        # 桁架桥挠度参考跨度 = 桥跨（而非最长杆件）
        opts = CheckOptions(steel_grade=p.steel_grade, ref_span=p.L)
        report = check_model(model, result, opts)
        self.last_model, self.last_result, self.last_report = model, result, report
        members = []
        for mid, r in sorted(report.results.items()):
            members.append({
                'id': mid,
                'stress_ratio': round(r.stress_ratio, 3),
                'stability_ratio': round(r.stability_ratio, 3),
                'slenderness': round(r.slenderness, 1),
                'ok': r.ok,
            })
        return {
            'model': {
                'nodes': model.num_nodes, 'members': model.num_members,
                'span': round(p.L, 2),
            },
            'max_displacement_mm': round(report.max_displacement * 1000, 2),
            'deflection_ratio': round(report.max_deflection_ratio, 3),
            'safe': report.safe,
            'worst_member': report.worst_member,
            'worst_ratio': round(report.results[report.worst_member].max_ratio, 3)
            if report.worst_member else None,
            'members': members,
        }

    def _ask_bridge(self, user_text: str) -> str:
        """桁架桥：LLM 解析优先，规则解析兜底。"""
        if self.llm.available:
            messages = [
                {'role': 'system', 'content': self.BRIDGE_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_text},
            ]
            resp = self.llm.chat(messages, tools=self.BRIDGE_TOOLS)
            if resp:
                try:
                    msg = resp['choices'][0]['message']
                    if msg.get('tool_calls'):
                        args = json.loads(
                            msg['tool_calls'][0]['function']['arguments'])
                        tool_result = self._execute_bridge(args)
                        messages.append(msg)
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': msg['tool_calls'][0]['id'],
                            'content': json.dumps(tool_result, ensure_ascii=False),
                        })
                        final = self.llm.chat(messages, temperature=0.4)
                        if final:
                            return final['choices'][0]['message']['content']
                        return self._format_bridge_report(tool_result)
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
        # 规则解析兜底
        params = parse_bridge_rules(user_text)
        tool_result = self._execute_bridge(vars(params))
        return self._format_bridge_report(tool_result)

    def _format_bridge_report(self, t: Dict[str, Any]) -> str:
        """桁架桥中文报告（无 LLM 时）。"""
        m = t['model']
        lines = [
            "已按你的描述完成空间桁架桥分析：",
            f"· 桥跨：{m['span']} m，{m['nodes']} 个节点，{m['members']} 根杆件",
            f"· 最大下挠：{t['max_displacement_mm']} mm"
            f"（挠跨比 {t['deflection_ratio']:.2f}）",
        ]
        lines.append("· 规范校核：全部构件满足 GB50017-2017，结构安全 ✓"
                     if t['safe'] else
                     f"· 规范校核：不满足，最不利杆件 {t['worst_member']} 号，"
                     f"最大控制指标 {t['worst_ratio']}（超限）")
        for mb in t['members'][:10]:
            lines.append(
                f"    - 杆件{mb['id']}: 强度 {mb['stress_ratio']:.2f} / "
                f"稳定 {mb['stability_ratio']:.2f} / λ {mb['slenderness']:.0f} "
                f"{'OK' if mb['ok'] else '超限'}")
        return "\n".join(lines)

    def _ask_with_llm(self, user_text: str) -> Optional[str]:
        """LLM Function Calling 主链路：注册全部建模工具，支持多轮修正。

        工具调用返回 error 时把错误回传给 LLM 让它修正参数（最多 3 轮），
        防止幻觉导致的无效构型。
        """
        tools = self.TOOLS + [self.CUSTOM_TOOL]
        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text},
        ]
        for _round in range(3):
            resp = self.llm.chat(messages, tools=tools)
            if not resp:
                return None
            try:
                msg = resp['choices'][0]['message']
            except (KeyError, IndexError):
                return None

            # 未触发工具调用：把内容按规则刚架参数解释（最后交给 LLM 总结）
            if not msg.get('tool_calls'):
                args = parse_params_rules(msg.get('content') or user_text)
                try:
                    tool_result = self._execute_build(vars(args))
                except ValueError as e:
                    tool_result = {'error': str(e)}
                messages.append(msg)
                messages.append({'role': 'tool', 'tool_call_id': '0',
                                 'content': json.dumps(tool_result,
                                                       ensure_ascii=False)})
                final = self.llm.chat(messages, temperature=0.4)
                if final:
                    try:
                        return final['choices'][0]['message']['content']
                    except (KeyError, IndexError):
                        pass
                return self._format_tool_report(tool_result)

            # 有工具调用：执行（取第一个 tool call）
            tc = msg['tool_calls'][0]
            try:
                args = json.loads(tc['function']['arguments'] or '{}')
            except json.JSONDecodeError:
                args = {}
            fn = tc['function']['name']
            try:
                if fn == 'build_custom_model':
                    tool_result = self._execute_custom(args)
                elif fn == 'build_truss_bridge':
                    tool_result = self._execute_bridge(args)
                else:
                    tool_result = self._execute_build(args)
            except ValueError as e:
                tool_result = {'error': str(e)}

            messages.append(msg)
            messages.append({
                'role': 'tool',
                'tool_call_id': tc['id'],
                'content': json.dumps(tool_result, ensure_ascii=False),
            })
            self._tool_result_cache = tool_result

            # 有错误 -> 继续循环让 LLM 修正；成功 -> 取最终总结
            if tool_result.get('error'):
                continue
            final = self.llm.chat(messages, temperature=0.4)
            if not final:
                return None
            try:
                return final['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return None

        # 三轮均失败
        last = self._last_tool_result.get('error', '模型参数无法通过程序校验')
        return (f"抱歉，自动建模在多次尝试后仍未成功：{last}。\n"
                f"你可以换一种描述，或改用'自由建模'直接粘贴节点/杆件表。")

    @property
    def _last_tool_result(self) -> Dict[str, Any]:
        return getattr(self, '_tool_result_cache', {})

    def _ask_with_rules(self, user_text: str) -> str:
        params = parse_params_rules(user_text)
        tool_result = self._execute_build(vars(params))
        return self._format_report(tool_result)

    def _format_tool_report(self, tool_result: Dict[str, Any]) -> str:
        """统一结构化结果 -> 中文报告（刚架 / 桁架桥 / 任意构型通用）。

        若结果带 error 键，直接回显错误（供无 LLM 兜底时提示用户）。
        """
        if tool_result.get('error'):
            return f"模型校验未通过：{tool_result['error']}"
        m = tool_result['model']
        lines = [
            f"已按你的描述完成结构分析：",
            f"· 模型：{m['nodes']} 个节点，{m['members']} 根杆件，"
            f"参考跨度 {m['span']} m",
            f"· 最大位移：{tool_result['max_displacement_mm']} mm"
            f"（位移比 {tool_result['deflection_ratio']:.2f}）",
        ]
        if tool_result['safe']:
            lines.append("· 规范校核：全部构件满足 GB50017-2017 要求，结构安全 ✓")
        else:
            lines.append(
                f"· 规范校核：结构**不满足**要求，最不利构件为 "
                f"{tool_result['worst_member']} 号，最大控制指标 "
                f"{tool_result['worst_ratio']}（超限）")
        lines.append("· 主要构件应力比：")
        for mb in tool_result['members'][:10]:
            lines.append(
                f"    - 杆件{mb['id']}: 强度 {mb['stress_ratio']:.2f} / "
                f"稳定 {mb['stability_ratio']:.2f} / λ {mb['slenderness']:.0f} "
                f"{'OK' if mb['ok'] else '超限'}")
        return "\n".join(lines)

    def _format_report(self, tool_result: Dict[str, Any]) -> str:
        """把结构化结果格式化为中文报告（无 LLM 时的最终回复）。"""
        return self._format_tool_report(tool_result)
