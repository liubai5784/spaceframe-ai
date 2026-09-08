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
from .templates import FrameParams, build_frame
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
            'description': '根据用户描述构建空间刚架模型并完成有限元分析与规范校核',
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

    SYSTEM_PROMPT = (
        "你是空间刚架结构智能计算助手。用户的描述可能不完整，请合理推断并补全参数"
        "（尺寸默认 6×4×3m、柱 HW200、梁 HN300、单层、Q355），"
        "然后调用 build_space_frame 工具完成分析。"
        "分析完成后，请用中文向用户解释：结构是否满足 GB50017-2017 要求、"
        "最大位移、最危险构件及其应力比，并给出直观结论。"
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

    # ---------------- 主流程 ----------------
    def ask(self, user_text: str) -> str:
        """处理用户自然语言请求，返回 Agent 的最终回复。"""
        # 方案 A：LLM Function Calling
        if self.llm.available:
            reply = self._ask_with_llm(user_text)
            if reply:
                return reply
            print("[Agent] LLM 链路异常，使用规则解析兜底")

        # 方案 B：规则解析（无 Key 或 LLM 失败）
        return self._ask_with_rules(user_text)

    def _ask_with_llm(self, user_text: str) -> Optional[str]:
        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text},
        ]
        resp = self.llm.chat(messages, tools=self.TOOLS)
        if not resp:
            return None
        try:
            msg = resp['choices'][0]['message']
        except (KeyError, IndexError):
            return None

        # 处理工具调用
        if msg.get('tool_calls'):
            try:
                args = json.loads(msg['tool_calls'][0]['function']['arguments'])
            except (json.JSONDecodeError, KeyError, IndexError):
                args = {}
            tool_result = self._execute_build(args)
            messages.append(msg)
            messages.append({
                'role': 'tool',
                'tool_call_id': msg['tool_calls'][0]['id'],
                'content': json.dumps(tool_result, ensure_ascii=False),
            })
            final = self.llm.chat(messages, temperature=0.4)
            if not final:
                return None
            try:
                return final['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return None

        # 未触发工具调用：把内容解释为参数
        args = parse_params_rules(msg.get('content') or user_text)
        tool_result = self._execute_build(vars(args))
        messages.append(msg)
        messages.append({'role': 'tool', 'tool_call_id': '0',
                         'content': json.dumps(tool_result, ensure_ascii=False)})
        final = self.llm.chat(messages, temperature=0.4)
        if final:
            try:
                return final['choices'][0]['message']['content']
            except (KeyError, IndexError):
                pass
        return self._format_report(tool_result)

    def _ask_with_rules(self, user_text: str) -> str:
        params = parse_params_rules(user_text)
        tool_result = self._execute_build(vars(params))
        return self._format_report(tool_result)

    def _format_report(self, tool_result: Dict[str, Any]) -> str:
        """把结构化结果格式化为中文报告（无 LLM 时的最终回复）。"""
        m = tool_result['model']
        lines = [
            f"已按你的描述完成空间刚架分析：",
            f"· 模型：{m['nodes']} 个节点，{m['members']} 根杆件，参考跨度 {m['span']} m",
            f"· 最大位移：{tool_result['max_displacement_mm']} mm"
            f"（位移比 {tool_result['deflection_ratio']:.2f}）",
        ]
        if tool_result['safe']:
            lines.append("· 规范校核：全部构件满足 GB50017-2017 要求，结构安全 ✓")
        else:
            lines.append(
                f"· 规范校核：结构**不满足**要求，最不利构件为 "
                f"{tool_result['worst_member']} 号，综合应力比 "
                f"{tool_result['worst_ratio']}（超限）")
        lines.append("· 主要构件应力比：")
        for mb in tool_result['members'][:10]:
            lines.append(
                f"    - 杆件{mb['id']}: 强度 {mb['stress_ratio']:.2f} / "
                f"稳定 {mb['stability_ratio']:.2f} / λ {mb['slenderness']:.0f} "
                f"{'OK' if mb['ok'] else '超限'}")
        return "\n".join(lines)
