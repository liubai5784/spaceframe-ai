"""
空间刚架智能计算 Agent —— Web 后端（FastAPI）
=============================================
接口：
- GET  /              前端页面
- POST /api/analyze   自然语言 -> 分析（返回模型 + 结果 + Agent 回复）
- POST /api/upload    .s2k 文件 -> 分析（同上）
- GET  /api/health    健康检查

启动：
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8000
（配置 LLM：环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL，未配置则规则模式）
"""

from __future__ import annotations

import io
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import SpaceFrameAgent, LLMClient
from src.s2k_parser import parse_s2k
from src.fem import solve
from src.checks import check_model, CheckOptions

app = FastAPI(title="空间刚架智能计算 Agent", version="0.4.0")

# 跨域放行：支持前端托管在 Cloudflare Pages / 其他域名时的 API 调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = os.path.join(os.path.dirname(__file__), '..', 'web')
app.mount('/vendor', StaticFiles(directory=os.path.join(WEB_DIR, 'vendor')),
          name='vendor')


class AnalyzeRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# 序列化：模型/结果 -> 前端 JSON
# ---------------------------------------------------------------------------

def _model_json(agent: SpaceFrameAgent) -> dict:
    """把 agent 最近一次分析的模型、结果、校核报告转为前端可渲染 JSON。"""
    model = agent.last_model
    result = agent.last_result
    report = agent.last_report
    if model is None or result is None or report is None:
        return {}

    nodes = [{'id': nid, 'x': n.x, 'y': n.y, 'z': n.z}
             for nid, n in sorted(model.nodes.items())]
    # 节点位移（mm）
    disp = {nid: [round(v, 4) for v in result.node_displacement(nid)[:3]]
            for nid in model.nodes}

    members = []
    for mid, mem in sorted(model.members.items()):
        r = report.results[mid]
        members.append({
            'id': mid, 'i': mem.node_i, 'j': mem.node_j,
            'stress_ratio': round(r.stress_ratio, 3),
            'stability_ratio': round(r.stability_ratio, 3),
            'slenderness': round(r.slenderness, 1),
            'max_ratio': round(r.max_ratio, 3),
            'ok': r.ok,
            'N_kN': round(r.forces['N'] / 1e3, 1),
            'Mz_kNm': round(r.forces['Mz'] / 1e3, 1),
        })

    worst = report.worst_member
    return {
        'model': {
            'nodes': nodes,
            'members': members,
            'num_nodes': model.num_nodes,
            'num_members': model.num_members,
            'displacements': disp,
            'deformed_scale': _auto_deform_scale(model, result),
        },
        'result': {
            'max_displacement_mm': round(report.max_displacement * 1000, 2),
            'deflection_ratio': round(report.max_deflection_ratio, 3),
            'span': round(report.span, 2),
            'safe': report.safe,
            'worst_member': worst,
            'worst_ratio': round(report.results[worst].max_ratio, 3) if worst else None,
        },
    }


def _auto_deform_scale(model, result) -> float:
    """自动变形放大系数：目标最大可视位移约 0.5m。"""
    box = [1e9, 1e9, 1e9, -1e9, -1e9, -1e9]
    for n in model.nodes.values():
        box[0] = min(box[0], n.x); box[1] = min(box[1], n.y); box[2] = min(box[2], n.z)
        box[3] = max(box[3], n.x); box[4] = max(box[4], n.y); box[5] = max(box[5], n.z)
    diag = max((box[3]-box[0], box[4]-box[1], box[5]-box[2]), default=1.0)
    max_d = result.max_displacement()
    if max_d < 1e-9:
        return 0.0
    return max(diag * 0.15 / max_d, 1.0)


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(WEB_DIR, 'index.html')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h1>前端页面缺失</h1>")


@app.get("/api/health")
def health():
    return {'status': 'ok',
            'llm': LLMClient().available}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    agent = SpaceFrameAgent()
    try:
        reply = agent.ask(req.text)
    except ValueError as e:
        # 参数解析/截面库不识别等业务错误 -> 返回友好中文提示（而非 500）
        from src import sections_db
        avail = '、'.join(sorted(sections_db._H_DIMS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"{e}。可用的截面：{avail}。示例：柱HW300 梁HN400",
        )
    return {
        'reply': reply,
        'data': _model_json(agent),
    }


@app.post("/api/upload")
async def upload(file: UploadFile):
    content = await file.read()
    suffix = os.path.splitext(file.filename or 'model.s2k')[1] or '.s2k'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        model = parse_s2k(tmp_path)
        result = solve(model)
        report = check_model(model, result, CheckOptions(steel_grade='Q355'))
        agent = SpaceFrameAgent()
        agent.last_model, agent.last_result, agent.last_report = model, result, report
        reply = report.summary()
        return {'reply': reply, 'data': _model_json(agent)}
    finally:
        os.unlink(tmp_path)
