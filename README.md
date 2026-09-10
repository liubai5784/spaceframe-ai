# 基于 AI 的空间刚架智能计算 Agent

课程项目：基于 AI 的空间刚架智能计算 Agent 开发（与大语言模型结合）

## 项目结构

```
spaceframe-ai/
├── src/
│   ├── model.py        # 数据模型：节点/杆件/截面/材料/荷载/支座
│   ├── fem.py          # 有限元内核：12×12 单元矩阵、组装、KU=F、杆端内力
│   ├── s2k_parser.py   # SAP2000 .s2k 文件解析导入
│   ├── checks.py        # GB50017 规范校核（强度/稳定/长细比/挠度）
│   ├── sections_db.py   # 常用 H 型钢截面库（GB/T 11263）
│   ├── templates.py     # 参数化空间刚架生成器（规则刚架 / 桁架桥）
│   ├── custom_model.py  # 通用构型建模器：任意节点/杆件/支座/荷载（M5 新增）
│   └── agent.py         # LLM Agent（Function Calling + 规则降级）
│   └── __init__.py
├── examples/
│   ├── verify_solver.py  # 7 组经典算例验证（与解析解对拍）
│   ├── sample_frame.s2k  # 示例 SAP2000 模型文件
│   ├── test_s2k.py       # .s2k 解析器往返测试
│   ├── test_checks.py    # GB50017 校核模块测试
│   ├── test_agent.py     # Agent 全链路测试
│   └── test_custom.py    # 通用构型建模测试（M5 新增）
├── app/
│   └── main.py          # FastAPI Web 后端（分析/上传/通用建模接口）
├── web/
│   ├── index.html       # 前端：Three.js 3D + 对话 + 自由建模
│   └── vendor/          # three.js 本地依赖（离线可用）
└── requirements.txt
```

## 运行验证

```bash
pip install -r requirements.txt
python examples/verify_solver.py   # 有限元内核验证
python examples/test_s2k.py        # .s2k 解析器测试
python examples/test_checks.py     # 规范校核测试
python examples/test_agent.py      # Agent 全链路测试
python examples/test_custom.py     # 通用构型建模测试（M5）
```

## Web 界面

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://<服务器>:8000
# 可选：export LLM_API_KEY=... 启用大模型（未配置时自动使用规则解析）
```

## 通用构型建模（M5）

早期版本只能按内置模板（规则矩形刚架 / 桁架桥）建模；M5 起支持**任意构型**，
共三条入口（Web 端「自由建模」页签 / 后端 / 自然语言）：

**① 自然语言描述任意构型（推荐，需 LLM）**
LLM 通过 `build_custom_model` 工具把描述离散为节点坐标与杆件连接，
程序侧强校验（节点存在性 / 截面库 / 支座 / 孤立节点）后求解。
例："一个 6m 高的四角锥塔架，底部四角固定，顶部受 200kN 竖向荷载"

**② 直接提交模型 JSON（`POST /api/model`，Web「自由建模」页签）**
```json
{
  "units": {"length": "m", "force": "kN"},
  "nodes": [{"id":1,"x":0,"y":0,"z":0}, {"id":2,"x":6,"y":0,"z":0}],
  "members": [{"i":1,"j":2,"section":"HW200"}],
  "supports": [{"node":1}],
  "nodal_loads": [{"node":2,"fz":-100}],
  "steel_grade": "Q355",
  "ref_span": 6.0
}
```
- 单位：默认 SI（m/N/N·m/N·m），`units` 可切 `mm`/`kN`
- `nodes.id` 可省略（自动编号）；`supports.fix` 省略=全固定，也接受"固定/铰接"
- `section` 支持截面库名（HW200/HN300…）与自定义 `sections` 参数
- 校验失败返回中文错误（未知截面、节点缺失、无支座、孤立节点等）

**③ 文本表（无 LLM 兜底，`/api/analyze` 直接粘贴）**
```
单位: m kN
节点:
1 0 0 0
2 6 0 0
杆件:
1-2 HW200
支座:
1 固定
荷载:
2 FZ=-100
```

## 已验证内容（M1）

| # | 算例 | 验证点 | 误差 |
|---|------|--------|------|
| 1 | 悬臂梁端部集中力 | 弯曲位移/转角 | 0.0000% |
| 2 | 悬臂梁端部弯矩 | 弯曲转角 | 0.0000% |
| 3 | 简支梁跨中集中力 | 弯曲位移 | 0.0000% |
| 4 | 简支梁均布荷载 | 等效节点荷载 | 0.0000% |
| 5 | 轴向拉杆 | 轴向变形 | 0.0000% |
| 6 | 纯扭杆 | 扭转变形 | 0.0000% |
| 7 | 平面门式刚架 | 力/力矩整体平衡 | 残差 ~1e-10 |

## 单元理论

空间刚架单元：每节点 6 自由度（3 平动 + 3 转动），12×12 刚度矩阵，
含轴向、扭转、双向弯曲；坐标变换采用局部 z 轴尽量靠近全局 Z 轴的默认定向
（杆件接近竖直时以全局 Y 为参考），支持 gamma 角绕局部 x 轴旋转截面。
