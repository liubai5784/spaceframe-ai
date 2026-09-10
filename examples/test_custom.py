"""
通用构型建模测试（M5：任意构型支持）
====================================
验证：
1. 通用建模器 build_custom_model：任意节点/杆件/支座/荷载
2. 单位换算（mm/kN 与 SI 等价）
3. 解析解对比（悬臂梁端部集中力）
4. 程序侧校验（未知截面/节点缺失/无支座/孤立节点 -> 中文错误）
5. 自定义截面
6. Agent 层自由文本建模（无 LLM 兜底）

运行：python examples/test_custom.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.custom_model import (build_custom_model, parse_custom_text,
                              example_specs, spec_summary)
from src.agent import SpaceFrameAgent, LLMClient
from src.fem import solve
from src.checks import check_model, CheckOptions


def _analyze(spec: dict):
    model = build_custom_model(spec)
    result = solve(model)
    steel = spec.get('steel_grade', 'Q355')
    report = check_model(model, result,
                         CheckOptions(steel_grade=steel,
                                      ref_span=spec.get('ref_span')))
    return model, result, report


def test_tower():
    print("[1] 通用建模器：四角锥塔架")
    spec = example_specs()['塔架']
    model, result, report = _analyze(spec)
    print(f"    节点 {model.num_nodes}，杆件 {model.num_members}，"
          f"支座 {len(model.supports)}，安全={report.safe}")
    assert model.num_nodes == 5 and model.num_members == 8
    # 反力平衡：ΣFz = -200kN 的反号
    R = result.support_reactions()
    fz = sum(r[3] for r in R)
    assert abs(fz - 2e5) < 1e-3, f"反力不平衡: {fz}"
    print(f"    反力合力 Fz={fz:.1f} N（与荷载平衡 ✓）")
    print("    通过 ✓")


def test_lframe():
    print("[2] 通用建模器：L 形平面框架（不规则构型）")
    spec = example_specs()['L形框架']
    model, result, report = _analyze(spec)
    print(f"    节点 {model.num_nodes}，杆件 {model.num_members}，"
          f"最大位移 {report.max_displacement*1000:.2f} mm")
    assert model.num_members == 10
    assert report.max_displacement > 0
    print("    通过 ✓")


def test_units_and_analytic():
    print("[3] 单位换算 + 解析解：悬臂梁端部集中力（mm/kN）")
    # 解析解：w = F·L³/(3·E·Iz)，HW200 Iz=4720cm⁴（但水平杆局部 z 向受弯用弱轴 Iy=1600cm⁴）
    # 悬臂 FZ=-100kN，L=6m：w = 1e5·216/(3·2.06e11·1600e-8) = 2.1845 m
    spec = {
        'units': {'length': 'mm', 'force': 'kN'},
        'nodes': [{'id': 1, 'x': 0, 'y': 0, 'z': 0},
                  {'id': 2, 'x': 6000, 'y': 0, 'z': 0}],
        'members': [{'i': 1, 'j': 2, 'section': 'HW200'}],
        'supports': [{'node': 1, 'fix': [True] * 6}],
        'nodal_loads': [{'node': 2, 'fz': -100}],
    }
    model, result, _ = _analyze(spec)
    w_num = abs(float(result.node_displacement(2)[2]))       # m
    w_ana = 1e5 * 6.0**3 / (3 * 2.06e11 * 1600e-8)           # m
    print(f"    数值 {w_num*1000:.2f} mm，解析 {w_ana*1000:.2f} mm，"
          f"误差 {abs(w_num-w_ana)/w_ana*100:.4f}%")
    assert abs(w_num - w_ana) / w_ana < 1e-6
    print("    通过 ✓")


def test_text_parser():
    print("[4] 自由文本解析")
    txt = """单位: mm kN
节点:
1 0 0 0
2 6000 0 0
3 6000 4000 0
杆件:
1-2 HW200
2-3 HN300
支座:
1 固定
荷载:
3 FZ=-100"""
    spec = parse_custom_text(txt)
    assert spec['units'] == {'length': 'mm', 'force': 'kN'}
    assert len(spec['nodes']) == 3 and len(spec['members']) == 2
    assert spec['supports'][0]['fix'] == '固定'
    assert spec['nodal_loads'][0]['fz'] == -100.0
    model = build_custom_model(spec)
    assert model.num_nodes == 3 and model.num_members == 2
    print(f"    节点 {len(spec['nodes'])}，杆件 {len(spec['members'])}，"
          f"单位 {spec['units']}")
    print("    通过 ✓")


def test_validation_errors():
    print("[5] 程序侧校验（防 LLM 幻觉/手误）")
    cases = [
        ({'nodes': [{'x': 0, 'y': 0, 'z': 0}, {'x': 6, 'y': 0, 'z': 0}],
          'members': [{'i': 1, 'j': 2, 'section': 'XXX'}]},
         '未知截面'),
        ({'nodes': [{'x': 0, 'y': 0, 'z': 0}, {'x': 6, 'y': 0, 'z': 0}],
          'members': [{'i': 1, 'j': 99, 'section': 'HW200'}]},
         '节点 99'),
        ({'nodes': [{'x': 0, 'y': 0, 'z': 0}, {'x': 6, 'y': 0, 'z': 0}],
          'members': [{'i': 1, 'j': 2, 'section': 'HW200'}]},
         '缺少支座'),
        ({'nodes': [{'x': 0, 'y': 0, 'z': 0}, {'x': 6, 'y': 0, 'z': 0},
                    {'x': 3, 'y': 3, 'z': 0}],
          'members': [{'i': 1, 'j': 2, 'section': 'HW200'}],
          'supports': [{'node': 1}]},
         '既没有连接任何杆件也没有支座'),
    ]
    for spec, expect in cases:
        try:
            build_custom_model(spec)
            raise AssertionError(f"应报错却成功: {spec}")
        except ValueError as e:
            assert expect in str(e), f"错误信息不符: {e}"
            print(f"    ✓ {expect}: {e}")
    print("    通过 ✓")


def test_custom_section():
    print("[6] 自定义截面")
    spec = {
        'nodes': [{'id': 1, 'x': 0, 'y': 0, 'z': 0},
                  {'id': 2, 'x': 6, 'y': 0, 'z': 0}],
        'members': [{'i': 1, 'j': 2, 'section': 'BOX300'}],
        'supports': [{'node': 1}],
        'nodal_loads': [{'node': 2, 'fz': -50000}],
        'sections': {'BOX300': {'A': 0.02, 'Iy': 1e-4, 'Iz': 1e-4,
                                'J': 1e-4, 'Wy': 1e-3, 'Wz': 1e-3}},
    }
    model, result, report = _analyze(spec)
    assert 'BOX300' in model.sections
    assert report.max_displacement > 0
    print(f"    自定义截面 BOX300 求解正常，位移 "
          f"{report.max_displacement*1000:.2f} mm")
    print("    通过 ✓")


def test_agent_custom():
    print("[7] Agent 自由文本建模（无 LLM 兜底）")
    agent = SpaceFrameAgent(llm=LLMClient())
    txt = """单位: m kN
节点:
1 0 0 0
2 6 0 0
杆件:
1-2 HW200
支座:
1 固定
荷载:
2 FZ=-100"""
    reply = agent.ask(txt)
    print(reply.split("\n")[0])
    print(reply.split("\n")[1])
    assert agent.last_model is not None
    assert agent.last_model.num_members == 1
    assert '最大位移' in reply and '杆件' in reply
    print("    通过 ✓")


def test_agent_custom_execute():
    print("[8] Agent 直接执行 build_custom_model（LLM 工具等价）")
    agent = SpaceFrameAgent(llm=LLMClient())
    spec = example_specs()['L形框架']
    tool_result = agent._execute_custom({'spec': spec})
    assert 'error' not in tool_result
    assert tool_result['model']['members'] == 10
    assert 'max_displacement_mm' in tool_result
    # 非法参数 -> 抛出 ValueError（_ask_with_llm 会捕获并转为 error 键反馈给 LLM 修正）
    try:
        agent._execute_custom({'spec': {'nodes': [{'x': 0, 'y': 0, 'z': 0}]}})
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert '节点' in str(e)
        bad_msg = str(e)
    print(f"    正常执行：{tool_result['model']['summary']}")
    print(f"    非法输入报错：{bad_msg[:40]}…")
    print("    通过 ✓")


if __name__ == "__main__":
    test_tower()
    print("-" * 50)
    test_lframe()
    print("-" * 50)
    test_units_and_analytic()
    print("-" * 50)
    test_text_parser()
    print("-" * 50)
    test_validation_errors()
    print("-" * 50)
    test_custom_section()
    print("-" * 50)
    test_agent_custom()
    print("-" * 50)
    test_agent_custom_execute()
    print("-" * 50)
    print("\n通用构型建模测试全部通过 ✔")
