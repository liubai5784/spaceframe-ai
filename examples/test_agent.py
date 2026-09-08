"""
Agent 层测试
============
验证（无 API Key 时走规则解析链路）：
1. 截面库数据完整性
2. 参数化刚架生成（节点/杆件/支座/荷载数量）
3. Agent 自然语言 -> 建模 -> 求解 -> 校核 -> 中文报告 全链路

运行：python examples/test_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import SpaceFrameAgent, LLMClient, parse_params_rules
from src.templates import FrameParams, build_frame
from src.fem import solve
from src.checks import check_model, CheckOptions
from src import sections_db


def test_section_db():
    print("[1] 截面库")
    for name in ['HW150', 'HW200', 'HW300', 'HN300', 'HN400', 'HN500']:
        p = sections_db.section_properties(name)
        assert p and p['A'] > 0 and p['Iz'] > p['Iy'] > 0, name
        print(f"    {name}: A={p['A']*1e4:.2f}cm² Iz={p['Iz']*1e8:.0f}cm⁴ "
              f"Iy={p['Iy']*1e8:.0f}cm⁴")
    # 别名解析
    assert sections_db.resolve('H200') == 'HW200'
    assert sections_db.resolve('HN300') == 'HN300'
    print("    通过 ✓")


def test_template():
    print("[2] 参数化刚架生成")
    p = FrameParams(Lx=6, Ly=4, Lz=3, nx=2, ny=2, nz=1,
                    load_kn_m2=20.0)
    m = build_frame(p)
    # 节点：(nz+1)*(nx+1)*(ny+1) = 2*3*3 = 18
    assert m.num_nodes == 18, m.num_nodes
    # 柱：nz*(nx+1)*(ny+1)=9；X梁：nz*(ny+1)*nx=6；Y梁：nz*(nx+1)*ny=6 → 21
    assert m.num_members == 21, m.num_members
    assert len(m.supports) == 9
    assert len(m.nodal_loads) == 9          # 顶部 3×3 节点
    print(f"    节点 {m.num_nodes}，杆件 {m.num_members}，支座 {len(m.supports)}，"
          f"节点荷载 {len(m.nodal_loads)}")
    # 求解 + 校核应可运行
    r = solve(m)
    report = check_model(m, r, CheckOptions(steel_grade='Q355'))
    print(f"    求解正常，最大位移 {report.max_displacement*1000:.2f} mm，"
          f"整体结论: {report.safe}")
    assert report.max_displacement > 0
    print("    通过 ✓")


def test_rules_parser():
    print("[3] 规则解析")
    p = parse_params_rules(
        "设计一个5×4×3m的两层空间刚架，柱H200，梁HN400，顶部荷载20kN/m²，Q355")
    print(f"    Lx={p.Lx} Ly={p.Ly} Lz={p.Lz} nz={p.nz} "
          f"柱={p.column_section} 梁={p.beam_section} "
          f"荷载={p.load_kn_m2}kN/m² 钢号={p.steel_grade}")
    assert p.Lx == 5 and p.Ly == 4 and p.Lz == 3
    assert p.nz == 2
    assert p.column_section == 'HW200'
    assert p.beam_section == 'HN400'
    assert p.load_kn_m2 == 20.0
    assert p.steel_grade == 'Q355'
    print("    通过 ✓")


def test_agent_pipeline():
    print("[4] Agent 全链路（规则模式）")
    agent = SpaceFrameAgent(llm=LLMClient())     # 无 Key -> 规则模式
    reply = agent.ask("做一个6×4×3m的单层刚架，柱HW200，梁HN300，顶部荷载15kN/m²")
    print(reply)
    assert agent.last_model is not None
    assert agent.last_report is not None
    assert '最大位移' in reply and '杆件' in reply
    print("    通过 ✓")


if __name__ == "__main__":
    test_section_db()
    print("-" * 50)
    test_template()
    print("-" * 50)
    test_rules_parser()
    print("-" * 50)
    test_agent_pipeline()
    print("-" * 50)
    print("\nAgent 层测试全部通过 ✔")
