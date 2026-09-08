"""空间刚架智能计算 Agent —— 核心求解模块。"""

from .model import FrameModel
from .fem import (solve, AnalysisResult, local_stiffness,
                  rotation_matrix, transformation_matrix,
                  global_stiffness, member_internal_forces,
                  member_section_forces, member_extreme_forces)
from .s2k_parser import parse_s2k, S2KError
from .checks import check_model, CheckOptions, CheckReport, stability_factor
from .sections_db import section_properties, all_sections, resolve as resolve_section
from .templates import FrameParams, build_frame
from .agent import SpaceFrameAgent, LLMClient, parse_params_rules

__all__ = [
    "FrameModel", "solve", "AnalysisResult", "local_stiffness",
    "rotation_matrix", "transformation_matrix", "global_stiffness",
    "member_internal_forces", "member_section_forces",
    "member_extreme_forces", "parse_s2k", "S2KError",
    "check_model", "CheckOptions", "CheckReport", "stability_factor",
    "section_properties", "all_sections", "resolve_section",
    "FrameParams", "build_frame",
    "SpaceFrameAgent", "LLMClient", "parse_params_rules",
]

__version__ = "0.4.0"
