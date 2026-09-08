"""空间刚架智能计算 Agent —— 核心求解模块。"""

from .model import FrameModel
from .fem import (solve, AnalysisResult, local_stiffness,
                  rotation_matrix, transformation_matrix,
                  global_stiffness, member_internal_forces)
from .s2k_parser import parse_s2k, S2KError

__all__ = [
    "FrameModel", "solve", "AnalysisResult", "local_stiffness",
    "rotation_matrix", "transformation_matrix", "global_stiffness",
    "member_internal_forces", "parse_s2k", "S2KError",
]

__version__ = "0.2.0"
