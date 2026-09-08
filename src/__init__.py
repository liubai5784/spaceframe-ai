"""空间刚架智能计算 Agent —— 核心求解模块。"""

from .model import FrameModel
from .fem import (solve, AnalysisResult, local_stiffness,
                  rotation_matrix, transformation_matrix,
                  global_stiffness, member_internal_forces)

__all__ = [
    "FrameModel", "solve", "AnalysisResult", "local_stiffness",
    "rotation_matrix", "transformation_matrix", "global_stiffness",
    "member_internal_forces",
]

__version__ = "0.1.0"
