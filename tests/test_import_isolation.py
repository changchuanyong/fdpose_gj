"""核心层必须在没有 ROS / CUDA 的环境下可导入（离线回放与 CI 守卫）。"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2  # noqa: F401
    import numpy as np
except ModuleNotFoundError:
    np = None

CORE_MODULES = (
    "fdonpose.domain.geometry",
    "fdonpose.application.registration",
    "fdonpose.application.tracking",
    "fdonpose.application.validation",
    "fdonpose.adapters.foundationpose",
    "fdonpose.adapters.ros2",
)

# sys.modules 中置 None 会让对应 import 直接抛 ImportError，从而证明核心不在顶层依赖它们。
BLOCKED = {"rclpy": None, "geometry_msgs": None, "torch": None, "estimater": None}


@unittest.skipIf(np is None, "NumPy/OpenCV 不可用")
class ImportIsolationTest(unittest.TestCase):
    def test_core_imports_without_ros_or_cuda(self):
        saved = {name: sys.modules.pop(name) for name in list(sys.modules)
                 if name == "fdonpose" or name.startswith("fdonpose.")}
        try:
            with patch.dict(sys.modules, BLOCKED):
                for name in CORE_MODULES:
                    with self.subTest(module=name):
                        importlib.import_module(name)
        finally:
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()