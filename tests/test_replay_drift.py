import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, sentinel


class DriftReplayTest(unittest.TestCase):
    def test_detection_box_uses_bgr_and_handles_missing_target(self):
        script = Path(__file__).resolve().parents[1] / "fdpose_ws/src/fdpose_tools/fdpose_tools/replay_drift.py"
        deps = {name: MagicMock() for name in ("cv2", "numpy", "torch", "trimesh", "estimater")}
        with patch.dict(sys.modules, deps), patch.object(sys, "path", sys.path.copy()):
            detect_box = runpy.run_path(str(script))["detect_box"]
            deps["cv2"].cvtColor.return_value = sentinel.bgr
            detector = MagicMock()
            detector.predict.side_effect = [(sentinel.box, 0.9, None), None]
            self.assertIs(detect_box(detector, sentinel.rgb), sentinel.box)
            deps["cv2"].cvtColor.assert_called_once_with(sentinel.rgb, deps["cv2"].COLOR_RGB2BGR)
            detector.predict.assert_called_once_with(sentinel.bgr)
            self.assertIsNone(detect_box(detector, sentinel.rgb))
