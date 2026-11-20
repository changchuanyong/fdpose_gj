from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from fdonpose.application.validation import validate_pose


@unittest.skipIf(np is None, "NumPy is unavailable")
class PoseValidationTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        self.corners = np.array([[-0.1, -0.1, 1.0], [0.1, 0.1, 1.0]])
        self.pose = np.eye(4)
        self.depth = np.full((80, 100), 1.0)

    def test_validates_depth_only_after_projected_box_matches(self):
        render_depth = Mock(return_value=np.full_like(self.depth, 0.98))
        align_depth = Mock(return_value={"accepted": True, "depth_pixels": 200})

        result = validate_pose(self.corners, self.pose, np.array([40, 30, 60, 50]), self.depth,
                               self.K, render_depth, align_depth)

        self.assertTrue(result["accepted"])
        self.assertEqual(1.0, result["alignment_iou"])
        render_depth.assert_called_once_with(self.K, self.depth)
        align_depth.assert_called_once_with(render_depth.return_value, self.depth)

    def test_skips_depth_render_when_detection_does_not_overlap(self):
        render_depth = Mock()
        align_depth = Mock()

        result = validate_pose(self.corners, self.pose, np.array([0, 0, 10, 10]), self.depth,
                               self.K, render_depth, align_depth)

        self.assertFalse(result["accepted"])
        self.assertEqual(0.0, result["alignment_iou"])
        render_depth.assert_not_called()
        align_depth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
