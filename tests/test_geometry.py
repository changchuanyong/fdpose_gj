from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/fdpose_perception"))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from fdpose_perception.geometry import box_in_view, box_iou, projected_box


@unittest.skipIf(np is None, "NumPy is unavailable")
class GeometryTest(unittest.TestCase):
    def test_projected_box_uses_camera_intrinsics(self):
        K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        corners = np.array([[-0.1, -0.1, 1.0], [0.1, 0.1, 1.0]])
        self.assertEqual([40.0, 30.0, 60.0, 50.0], projected_box(K, corners, np.eye(4)).tolist())

    def test_projected_box_rejects_points_behind_camera(self):
        self.assertIsNone(projected_box(np.eye(3), [[0.0, 0.0, -1.0]], np.eye(4)))

    def test_iou_and_view_bounds(self):
        self.assertAlmostEqual(1 / 7, box_iou(np.array([0, 0, 2, 2]), np.array([1, 1, 3, 3])))
        self.assertEqual(0.0, box_iou(None, np.array([0, 0, 2, 2])))
        self.assertTrue(box_in_view([8, 8, 92, 92], (100, 100, 3)))
        self.assertFalse(box_in_view([0, 8, 92, 92], (100, 100, 3)))


if __name__ == "__main__":
    unittest.main()
