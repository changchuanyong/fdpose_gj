from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from fdonpose.adapters.ros2 import rotation_matrix_to_quaternion


@unittest.skipIf(np is None, "NumPy is unavailable")
class Ros2AdapterTest(unittest.TestCase):
    def test_identity_rotation(self):
        self.assertEqual([0.0, 0.0, 0.0, 1.0], rotation_matrix_to_quaternion(np.eye(3)))

    def test_quarter_turn_about_z(self):
        angle = np.pi / 2
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        quaternion = rotation_matrix_to_quaternion(matrix)
        np.testing.assert_allclose(quaternion, [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
