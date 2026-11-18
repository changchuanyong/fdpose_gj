from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from realsense_capture import (camera_matrix, depth_scale_supported,
                               distortion_coefficients, profile_description)


class RealSenseCaptureTest(unittest.TestCase):
    def test_realsense_intrinsic_conversion(self) -> None:
        intrinsic = SimpleNamespace(fx=610.2, fy=611.3, ppx=319.4, ppy=239.5)
        self.assertEqual([[610.2, 0.0, 319.4], [0.0, 611.3, 239.5], [0.0, 0.0, 1.0]],
                         camera_matrix(intrinsic))

    def test_realsense_distortion_conversion(self) -> None:
        intrinsic = SimpleNamespace(coeffs=[0.1, -0.02, 0.003, 0.004, 0.001])
        self.assertEqual([0.1, -0.02, 0.003, 0.004, 0.001],
                         distortion_coefficients(intrinsic))

    def test_profile_description(self) -> None:
        profile = SimpleNamespace(width=lambda: 640, height=lambda: 480,
                                  fps=lambda: 30, format=lambda: "rgb8")
        self.assertEqual({"width": 640, "height": 480, "fps": 30, "format": "rgb8"},
                         profile_description(profile))

    def test_float32_millimetre_depth_scale_is_accepted(self) -> None:
        self.assertTrue(depth_scale_supported(0.0010000000474974513))
        self.assertTrue(depth_scale_supported(0.001))
        self.assertTrue(depth_scale_supported(0.0001))
        self.assertFalse(depth_scale_supported(0.0))
        self.assertFalse(depth_scale_supported(-0.001))
        self.assertFalse(depth_scale_supported(0.01))


if __name__ == "__main__":
    unittest.main()