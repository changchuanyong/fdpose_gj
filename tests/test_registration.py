import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fdpose_ws/src/fdpose_perception'))
try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    np = None
if np is not None:
    from fdpose_perception.pipeline import RejectedMask, prepare_registration


@unittest.skipIf(np is None, 'Requires OpenCV and NumPy')
class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 80, 3), np.uint8)
        self.depth = np.ones((64, 80), np.float32)
        self.K = np.array([[80., 0., 40.], [0., 80., 32.], [0., 0., 1.]])
        self.box = np.array([4, 4, 44, 44])
        self.mask = np.zeros(self.depth.shape, bool)
        self.mask[4:44, 4:44] = True

    def prepare(self, mask, segment):
        return prepare_registration(self.rgb, self.depth, self.K, 40, self.box, mask, segment)

    def test_instance_mask_and_box_prompt_use_same_pixel_coordinates(self):
        segment = Mock(return_value=(self.mask[::2, ::2], 0.9))
        direct = self.prepare(self.mask, segment)
        segment.assert_not_called()
        sam = self.prepare(None, segment)
        np.testing.assert_array_equal(segment.call_args.args[1], self.box / 2)
        np.testing.assert_array_equal(direct.mask, sam.mask)
        np.testing.assert_array_equal(sam.box, self.box / 2)
        np.testing.assert_array_equal(sam.K[:2], self.K[:2] / 2)
        self.assertEqual((32, 40), sam.depth.shape)
        self.assertEqual(0.9, sam.score)

    def test_rejects_clipped_mask_and_missing_depth(self):
        self.mask[:] = True
        with self.assertRaisesRegex(RejectedMask, 'clipped'):
            self.prepare(self.mask, None)
        self.depth[:] = 0
        with self.assertRaisesRegex(RejectedMask, 'depth'):
            self.prepare(self.mask, None)
