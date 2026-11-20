import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/fdpose_perception"))
    from fdpose_perception.preview import GeminiPreview


@unittest.skipIf(np is None, "Requires OpenCV and NumPy")
class GeminiPreviewTest(unittest.TestCase):
    def test_present_queues_full_resolution_overlay_without_resizing(self):
        preview = GeminiPreview.__new__(GeminiPreview)
        preview.process = SimpleNamespace(exitcode=None)
        preview.stop = SimpleNamespace(is_set=lambda: False)
        preview.source = (143, 1795067194428.0)
        preview.updates = MagicMock()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[10, 20] = (1, 2, 3)

        self.assertTrue(preview.present(frame, "Tracking"))

        caption, displayed = preview.updates.put_nowait.call_args.args[0]
        self.assertIn("Source frame 143", caption)
        self.assertEqual(frame.shape, displayed.shape)
        np.testing.assert_array_equal(displayed[10, 20], (3, 2, 1))


if __name__ == "__main__":
    unittest.main()
