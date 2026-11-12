from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from rgbd_dataset import load_frame_index, load_metadata, write_json
from orbbec_capture import camera_matrix, distortion_coefficients

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from reconstruct_rgbd import masked_depth


def metadata() -> dict:
    return {
        "schema_version": 1,
        "width": 640,
        "height": 480,
        "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        "depth_scale_m": 0.001,
        "rectified": True,
    }


class RgbdDatasetTest(unittest.TestCase):
    def test_load_metadata_and_ordered_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metadata.json", metadata())
            records = [
                {"index": 0, "rgb": "frames/000000_rgb.png", "depth": "frames/000000_depth.png"},
                {"index": 1, "rgb": "frames/000001_rgb.png", "depth": "frames/000001_depth.png"},
            ]
            (root / "frames.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
            )
            self.assertEqual(0.001, load_metadata(root)["depth_scale_m"])
            self.assertEqual(records, load_frame_index(root))

    def test_unrectified_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = metadata()
            value["rectified"] = False
            write_json(root / "metadata.json", value)
            with self.assertRaisesRegex(ValueError, "rectified"):
                load_metadata(root)

    def test_noncontiguous_frames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"index": 2, "rgb": "rgb.png", "depth": "depth.png"}
            (root / "frames.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_frame_index(root)

    def test_orbbec_intrinsic_and_distortion_conversion(self) -> None:
        intrinsic = SimpleNamespace(fx=610.2, fy=611.3, cx=319.4, cy=239.5)
        distortion = SimpleNamespace(k1=0.1, k2=-0.02, p1=0.003, p2=0.004,
                                     k3=0.001, k4=0.0, k5=0.0, k6=0.0)
        self.assertEqual([[610.2, 0.0, 319.4], [0.0, 611.3, 239.5], [0.0, 0.0, 1.0]],
                         camera_matrix(intrinsic))
        self.assertEqual([0.1, -0.02, 0.003, 0.004, 0.001, 0.0, 0.0, 0.0],
                         distortion_coefficients(distortion))


@unittest.skipIf(np is None, "NumPy is unavailable in the standard-library-only smoke environment")
class MaskedDepthTest(unittest.TestCase):
    def test_mask_and_depth_range_preserve_only_object(self) -> None:
        depth = np.full((20, 20), 500, dtype=np.uint16)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:-2, 2:-2] = 255
        selected = masked_depth(depth, mask, 0.2, 1.0, 0.001)
        self.assertEqual(256, np.count_nonzero(selected))
        self.assertEqual(0, selected[0, 0])
        self.assertEqual(500, selected[10, 10])

    def test_too_little_object_depth_is_rejected(self) -> None:
        depth = np.full((20, 20), 500, dtype=np.uint16)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "128"):
            masked_depth(depth, mask, 0.2, 1.0, 0.001)


if __name__ == "__main__":
    unittest.main()
