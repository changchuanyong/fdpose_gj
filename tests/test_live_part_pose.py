import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    _root = Path(__file__).resolve().parents[1]
    with patch.object(sys, "path", [str(_root / "apps"),
                                    str(_root / "fdpose_ws/src/fdpose_perception"),
                                    str(_root / "fdpose_ws/src/fdpose_camera"), *sys.path]):
        from jetson_foundationpose_d415 import load_cad_from_target, parse_args
        from fdpose_camera.orbbec_ros import PairBuffer, aligned_calibration, decode_pair, stamp_ns
        from fdpose_camera.rgbd import open_d415
        from fdpose_perception.adapters.detector import TargetDetector, select_target


class Array:
    """Minimal stand-in for an Ultralytics result tensor."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return Array(self.value[index])


class Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf.value)


@unittest.skipIf(np is None, "Requires OpenCV and NumPy")
class LivePartPoseTest(unittest.TestCase):
    def setUp(self):
        self.result = SimpleNamespace(
            boxes=Boxes(cls=Array([0, 2, 2]), conf=Array([0.98, 0.7, 0.9]),
                        xyxy=Array([[0, 0, 2, 1], [1, 0, 3, 1], [2, 1, 3, 2]])),
            masks=SimpleNamespace(data=[Array(np.zeros((2, 3))), Array(np.ones((2, 3))),
                                        Array([[0, 0, 0], [0, 0, 1]])]),
        )

    def test_selects_matching_instance_mask_in_rgb_coordinates(self):
        box, score, mask = select_target(self.result, (2, 3, 3), 2, 0.5, require_mask=True)
        np.testing.assert_array_equal(box, [2, 1, 3, 2])
        np.testing.assert_array_equal(mask, [[False, False, False], [False, False, True]])
        self.assertAlmostEqual(score, 0.9)
        self.assertIsNone(select_target(self.result, (2, 3, 3), 2, 0.95, require_mask=True))
        self.assertIsNone(select_target(self.result, (2, 3, 3), 1, 0.5, require_mask=True))

    def test_rejects_misaligned_or_missing_masks(self):
        with self.assertRaisesRegex(RuntimeError, "does not match RGB pixels"):
            select_target(self.result, (4, 3, 3), 2, 0.5, require_mask=True)
        self.result.masks.data.pop()
        with self.assertRaisesRegex(RuntimeError, "one-to-one"):
            select_target(self.result, (2, 3, 3), 2, 0.5, require_mask=True)

    def test_detector_checks_weights_task_and_passes_backend_options(self):
        model = MagicMock(task="segment", names={0: "socket", 2: "part"})
        model.predict.return_value = [self.result]
        bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: model)}):
            detector = TargetDetector(Path("part.pt"), "yolo-seg", 2)
            self.assertIsNotNone(detector.predict(bgr)[2])
            self.assertEqual([2], model.predict.call_args.kwargs["classes"])
            self.assertTrue(model.predict.call_args.kwargs["retina_masks"])
            with self.assertRaisesRegex(ValueError, "Class ID 1"):
                TargetDetector(Path("part.pt"), "yolo-seg", 1)
            with self.assertRaisesRegex(ValueError, "needs detect weights"):
                TargetDetector(Path("part.pt"), "sam2", 2)
            model.task = "detect"
            self.result.masks = None
            detector = TargetDetector(Path("part.pt"), "sam2", 2)
            self.assertIsNone(detector.predict(bgr)[2])
            self.assertNotIn("retina_masks", model.predict.call_args.kwargs)

    def test_only_sam2_needs_checkpoint_and_cli_preserves_defaults(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("estimater.py", "part.PLY", "part.json", "part.pt"):
                (root / name).touch()
            checkpoint = root / "sam2.pt"
            args = ["--foundationpose", str(root), "--mesh", str(root / "part.PLY"),
                    "--geometry", str(root / "part.json"), "--weights", str(root / "part.pt"),
                    "--serial", "test", "--output", str(root / "output"),
                    "--sam2-checkpoint", str(checkpoint)]
            seg = parse_args(args + ["--mask-backend", "yolo-seg", "--class-id", "2", "--mesh-units", "m"])
            self.assertEqual(("yolo-seg", 2, "m"), (seg.mask_backend, seg.class_id, seg.mesh_units))
            with patch("sys.stderr") as stderr, self.assertRaises(SystemExit):
                parse_args(args)
            self.assertIn("SAM2 checkpoint not found", str(stderr.write.call_args))
            checkpoint.touch()
            sam = parse_args(args)
            self.assertEqual(("sam2", 0, "mm"), (sam.mask_backend, sam.class_id, sam.mesh_units))

    def test_geometry_uses_its_own_translation_units(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            transform = np.eye(4)
            for units, translation in (("mm", 25), ("m", 0.025)):
                transform[0, 3] = translation
                path.write_text(json.dumps({"units": units, "cad_to_target": transform.tolist()}))
                self.assertAlmostEqual(-0.025, load_cad_from_target(path)[0, 3])
            transform[0, 0] = -1
            path.write_text(json.dumps({"cad_to_target": transform.tolist()}))
            with self.assertRaisesRegex(ValueError, "proper rotation"):
                load_cad_from_target(path)

    def test_d415_converts_aligned_depth_and_stops_after_stream_error(self):
        rs = MagicMock()
        pipeline = rs.pipeline.return_value
        profile = pipeline.start.return_value
        device = profile.get_device.return_value
        device.get_info.side_effect = ["test-serial", "Intel RealSense D415"]
        device.first_depth_sensor.return_value.get_depth_scale.return_value = 0.0005
        video = profile.get_stream.return_value.as_video_stream_profile.return_value
        video.get_intrinsics.return_value = SimpleNamespace(
            fx=6, fy=7, ppx=1, ppy=0.5, width=3, height=2,
            model=rs.distortion.none, coeffs=[0] * 5,
        )
        video.width.return_value = 3
        video.height.return_value = 2
        video.fps.return_value = 30
        aligned = rs.align.return_value.process.return_value
        color, depth = aligned.get_color_frame.return_value, aligned.get_depth_frame.return_value
        color.get_data.return_value = np.zeros((2, 3, 3), dtype=np.uint8)
        color.get_frame_number.return_value = 17
        color.get_timestamp.return_value = 120.5
        depth.get_data.return_value = np.full((2, 3), 1200, dtype=np.uint16)
        with patch.dict(sys.modules, {"pyrealsense2": rs}):
            with self.assertRaisesRegex(RuntimeError, "must match the RGB intrinsics"):
                with open_d415("test-serial", 3, 2) as stream:
                    rgb, meters, number, timestamp = stream.read_frame()
                    self.assertEqual((2, 3, 3), rgb.shape)
                    self.assertEqual(np.float32, meters.dtype)
                    np.testing.assert_allclose(meters, 0.6)
                    np.testing.assert_array_equal(stream.K, [[6, 0, 1], [0, 7, 0.5], [0, 0, 1]])
                    self.assertEqual((17, 120.5), (number, timestamp))
                    depth.get_data.return_value = np.zeros((1, 3), dtype=np.uint16)
                    stream.read_frame()
        rs.align.assert_called_once_with(rs.stream.color)
        pipeline.stop.assert_called_once_with()

    def test_orbbec_depth_decodes_millimeters_to_meters_and_checks_alignment(self):
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        z = np.array([[400, 0, 600], [700, 800, 900]], dtype="<u2")
        color_msg = SimpleNamespace(encoding="rgb8", width=3, height=2, step=9,
                                    data=rgb.tobytes(), is_bigendian=False)
        depth_msg = SimpleNamespace(encoding="16UC1", width=3, height=2, step=6,
                                    data=z.tobytes(), is_bigendian=False)
        decoded_rgb, decoded_depth = decode_pair(color_msg, depth_msg)
        np.testing.assert_array_equal(decoded_rgb, rgb)
        np.testing.assert_allclose(decoded_depth, z.astype(np.float32) / 1000)

        K = np.array([[4, 0, 1], [0, 5, 0.5], [0, 0, 1]], dtype=float)
        frame = "gemini_color_optical_frame"
        color = SimpleNamespace(width=3, height=2, header=SimpleNamespace(frame_id=frame))
        depth = SimpleNamespace(width=3, height=2, header=SimpleNamespace(frame_id=frame))
        info = SimpleNamespace(width=3, height=2, header=SimpleNamespace(frame_id=frame),
                               k=K.ravel().tolist(), d=[0] * 8,
                               distortion_model="rational_polynomial")
        result, _ = aligned_calibration(color, depth, info, info)
        np.testing.assert_array_equal(result, K)
        depth.header.frame_id = "gemini_depth_optical_frame"
        with self.assertRaisesRegex(RuntimeError, "registered to the color"):
            aligned_calibration(color, depth, info, info)

    def test_orbbec_pair_buffer_matches_nearby_monotonic_frames(self):
        def msg(milliseconds, extra_ns=0):
            nanoseconds = milliseconds * 1_000_000 + extra_ns
            return SimpleNamespace(header=SimpleNamespace(
                stamp=SimpleNamespace(sec=nanoseconds // 1_000_000_000,
                                      nanosec=nanoseconds % 1_000_000_000)))

        color_info = object()
        depth_info = object()
        buffer = PairBuffer()
        buffer.infos = {"color": color_info, "depth": depth_info}
        color, depth = msg(100), msg(100, 10_000)
        buffer.images = {"color": color, "depth": depth}
        pair = buffer.pop()
        self.assertEqual((color, depth, color_info, depth_info), pair)
        self.assertAlmostEqual(100.01, stamp_ns(pair[1]) / 1e6)

        buffer.images = {"color": msg(200), "depth": msg(216)}
        self.assertIsNone(buffer.pop())
        self.assertIn("depth", buffer.images)
        buffer.images = {"color": msg(100), "depth": msg(100)}
        self.assertIsNone(buffer.pop())
