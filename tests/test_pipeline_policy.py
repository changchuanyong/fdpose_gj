"""PosePipeline 决策梯与 Recorder 的离线策略测试（mock 估计器/检测器）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/fdpose_perception"))

try:
    import cv2  # noqa: F401
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from fdpose_perception.pipeline import PosePipeline
    from fdpose_perception.recording import Recorder


def _ui(read_frame):
    return SimpleNamespace(read_frame=read_frame, present=Mock(return_value=True),
                           emit=Mock(), log=Mock(), draw=Mock(return_value=np.zeros((2, 2, 3), np.uint8)),
                           interactive=False, closed=Mock(return_value=False))


def _pipeline(estimator, session, detector, validate):
    K = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    corners = np.array([[-0.1, -0.1, 0.4], [0.1, 0.1, 0.6]])
    return PosePipeline(estimator, session, detector, None, corners, K, validate,
                        init_width=640, est_refine_iter=3, mask_backend="sam2")


@unittest.skipIf(np is None, "Requires OpenCV and NumPy")
class SeekPolicyTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.ones((480, 640), np.float32)
        self.pose = np.eye(4)
        self.pose[2, 3] = 0.5
        self.estimator = Mock(pose_last=object())
        self.session = Mock()
        self.detector = Mock()
        self.detector.predict.return_value = (np.array([10, 10, 100, 100]), 0.9, None)
        self.validate = Mock(return_value={"accepted": True})
        self.pipeline = _pipeline(self.estimator, self.session, self.detector, self.validate)
        self.ui = _ui(Mock(return_value=(self.rgb, self.depth, 1, 100.0)))

    def test_register_accepted_returns_register_stage(self):
        self.session.seek_action.return_value = "register"
        self.session.register.return_value = self.pose
        candidate = SimpleNamespace(rgb=self.rgb, depth=self.depth, K=np.eye(3),
                                    mask=np.ones((480, 640), bool), box=np.array([10, 10, 100, 100]),
                                    score=0.9)
        with TemporaryDirectory() as directory, \
                patch("fdpose_perception.pipeline.prepare_registration", return_value=candidate):
            result = self.pipeline.seek(self.ui, Path(directory))
        self.assertEqual("register", result[-1])
        self.assertIs(self.pose, result[0])
        self.session.begin_registration.assert_called_once()
        self.assertTrue(any(c.args[0].get("event") == "registration_accepted"
                            for c in self.ui.emit.call_args_list))

    def test_recover_accepted_returns_recover_stage(self):
        self.session.seek_action.return_value = "recover"
        self.session.track.return_value = self.pose
        with TemporaryDirectory() as directory:
            result = self.pipeline.seek(self.ui, Path(directory))
        self.assertEqual("recover", result[-1])
        self.session.track.assert_called_once_with(self.rgb, self.depth, self.pipeline.K, recover=True)

    def test_registration_rejected_when_validation_fails(self):
        self.session.seek_action.return_value = "register"
        self.session.register.return_value = self.pose
        self.validate.return_value = {"accepted": False}
        candidate = SimpleNamespace(rgb=self.rgb, depth=self.depth, K=np.eye(3),
                                    mask=np.ones((480, 640), bool), box=np.array([10, 10, 100, 100]),
                                    score=None)
        self.detector.predict.side_effect = [(np.array([10, 10, 100, 100]), 0.9, None), None]
        self.ui.present.side_effect = [True, False]
        with TemporaryDirectory() as directory, \
                patch("fdpose_perception.pipeline.prepare_registration", return_value=candidate):
            result = self.pipeline.seek(self.ui, Path(directory))
        self.assertIsNone(result)
        self.assertIsNone(self.estimator.pose_last)
        self.assertTrue(any(c.args[0].get("event") == "registration_rejected"
                            for c in self.ui.emit.call_args_list))


@unittest.skipIf(np is None, "Requires OpenCV and NumPy")
class RunPolicyTest(unittest.TestCase):
    def test_tracks_requested_frames_without_drift(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.ones((480, 640), np.float32)
        pose = np.eye(4)
        pose[2, 3] = 0.5
        estimator = Mock(pose_last=object())
        session = Mock()
        session.seek_action.return_value = "register"
        session.register.return_value = pose
        session.track.return_value = pose
        session.confirmed_frame = (rgb, depth, 13, 100.0, pose)
        detector = Mock()
        detector.predict.return_value = (np.array([10, 10, 100, 100]), 0.9, None)
        validate = Mock(return_value={"accepted": True})
        pipeline = _pipeline(estimator, session, detector, validate)
        frames = [(rgb, depth, n, 100.0 + n) for n in range(1, 40)]
        ui = _ui(Mock(side_effect=frames))
        candidate = SimpleNamespace(rgb=rgb, depth=depth, K=np.eye(3),
                                    mask=np.ones((480, 640), bool), box=np.array([10, 10, 100, 100]),
                                    score=None)
        with TemporaryDirectory() as directory, \
                patch("fdpose_perception.pipeline.prepare_registration", return_value=candidate):
            recorder = Recorder(Path(directory), pipeline.K, np.eye(4))
            pipeline.run(ui, recorder, frames=2)
            self.assertEqual(2, recorder.tracked_poses)
            self.assertEqual(1, recorder.registrations)
            self.assertEqual(0, recorder.resets)
            self.assertTrue((Path(directory) / "poses.jsonl").is_file())
            self.assertTrue((Path(directory) / "last_pose.jpg").is_file())


@unittest.skipIf(np is None, "Requires OpenCV and NumPy")
class RecorderTest(unittest.TestCase):
    def test_write_pose_rejects_non_rigid_transform(self):
        with TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), np.eye(3), np.eye(4)).open()
            bad = np.eye(4)
            bad[0, 0] = 2.0
            with self.assertRaisesRegex(RuntimeError, "invalid rigid transform"):
                recorder.write_pose(bad, 1, 100.0, "track", 5.0)
            recorder.close()

    def test_write_pose_applies_cad_from_target_and_publishes(self):
        published = []
        cad_from_target = np.eye(4)
        cad_from_target[0, 3] = 0.1
        with TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory), np.eye(3), cad_from_target,
                                publish=published.append).open()
            pose = np.eye(4)
            pose[2, 3] = 0.5
            target = recorder.write_pose(pose, 1, 100.0, "register", 5.0)
            recorder.close()
        np.testing.assert_allclose(target[:3, 3], [0.1, 0.0, 0.5])
        self.assertEqual(1, len(published))


if __name__ == "__main__":
    unittest.main()