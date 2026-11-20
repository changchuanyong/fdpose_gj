from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdonpose.application.tracking import TrackingSession


class TrackingSessionTest(unittest.TestCase):
    def setUp(self):
        self.estimator = Mock()
        self.synchronize = Mock()
        self.session = TrackingSession(self.estimator, self.synchronize)

    def test_registration_retry_and_tracking_loss_windows(self):
        self.assertEqual("register", self.session.seek_action(0.0))
        self.session.begin_registration(10.0)
        self.assertEqual("wait", self.session.seek_action(14.9))
        self.assertEqual("register", self.session.seek_action(15.0))

        self.session.confirm("rgb", "depth", 3, 2.5, SimpleNamespace(copy=lambda: "pose"))
        self.assertEqual(("rgb", "depth", 3, 2.5, "pose"), self.session.confirmed_frame)
        self.session.begin_registration(20.0)
        self.assertEqual("recover", self.session.seek_action(21.0))

        self.session.tracking_lost(22.0)
        self.assertIsNone(self.estimator.pose_last)
        self.assertEqual("recover", self.session.seek_action(24.9))
        self.assertEqual("register", self.session.seek_action(25.0))

    def test_recovery_restores_confirmed_state_and_uses_more_iterations(self):
        confirmed_state = Mock()
        confirmed_state.clone.return_value = "restored"
        self.session._confirmed_state = confirmed_state
        self.estimator.track_one.return_value = "pose"

        pose = self.session.track("rgb", "depth", "K", recover=True)

        self.assertEqual("pose", pose)
        self.assertEqual("restored", self.estimator.pose_last)
        self.estimator.track_one.assert_called_once_with(
            rgb="rgb", depth="depth", K="K", iteration=4
        )
        self.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
