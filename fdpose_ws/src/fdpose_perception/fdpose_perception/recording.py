"""Pose stream, drift artifacts and tracking statistics; ROS-free, publish hook injected."""

from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np


class Recorder:
    def __init__(self, output, K, cad_from_target, publish=None):
        self.output = Path(output)
        self.K = K
        self.cad_from_target = cad_from_target
        self.publish = publish
        self.stream = None
        self.times = deque(maxlen=600)
        self.translations = deque(maxlen=600)
        self.tracked_poses = 0
        self.registrations = 0
        self.checks = 0
        self.recoveries = 0
        self.resets = 0
        self.duration = 0.0
        self.init_ms = None

    def open(self):
        self.stream = (self.output / "poses.jsonl").open("x", encoding="utf-8")
        return self

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def write_pose(self, pose, frame, device_ms, stage, elapsed_ms, validation=None, change=None):
        target_pose = np.asarray(pose, dtype=float) @ self.cad_from_target
        rotation = target_pose[:3, :3]
        if (not np.isfinite(target_pose).all() or target_pose[2, 3] <= 0
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or abs(np.linalg.det(rotation) - 1) > 1e-3):
            raise RuntimeError("FoundationPose returned an invalid rigid transform")
        record = {"frame": frame, "device_time_ms": device_ms, "stage": stage,
                  "elapsed_ms": elapsed_ms, "validation": validation,
                  "change_from_confirmed": change, "T_camera_target_m": target_pose.tolist()}
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        if self.publish is not None:
            self.publish(target_pose)
        return target_pose

    def save_drift(self, before, after, rejected_pose, recovery_pose, rejected_validation,
                   recovery_validation, change, emit):
        directory = self.output / "drift" / f"{after[2]:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        for name, frame in (("before", before), ("after", after)):
            rgb, depth, number, timestamp, pose = frame
            cv2.imwrite(str(directory / f"{name}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(directory / f"{name}_depth.npy", depth, allow_pickle=False)
            cv2.imwrite(str(directory / f"{name}_depth_mm.png"),
                        np.rint(depth * 1000).astype(np.uint16))
            np.savetxt(directory / f"{name}_pose.txt", pose)
        np.savetxt(directory / "cam_K.txt", self.K)
        np.savetxt(directory / "rejected_pose.txt", rejected_pose)
        np.savetxt(directory / "recovery_pose.txt", recovery_pose)
        metadata = {"before_frame": before[2], "after_frame": after[2],
                    "before_device_time_ms": before[3], "after_device_time_ms": after[3],
                    "change_from_confirmed": {"translation_m": change[0], "rotation_deg": change[1]},
                    "rejected_validation": rejected_validation,
                    "recovery_validation": recovery_validation}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                   encoding="utf-8")
        emit({"event": "drift_rgbd_saved", "path": str(directory)})