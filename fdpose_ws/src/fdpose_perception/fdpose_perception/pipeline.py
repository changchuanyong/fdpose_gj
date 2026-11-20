"""Registration preparation and live pose session state; no camera, UI or CUDA imports."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from fdpose_perception.geometry import box_in_view, projected_box
from fdpose_perception.validation import (MAX_REVIEW_ROTATION_DEG, MAX_REVIEW_TRANSLATION_M,
                                          pose_delta)


class RejectedMask(ValueError):
    """A candidate that cannot be registered on this frame."""


@dataclass
class RegistrationInput:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    mask: np.ndarray
    box: np.ndarray
    score: float | None


def prepare_registration(rgb, depth, K, max_width, box, mask, segment_box):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_width / w)
    size = round(w * scale), max(1, round(h * scale))
    sx, sy = size[0] / w, size[1] / h
    image = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    z = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    intrinsic = K.copy()
    intrinsic[0] *= sx
    intrinsic[1] *= sy
    score = None
    if mask is None:
        mask, score = segment_box(image, box * [sx, sy, sx, sy])
    else:
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 128 or np.count_nonzero(z[mask] > 0) < max(128, len(xs) // 10):
        raise RejectedMask("Insufficient aligned depth in target mask")
    mask_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    if not box_in_view(mask_box, image.shape, margin=1):
        raise RejectedMask("Target mask clipped | Move target fully into view")
    return RegistrationInput(image, z, intrinsic, mask, mask_box, score)


class TrackingSession:
    def __init__(self, estimator, synchronize):
        self.estimator = estimator
        self.synchronize = synchronize
        self.confirmed_frame = None
        self._confirmed_state = None
        self._register_at = 0.0

    def seek_action(self, now):
        if now >= self._register_at:
            return "register"
        return "recover" if self._confirmed_state is not None else "wait"

    def begin_registration(self, now):
        # Throttle even if segmentation/depth checks reject the candidate.
        self._register_at = now + 5.0

    def register(self, rgb, depth, K, mask, iterations):
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iterations)
        self.synchronize()
        return pose

    def track(self, rgb, depth, K, *, recover=False):
        if recover:
            self.estimator.pose_last = self._confirmed_state.clone()
        pose = self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=4 if recover else 2)
        self.synchronize()
        return pose

    def confirm(self, rgb, depth, number, timestamp, pose):
        self.confirmed_frame = rgb, depth, number, timestamp, pose.copy()
        self._confirmed_state = self.estimator.pose_last.clone()

    def tracking_lost(self, now):
        self.estimator.pose_last = None
        self._register_at = now + 3.0


class PosePipeline:
    """Shared register/track/recover policy. Frontends inject I/O via the ui object.

    ui attributes: read_frame(), present(bgr, status)->bool, emit(dict), log(str),
    draw(bgr, cad_pose, target_pose)->image, interactive (bool), closed()->bool.
    """

    def __init__(self, estimator, session, detector, segment_box, corners, K, validate,
                 *, init_width, est_refine_iter, mask_backend, check_period_s=0.6):
        self.estimator = estimator
        self.session = session
        self.detector = detector
        self.segment_box = segment_box
        self.corners = corners
        self.K = K
        self.validate = validate
        self.init_width = init_width
        self.est_refine_iter = est_refine_iter
        self.mask_backend = mask_backend
        self.check_period_s = check_period_s

    def seek(self, ui, output):
        deadline = float("inf") if ui.interactive else time.monotonic() + 25
        status = "Aim at the target"
        while time.monotonic() < deadline:
            started = time.perf_counter()
            rgb, depth, number, timestamp = ui.read_frame()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detection = self.detector.predict(bgr)
            if detection is None or not box_in_view(detection[0], bgr.shape):
                status = "Aim at the target" if detection is None else "Target clipped | Move it fully into view"
                if not ui.present(bgr, f"{status} | Q or Esc to quit"):
                    return None
                continue
            box, confidence, mask = detection
            action = self.session.seek_action(time.monotonic())
            if action == "recover":
                recovered = self.session.track(rgb, depth, self.K, recover=True)
                validation = self.validate(recovered, box, depth)
                elapsed = (time.perf_counter() - started) * 1000
                if validation["accepted"]:
                    ui.emit({"event": "tracking_recovered", "frame": number,
                             "validation": validation, "backend": "last_confirmed_pose"})
                    return (recovered, rgb, depth, bgr, bgr, number, timestamp,
                            elapsed, validation, "recover")
                status = "Recovering from last confirmed pose"
                if not ui.present(bgr, status):
                    return None
                continue
            if action == "wait":
                status = "Waiting before registration retry"
                if not ui.present(bgr, status):
                    return None
                continue
            self.session.begin_registration(time.monotonic())
            try:
                candidate = prepare_registration(rgb, depth, self.K, self.init_width, box, mask,
                                                 self.segment_box)
            except RejectedMask as exc:
                status = str(exc)
                if not ui.present(bgr, status):
                    return None
                continue
            mask_info = {"mask_backend": self.mask_backend}
            if candidate.score is not None:
                mask_info["sam2_score"] = candidate.score
            overlay = bgr.copy()
            overlay_mask = cv2.resize(candidate.mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[overlay_mask] = (overlay[overlay_mask] * 0.55
                                     + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
            x0, y0, x1, y1 = np.rint(box).astype(int)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.imwrite(str(output / "candidate_mask.jpg"), overlay)
            if not ui.present(overlay, "Target mask | Initializing FoundationPose"):
                return None
            started = time.perf_counter()
            initial = self.session.register(candidate.rgb, candidate.depth, candidate.K, candidate.mask,
                                            self.est_refine_iter)
            elapsed = (time.perf_counter() - started) * 1000
            validation = self.validate(initial, candidate.box, candidate.depth, evaluation_K=candidate.K)
            if self.estimator.pose_last is None or not validation["accepted"]:
                self.estimator.pose_last = None
                status = "FoundationPose pose failed projection/depth validation"
                ui.emit({"event": "registration_rejected", "validation": validation})
                continue
            ui.emit({"event": "registration_accepted", "frame": number,
                     "detector_confidence": confidence, **mask_info,
                     "validation": validation, "register_ms": elapsed})
            return (initial, rgb, depth, bgr, overlay, number, timestamp,
                    elapsed, validation, "register")
        cv2.imwrite(str(output / "camera.jpg"), bgr)
        ui.log(f"No aligned target pose before timeout: {status}")
        return None

    def run(self, ui, recorder, frames):
        for _ in range(12):
            ui.read_frame()
        image = last_overlay = None
        recorder.open()
        try:
            while not frames or recorder.tracked_poses < frames:
                registration = self.seek(ui, recorder.output)
                if registration is None:
                    break
                (pose, rgb, depth, bgr, mask_overlay, number, timestamp,
                 elapsed, validation, stage) = registration
                target_pose = recorder.write_pose(pose, number, timestamp, stage, elapsed,
                                                  validation=validation)
                if stage == "register":
                    recorder.registrations += 1
                else:
                    recorder.recoveries += 1
                self.session.confirm(rgb, depth, number, timestamp, pose)
                last_overlay = bgr, pose, target_pose
                image = ui.draw(bgr.copy(), pose, target_pose)
                if recorder.registrations == 1 and stage == "register":
                    recorder.init_ms = elapsed
                    cv2.imwrite(str(recorder.output / "initial_mask.jpg"), mask_overlay)
                    cv2.imwrite(str(recorder.output / "first_pose.jpg"), image)
                if not ui.present(image, "Tracking | Q or Esc to quit"):
                    break
                next_check_at = 0.0
                while not frames or recorder.tracked_poses < frames:
                    started = time.perf_counter()
                    rgb, depth, new_number, timestamp = ui.read_frame()
                    if new_number <= number:
                        raise RuntimeError("Camera frame number did not advance")
                    number = new_number
                    stage, validation = "track", None
                    pose = self.session.track(rgb, depth, self.K)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    projection = projected_box(self.K, self.corners, pose)
                    translation_change, rotation_change = pose_delta(self.session.confirmed_frame[4], pose)
                    change = {"translation_m": translation_change, "rotation_deg": rotation_change}
                    sudden_change = (translation_change > MAX_REVIEW_TRANSLATION_M
                                     or rotation_change > MAX_REVIEW_ROTATION_DEG)
                    if projection is None or sudden_change or started >= next_check_at:
                        detection = self.detector.predict(bgr)
                        detection_box = detection[0] if detection and box_in_view(detection[0], bgr.shape) else None
                        validation = self.validate(pose, detection_box, depth)
                        recorder.checks += 1
                        next_check_at = started + self.check_period_s
                        if not validation["accepted"]:
                            rejected_pose = pose.copy()
                            rejected_validation = validation
                            recovered = self.session.track(rgb, depth, self.K, recover=True)
                            recovery_validation = self.validate(recovered, detection_box, depth)
                            recorder.save_drift(self.session.confirmed_frame,
                                                (rgb, depth, number, timestamp, rejected_pose),
                                                rejected_pose, recovered, rejected_validation,
                                                recovery_validation, (translation_change, rotation_change),
                                                ui.emit)
                            if recovery_validation["accepted"]:
                                pose, validation, stage = recovered, recovery_validation, "recover"
                                translation_change, rotation_change = pose_delta(self.session.confirmed_frame[4], pose)
                                change = {"translation_m": translation_change,
                                          "rotation_deg": rotation_change}
                                recorder.recoveries += 1
                                ui.emit({"event": "tracking_recovered", "frame": number,
                                         "validation": validation, "backend": "last_confirmed_pose"})
                            else:
                                recorder.resets += 1
                                self.session.tracking_lost(time.monotonic())
                                ui.present(bgr, "Pose rejected | Recovering target")
                                recorder.duration += time.perf_counter() - started
                                ui.emit({"event": "tracking_lost", "frame": number,
                                         "rejected_validation": rejected_validation,
                                         "recovery_validation": recovery_validation,
                                         "fallback": "cached_pose_then_throttled_register"})
                                break
                    ms = (time.perf_counter() - started) * 1000
                    target_pose = recorder.write_pose(pose, number, timestamp, stage, ms,
                                                      validation=validation, change=change)
                    recorder.times.append(ms)
                    recorder.translations.append(target_pose[:3, 3])
                    recorder.tracked_poses += 1
                    if validation is not None and validation["accepted"]:
                        self.session.confirm(rgb, depth, number, timestamp, pose)
                    last_overlay = bgr, pose, target_pose
                    if ui.interactive:
                        image = ui.draw(bgr, pose, target_pose)
                        ui.present(image, "Tracking | Q or Esc to quit")
                    recorder.duration += time.perf_counter() - started
                    if ui.closed():
                        break
                if ui.closed():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            recorder.close()
        if not ui.interactive and last_overlay is not None:
            image = ui.draw(*last_overlay)
        if image is not None:
            cv2.imwrite(str(recorder.output / "last_pose.jpg"), image)