"""Pose tracking, projection geometry and validation without camera, UI or CUDA imports."""

from dataclasses import dataclass
import json
import time

import cv2
import numpy as np


UNIT_TO_M = {"mm": 0.001, "m": 1.0}
MAX_REVIEW_ROTATION_DEG = 12.0
MAX_REVIEW_TRANSLATION_M = 0.05


def load_cad_from_target(path):
    geometry = json.loads(path.read_text(encoding="utf-8-sig"))
    units = geometry.get("units", "mm")
    if units not in UNIT_TO_M:
        raise ValueError(f"Unsupported geometry units {units!r}: {path}")
    target_from_cad = np.asarray(geometry["cad_to_target"], dtype=float)
    if (target_from_cad.shape != (4, 4) or not np.isfinite(target_from_cad).all()
            or not np.allclose(target_from_cad[3], [0, 0, 0, 1])):
        raise ValueError(f"cad_to_target must be a finite 4x4 homogeneous transform: {path}")
    rotation = target_from_cad[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError(f"cad_to_target must contain a proper rotation: {path}")
    target_from_cad[:3, 3] *= UNIT_TO_M[units]
    return np.linalg.inv(target_from_cad)


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

    def __init__(self, session, detector, segment_box, corners, K, validate,
                 *, init_width, est_refine_iter, mask_backend, check_period_s=0.6):
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
            if self.session.estimator.pose_last is None or not validation["accepted"]:
                self.session.estimator.pose_last = None
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


def box_in_view(box, shape, margin=8):
    x0, y0, x1, y1 = box
    height, width = shape[:2]
    return margin <= x0 < x1 <= width - margin and margin <= y0 < y1 <= height - margin


def projected_box(K, corners, pose):
    points = corners @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return None
    pixels = points @ K.T
    xy = pixels[:, :2] / pixels[:, 2:]
    return np.concatenate((xy.min(axis=0), xy.max(axis=0)))


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    overlap = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0)
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection
    return intersection / union if union > 0 else 0.0


def validate_pose(corners, pose, detection_box, measured_depth, K, render_depth,
                  align_depth, minimum_iou=0.35):
    overlap = box_iou(projected_box(K, corners, pose), detection_box)
    result = {"accepted": False, "alignment_iou": overlap}
    if detection_box is None or overlap < minimum_iou:
        return result
    rendered_depth = render_depth(K, measured_depth)
    result.update(align_depth(rendered_depth, measured_depth))
    return result


def pose_delta(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = np.linalg.inv(reference) @ candidate
    cosine = np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)
    translation = float(np.linalg.norm(delta[:3, 3]))
    return translation, float(np.degrees(np.arccos(cosine)))


def depth_alignment(rendered_depth, measured_depth, min_pixels=128, tolerance_m=0.02,
                    max_median_error_m=0.015, min_inlier_ratio=0.65):
    rendered_depth = np.asarray(rendered_depth)
    measured_depth = np.asarray(measured_depth)
    interior = cv2.erode((rendered_depth > 0.001).astype(np.uint8), np.ones((3, 3), np.uint8))
    valid = (interior > 0) & (measured_depth > 0.001)
    pixels = int(np.count_nonzero(valid))
    if pixels < min_pixels:
        return {"accepted": False, "depth_pixels": pixels,
                "depth_median_error_mm": None, "depth_inlier_ratio": None}
    errors = np.abs(rendered_depth[valid] - measured_depth[valid])
    median_error = float(np.median(errors))
    inlier_ratio = float(np.mean(errors <= tolerance_m))
    return {"accepted": bool(median_error <= max_median_error_m
                             and inlier_ratio >= min_inlier_ratio),
            "depth_pixels": pixels, "depth_median_error_mm": median_error * 1000,
            "depth_inlier_ratio": inlier_ratio, "depth_inlier_tolerance_mm": tolerance_m * 1000}
