"""Registration preparation and live pose session state; no camera, UI or CUDA imports."""

from dataclasses import dataclass

import cv2
import numpy as np

from fdpose_perception.geometry import box_in_view


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