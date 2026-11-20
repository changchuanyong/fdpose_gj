"""Prepare aligned RGB, depth, mask and intrinsics for registration."""

from dataclasses import dataclass

import cv2
import numpy as np

from fdonpose.domain.geometry import box_in_view


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
