"""Geometry operations shared by live tracking and offline evaluation."""

import numpy as np


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
