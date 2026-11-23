"""Undistortion map construction shared by the D415 and Orbbec camera adapters."""

import numpy as np


def rectify_maps(K, coefficients, size):
    """Return cv2 remap maps, or None when the distortion is negligible.

    Distortion-model acceptance stays with the caller: each SDK reports models
    differently and rejecting an unsupported model is a device-boundary check.
    """
    import cv2

    coefficients = np.asarray(coefficients, dtype=float)
    if not np.any(np.abs(coefficients) > 0.0):
        return None
    matrix = np.asarray(K, dtype=float)
    return cv2.initUndistortRectifyMap(matrix, coefficients, None, matrix,
                                       (int(size[0]), int(size[1])), cv2.CV_32FC1)