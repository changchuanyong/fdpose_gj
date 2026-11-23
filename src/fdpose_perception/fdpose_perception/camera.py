"""Live RGB-D camera contract, rectification maps and RealSense D415 adapter."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RGBDStream:
    """K in RGB pixels; read_frame returns RGB8, aligned depth in meters, frame counter, timestamp ms."""

    K: np.ndarray
    metadata: dict
    read_frame: Callable[[], tuple[np.ndarray, np.ndarray, int, float]]
    ros_stamp_ns: Callable[[], int] | None = None


@contextmanager
def open_d415(serial, width, height, fps=30):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        actual_serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        if actual_serial != serial or name.split()[-1] != "D415":
            raise RuntimeError(f"Unexpected camera {name} {actual_serial}")
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        scale = float(device.first_depth_sensor().get_depth_scale())
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"Invalid RealSense depth scale {scale}")
        coefficients = np.array(intr.coeffs, dtype=float)
        rectify = None
        if intr.model != rs.distortion.none:
            if np.any(coefficients != 0) and intr.model != rs.distortion.brown_conrady:
                raise RuntimeError(f"Unsupported color distortion model {intr.model}")
            rectify = rectify_maps(K, coefficients, (intr.width, intr.height))
        align = rs.align(rs.stream.color)
        metadata = {"camera": name, "serial": actual_serial, "color_fps": color_profile.fps(),
                    "rgb_resolution": [intr.width, intr.height],
                    "depth_resolution": [depth_profile.width(), depth_profile.height()],
                    "K": K.tolist(), "distortion": str(intr.model),
                    "dist_coeffs": coefficients.tolist(), "depth_scale_m": scale}

        def read_frame():
            frames = align.process(pipeline.wait_for_frames(4000))
            color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth_frame:
                raise RuntimeError("No aligned RGB-D pair")
            rgb = np.asanyarray(color.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
            if rgb.shape != (intr.height, intr.width, 3) or depth.shape != rgb.shape[:2]:
                raise RuntimeError("Aligned frames must match the RGB intrinsics and resolution")
            if rectify is not None:
                rgb = cv2.remap(rgb, *rectify, cv2.INTER_LINEAR)
                depth = cv2.remap(depth, *rectify, cv2.INTER_NEAREST)
            return rgb, depth, color.get_frame_number(), color.get_timestamp()

        yield RGBDStream(K, metadata, read_frame)
    finally:
        pipeline.stop()


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
