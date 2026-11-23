"""Subscribe to one Gemini 335L/336L running the Orbbec ROS 2 driver."""

from contextlib import ExitStack, contextmanager
import time

import cv2
import numpy as np

from fdpose_perception.calibration import rectify_maps
from fdpose_perception.rgbd import RGBDStream


# The Orbbec ROS driver multiplies raw depth by the device ValueScale before
# publishing 16UC1: the topic already carries millimeters, not SDK raw counts.
DEPTH_SCALE_M = 0.001
MAX_PAIR_DELTA_NS = 15_000_000


def stamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def image_array(message, dtype, channels):
    dtype = np.dtype(dtype)
    row_bytes = message.width * channels * dtype.itemsize
    if message.step < row_bytes or len(message.data) != message.step * message.height:
        raise RuntimeError("Invalid Orbbec ROS image stride or payload size")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    shape = (message.height, message.width, channels) if channels > 1 else (message.height, message.width)
    return np.ascontiguousarray(rows[:, :row_bytes]).view(dtype).reshape(shape)


def aligned_calibration(color, depth, color_info, depth_info):
    size = (color.width, color.height)
    frame = color.header.frame_id
    if (not all(value > 0 for value in size) or not frame.endswith('_color_optical_frame')
            or any((msg.width, msg.height) != size or msg.header.frame_id != frame
                   for msg in (depth, color_info, depth_info))):
        raise RuntimeError("Expected depth and CameraInfo registered to the color optical frame")
    K = np.asarray(color_info.k, dtype=float).reshape(3, 3)
    if (not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K.ravel(), depth_info.k)):
        raise RuntimeError("Invalid or mismatched color and aligned depth intrinsics")
    distortion = np.asarray(color_info.d, dtype=float)
    if (not np.isfinite(distortion).all() or (np.any(distortion) and (
            color_info.distortion_model not in ('plumb_bob', 'rational_polynomial')
            or len(distortion) not in (4, 5, 8)))):
        raise RuntimeError(f"Unsupported Orbbec color distortion: {color_info.distortion_model}")
    return K, distortion


def decode_pair(color, depth, rectify=None):
    if color.encoding not in ('rgb8', 'bgr8') or depth.encoding != '16UC1':
        raise RuntimeError(f"Expected RGB8/BGR8 and millimeter 16UC1, got {color.encoding}/{depth.encoding}")
    rgb = image_array(color, np.uint8, 3)
    if color.encoding == 'bgr8':
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    z = image_array(depth, '>u2' if depth.is_bigendian else '<u2', 1).astype(np.float32)
    z *= DEPTH_SCALE_M
    if rectify is not None:
        rgb = cv2.remap(rgb, *rectify, cv2.INTER_LINEAR)
        z = cv2.remap(z, *rectify, cv2.INTER_NEAREST)
    return rgb, z


class PairBuffer:
    """Keep the latest color/depth messages, never reuse frames or pair stale timestamps."""

    def __init__(self):
        self.images = {}
        self.infos = {}
        self.last_stamps = (0, 0)

    def pop(self):
        if len(self.images) != 2 or len(self.infos) != 2:
            return None
        color, depth = self.images['color'], self.images['depth']
        stamps = stamp_ns(color), stamp_ns(depth)
        if abs(stamps[0] - stamps[1]) > MAX_PAIR_DELTA_NS:
            del self.images['color' if stamps[0] < stamps[1] else 'depth']
            return None
        self.images.clear()
        if any(current <= last for current, last in zip(stamps, self.last_stamps)):
            return None
        self.last_stamps = stamps
        return color, depth, self.infos['color'], self.infos['depth']


class RosRgbdSource:
    """Generic ROS RGB-D source: pair color/depth topics registered to the color optical frame."""

    def __init__(self, node, executor, serial, namespace, camera_name, firmware=None):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        namespace = '/' + namespace.strip('/')
        if namespace == '/':
            raise ValueError('An explicit camera namespace is required')
        self._executor = executor
        self._buffer = PairBuffer()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for kind in ('color', 'depth'):
            path = f'{namespace}/{kind}'
            node.create_subscription(Image, path + '/image_raw',
                                     lambda msg, key=kind: self._buffer.images.__setitem__(key, msg), qos)
            node.create_subscription(CameraInfo, path + '/camera_info',
                                     lambda msg, key=kind: self._buffer.infos.__setitem__(key, msg), qos)

        first = self._next_pair()
        K, distortion = aligned_calibration(*first)
        w, h = first[0].width, first[0].height
        frame_id = first[0].header.frame_id
        self._rectify = rectify_maps(K, distortion, (w, h))
        self._w, self._h, self._frame_id = w, h, frame_id
        self._K, self._distortion = K, distortion
        self._first = first
        self._frame_number = 0
        self.K = K
        self.metadata = {"camera": camera_name, "serial": serial,
                         "firmware": firmware, "rgb_resolution": [w, h],
                         "depth_resolution": [w, h], "K": K.tolist(),
                         "dist_coeffs": distortion.tolist(), "depth_scale_m": DEPTH_SCALE_M,
                         "timestamp_source": "ros_color_header", "frame_id": frame_id,
                         "camera_namespace": namespace}
        self.ros_stamp_ns = lambda: self._buffer.last_stamps[0]

    def _next_pair(self):
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            pair = self._buffer.pop()
            if pair is not None:
                return pair
        raise RuntimeError('No synchronized RGB-D pair with CameraInfo within 4 s')

    def read_frame(self):
        pair = self._first if self._first is not None else self._next_pair()
        self._first = None
        color, depth, _, _ = pair
        observed_K, observed_d = aligned_calibration(*pair)
        if ((color.width, color.height, color.header.frame_id) != (self._w, self._h, self._frame_id)
                or not np.array_equal(observed_K, self._K) or not np.array_equal(observed_d, self._distortion)):
            raise RuntimeError('Orbbec camera calibration or stream format changed; restart tracking')
        rgb, meters = decode_pair(color, depth, self._rectify)
        self._frame_number += 1
        return rgb, meters, self._frame_number, stamp_ns(color) / 1_000_000


class OrbbecRosSource(RosRgbdSource):
    """Gemini source that verifies the Orbbec device serial/model before subscribing."""

    def __init__(self, node, executor, serial, namespace='/gemini'):
        from orbbec_camera_msgs.srv import GetDeviceInfo

        namespace = '/' + namespace.strip('/')
        client = node.create_client(GetDeviceInfo, namespace + '/get_device_info')
        if not client.wait_for_service(timeout_sec=8.0):
            raise RuntimeError(f'No Orbbec device info service under {namespace}; check the ROS domain and namespace')
        future = client.call_async(GetDeviceInfo.Request())
        executor.spin_until_future_complete(future, timeout_sec=4.0)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise RuntimeError(f'Orbbec device info request failed under {namespace}')
        device = response.info
        if (device.serial_number != serial or 'gemini' not in device.name.lower()
                or device.name.strip().upper().split()[-1] not in ('335L', '336L')):
            raise RuntimeError(f'Unexpected Orbbec camera {device.name} {device.serial_number}, expected {serial}')
        super().__init__(node, executor, device.serial_number, namespace, device.name,
                         device.firmware_version)


@contextmanager
def open_orbbec_ros(serial, namespace='/gemini'):
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
    except ImportError as exc:
        raise RuntimeError('Source ROS 2 and the Orbbec driver workspace before running the RGB-D client') from exc

    context = Context()
    rclpy.init(context=context)
    with ExitStack() as stack:
        stack.callback(context.try_shutdown)
        node = rclpy.create_node('fdpose_rgbd', context=context, enable_rosout=False)
        stack.callback(node.destroy_node)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        stack.callback(executor.shutdown)
        source = OrbbecRosSource(node, executor, serial, namespace)
        yield RGBDStream(source.K, source.metadata, source.read_frame, source.ros_stamp_ns)