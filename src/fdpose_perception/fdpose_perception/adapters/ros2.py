"""Optional ROS 2 output adapter for pose consumers."""

import numpy as np


def rotation_matrix_to_quaternion(rotation):
    """Return a geometry_msgs-compatible [x, y, z, w] quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * np.sqrt(max(1.0 + diagonal[index] - diagonal[next_index]
                                  - diagonal[last_index], 0.0))
        values = [0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        values[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        x, y, z = values
        w = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
    quaternion = np.array([x, y, z, w], dtype=float)
    return (quaternion / np.linalg.norm(quaternion)).tolist()


def fill_pose_stamped(message, pose, stamp_ns, frame_id):
    """Populate a PoseStamped from a 4x4 camera-from-target transform in metres."""
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    message.header.frame_id = frame_id
    message.pose.position.x = float(pose[0, 3])
    message.pose.position.y = float(pose[1, 3])
    message.pose.position.z = float(pose[2, 3])
    message.pose.orientation.x, message.pose.orientation.y, \
        message.pose.orientation.z, message.pose.orientation.w = \
        rotation_matrix_to_quaternion(pose[:3, :3])
    return message


class PosePublisher:
    """Publish target poses without making ROS a required import for offline tools."""

    def __init__(self, topic, frame_id, node_name="fdpose_pose"):
        import rclpy
        from geometry_msgs.msg import PoseStamped

        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(node_name, context=self._context, enable_rosout=False)
        self._publisher = self._node.create_publisher(PoseStamped, topic, 10)
        self._frame_id = frame_id
        self._pose_type = PoseStamped

    def now_ns(self):
        return self._node.get_clock().now().nanoseconds

    def publish(self, pose, stamp_ns):
        message = fill_pose_stamped(self._pose_type(), pose, stamp_ns, self._frame_id)
        self._publisher.publish(message)

    def close(self):
        self._node.destroy_node()
        self._context.try_shutdown()
