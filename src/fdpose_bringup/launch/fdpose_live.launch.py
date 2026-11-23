"""Launch the existing FoundationPose application with optional ROS pose output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdpose_common import common_arguments, description, process

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    arguments = common_arguments(frames="60")
    arguments.append(DeclareLaunchArgument("camera", default_value="orbbec-ros"))
    root = LaunchConfiguration("project_root")
    entry = PathJoinSubstitution([root, "apps", "live_pose.py"])
    flags = ["--camera", LaunchConfiguration("camera"),
             "--camera-namespace", LaunchConfiguration("camera_namespace"),
             "--serial", LaunchConfiguration("serial"),
             "--mesh", LaunchConfiguration("mesh"),
             "--geometry", LaunchConfiguration("geometry"),
             "--weights", LaunchConfiguration("weights"),
             "--output", LaunchConfiguration("output"),
             "--ros-pose-topic", LaunchConfiguration("pose_topic"),
             "--frames", LaunchConfiguration("frames")]
    return description(arguments, [process(LaunchConfiguration("python"), entry, flags, root)])