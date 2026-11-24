"""Launch the FoundationPose live pose app: D415 or Gemini, desktop or headless."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


DEFAULT_ROOT = os.environ.get("FDPOSE_ROOT", str(Path.home() / "fdonpose-git"))
DEFAULT_PYTHON = os.environ.get(
    "FDPOSE_PYTHON", str(Path.home() / "miniforge3/envs/fdpose/bin/python"))


def generate_launch_description():
    root = LaunchConfiguration("project_root")
    entry = PathJoinSubstitution([root, "src", "fdpose_perception", "fdpose_perception", "live_pose.py"])
    flags = ["--camera", LaunchConfiguration("camera"),
             "--camera-namespace", LaunchConfiguration("camera_namespace"),
             "--serial", LaunchConfiguration("serial"),
             "--mesh", LaunchConfiguration("mesh"),
             "--geometry", LaunchConfiguration("geometry"),
             "--weights", LaunchConfiguration("weights"),
             "--output", LaunchConfiguration("output"),
             "--ros-pose-topic", LaunchConfiguration("pose_topic"),
             "--frames", LaunchConfiguration("frames")]
    return LaunchDescription([
        DeclareLaunchArgument("project_root", default_value=DEFAULT_ROOT),
        DeclareLaunchArgument("python", default_value=DEFAULT_PYTHON),
        DeclareLaunchArgument("camera_namespace", default_value="/gemini"),
        DeclareLaunchArgument("serial"),
        DeclareLaunchArgument("mesh", default_value="config/mesh/socket_mod.PLY"),
        DeclareLaunchArgument("geometry", default_value="config/geometry/socket_openings_ply.json"),
        DeclareLaunchArgument("weights", default_value="weights/best.pt"),
        DeclareLaunchArgument("output"),
        DeclareLaunchArgument("pose_topic", default_value="/fdpose/pose"),
        DeclareLaunchArgument("frames", default_value="60"),
        DeclareLaunchArgument("camera", default_value="orbbec-ros"),
        ExecuteProcess(cmd=[LaunchConfiguration("python"), "-u", entry] + flags,
                       cwd=root,
                       additional_env={"FDPOSE_ROOT": root, "PYTHONNOUSERSITE": "1"},
                       output="screen"),
    ])
