"""Launch the existing FoundationPose application with optional ROS pose output."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    default_root = os.environ.get("FDPOSE_ROOT", str(Path.home() / "fdonpose-git"))
    arguments = [
        DeclareLaunchArgument("project_root", default_value=default_root),
        DeclareLaunchArgument("python", default_value=str(Path.home() / "venvs/ev-charging-d415-gpu/bin/python")),
        DeclareLaunchArgument("camera", default_value="orbbec-ros"),
        DeclareLaunchArgument("camera_namespace", default_value="/gemini"),
        DeclareLaunchArgument("serial"),
        DeclareLaunchArgument("mesh", default_value="assets/mesh/socket_mod.PLY"),
        DeclareLaunchArgument("geometry", default_value="config/geometry/socket_openings_ply.json"),
        DeclareLaunchArgument("weights", default_value="weights/best.pt"),
        DeclareLaunchArgument("output"),
        DeclareLaunchArgument("pose_topic", default_value="/fdpose/pose"),
    ]
    root = LaunchConfiguration("project_root")
    entry = Path(get_package_share_directory("fdpose_perception")) / "apps/jetson_foundationpose_d415.py"
    return LaunchDescription(arguments + [
        ExecuteProcess(
            cmd=[
                LaunchConfiguration("python"), "-u", str(entry),
                "--camera", LaunchConfiguration("camera"),
                "--camera-namespace", LaunchConfiguration("camera_namespace"),
                "--serial", LaunchConfiguration("serial"),
                "--mesh", LaunchConfiguration("mesh"),
                "--geometry", LaunchConfiguration("geometry"),
                "--weights", LaunchConfiguration("weights"),
                "--output", LaunchConfiguration("output"),
                "--ros-pose-topic", LaunchConfiguration("pose_topic"),
            ],
            cwd=root,
            additional_env={"FDPOSE_ROOT": root},
            output="screen",
        )
    ])
