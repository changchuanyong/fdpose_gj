"""Launch the headless FoundationPose ROS node (Gemini RGB-D -> PoseStamped)."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    default_root = os.environ.get("FDPOSE_ROOT", str(Path.home() / "fdonpose-git"))
    arguments = [
        DeclareLaunchArgument("project_root", default_value=default_root),
        DeclareLaunchArgument("python", default_value=str(Path.home() / "venvs/ev-charging-d415-gpu/bin/python")),
        DeclareLaunchArgument("camera_namespace", default_value="/gemini"),
        DeclareLaunchArgument("serial"),
        DeclareLaunchArgument("mesh", default_value="assets/mesh/socket_mod.PLY"),
        DeclareLaunchArgument("geometry", default_value="config/geometry/socket_openings_ply.json"),
        DeclareLaunchArgument("weights", default_value="weights/best.pt"),
        DeclareLaunchArgument("output"),
        DeclareLaunchArgument("pose_topic", default_value="/fdpose/pose"),
        DeclareLaunchArgument("frames", default_value="0"),
    ]
    root = LaunchConfiguration("project_root")
    entry = PathJoinSubstitution([root, "src", "fdpose_perception",
                                  "fdpose_perception", "node.py"])
    return LaunchDescription(arguments + [
        ExecuteProcess(
            cmd=[
                LaunchConfiguration("python"), "-u", entry,
                "--camera-namespace", LaunchConfiguration("camera_namespace"),
                "--serial", LaunchConfiguration("serial"),
                "--mesh", LaunchConfiguration("mesh"),
                "--geometry", LaunchConfiguration("geometry"),
                "--weights", LaunchConfiguration("weights"),
                "--output", LaunchConfiguration("output"),
                "--pose-topic", LaunchConfiguration("pose_topic"),
                "--frames", LaunchConfiguration("frames"),
            ],
            cwd=root,
            additional_env={"FDPOSE_ROOT": root},
            output="screen",
        )
    ])