"""Launch the D415 RGB-D publisher and the headless pose node together."""

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
        DeclareLaunchArgument("serial"),
        DeclareLaunchArgument("camera_namespace", default_value="/d415"),
        DeclareLaunchArgument("mesh", default_value="assets/mesh/socket_mod.PLY"),
        DeclareLaunchArgument("geometry", default_value="config/geometry/socket_openings_ply.json"),
        DeclareLaunchArgument("weights", default_value="weights/best.pt"),
        DeclareLaunchArgument("output"),
        DeclareLaunchArgument("pose_topic", default_value="/fdpose/pose"),
        DeclareLaunchArgument("frames", default_value="0"),
        DeclareLaunchArgument("stream_width", default_value="640"),
        DeclareLaunchArgument("stream_height", default_value="480"),
    ]
    root = LaunchConfiguration("project_root")
    python = LaunchConfiguration("python")
    namespace = LaunchConfiguration("camera_namespace")
    camera_entry = PathJoinSubstitution([root, "src", "fdpose_camera",
                                         "fdpose_camera", "d415_node.py"])
    node_entry = PathJoinSubstitution([root, "src", "fdpose_perception",
                                       "fdpose_perception", "node.py"])
    common = {"cwd": root, "additional_env": {"FDPOSE_ROOT": root}, "output": "screen"}
    return LaunchDescription(arguments + [
        ExecuteProcess(
            cmd=[python, "-u", camera_entry,
                 "--serial", LaunchConfiguration("serial"),
                 "--namespace", namespace,
                 "--stream-width", LaunchConfiguration("stream_width"),
                 "--stream-height", LaunchConfiguration("stream_height")],
            **common),
        ExecuteProcess(
            cmd=[python, "-u", node_entry,
                 "--camera", "d415",
                 "--camera-namespace", namespace,
                 "--serial", LaunchConfiguration("serial"),
                 "--mesh", LaunchConfiguration("mesh"),
                 "--geometry", LaunchConfiguration("geometry"),
                 "--weights", LaunchConfiguration("weights"),
                 "--output", LaunchConfiguration("output"),
                 "--pose-topic", LaunchConfiguration("pose_topic"),
                 "--frames", LaunchConfiguration("frames")],
            **common),
    ])