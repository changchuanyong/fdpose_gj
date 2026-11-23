"""Launch arguments shared by the three fdpose_bringup entry points."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


DEFAULT_ROOT = os.environ.get("FDPOSE_ROOT", str(Path.home() / "fdonpose-git"))
DEFAULT_PYTHON = str(Path.home() / "venvs/ev-charging-d415-gpu/bin/python")


def common_arguments(*, camera_namespace="/gemini", frames=None, stream=False):
    arguments = [
        DeclareLaunchArgument("project_root", default_value=DEFAULT_ROOT),
        DeclareLaunchArgument("python", default_value=DEFAULT_PYTHON),
        DeclareLaunchArgument("camera_namespace", default_value=camera_namespace),
        DeclareLaunchArgument("serial"),
        DeclareLaunchArgument("mesh", default_value="config/mesh/socket_mod.PLY"),
        DeclareLaunchArgument("geometry", default_value="config/geometry/socket_openings_ply.json"),
        DeclareLaunchArgument("weights", default_value="weights/best.pt"),
        DeclareLaunchArgument("output"),
        DeclareLaunchArgument("pose_topic", default_value="/fdpose/pose"),
    ]
    if frames is not None:
        arguments.append(DeclareLaunchArgument("frames", default_value=frames))
    if stream:
        arguments.extend([
            DeclareLaunchArgument("stream_width", default_value="640"),
            DeclareLaunchArgument("stream_height", default_value="480"),
        ])
    return arguments


def process(python, entry, flags, root):
    return ExecuteProcess(
        cmd=[python, "-u", entry] + flags,
        cwd=root, additional_env={"FDPOSE_ROOT": root}, output="screen")


def pose_flags(namespace, serial, mesh, geometry, weights, output, pose_topic, frames=None):
    flags = ["--camera-namespace", namespace, "--serial", serial, "--mesh", mesh,
             "--geometry", geometry, "--weights", weights, "--output", output,
             "--pose-topic", pose_topic]
    if frames is not None:
        flags += ["--frames", frames]
    return flags


def entry_path(root, package, module):
    return PathJoinSubstitution([root, "src", package, package, module])


def description(arguments, actions):
    return LaunchDescription(arguments + actions)