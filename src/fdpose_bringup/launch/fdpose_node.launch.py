"""Launch the headless FoundationPose ROS node (Gemini RGB-D -> PoseStamped)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdpose_common import common_arguments, description, entry_path, pose_flags, process

from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    arguments = common_arguments(frames="0")
    root = LaunchConfiguration("project_root")
    entry = entry_path(root, "fdpose_perception", "node.py")
    flags = pose_flags(LaunchConfiguration("camera_namespace"), LaunchConfiguration("serial"),
                       LaunchConfiguration("mesh"), LaunchConfiguration("geometry"),
                       LaunchConfiguration("weights"), LaunchConfiguration("output"),
                       LaunchConfiguration("pose_topic"), LaunchConfiguration("frames"))
    return description(arguments, [process(LaunchConfiguration("python"), entry, flags, root)])