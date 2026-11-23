"""Launch the D415 RGB-D publisher and the headless pose node together."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fdpose_common import common_arguments, description, entry_path, pose_flags, process

from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    arguments = common_arguments(camera_namespace="/d415", frames="0", stream=True)
    root = LaunchConfiguration("project_root")
    python = LaunchConfiguration("python")
    namespace = LaunchConfiguration("camera_namespace")
    serial = LaunchConfiguration("serial")
    camera_flags = ["--serial", serial, "--namespace", namespace,
                    "--stream-width", LaunchConfiguration("stream_width"),
                    "--stream-height", LaunchConfiguration("stream_height")]
    node_flags = ["--camera", "d415"] + pose_flags(
        namespace, serial, LaunchConfiguration("mesh"), LaunchConfiguration("geometry"),
        LaunchConfiguration("weights"), LaunchConfiguration("output"),
        LaunchConfiguration("pose_topic"), LaunchConfiguration("frames"))
    return description(arguments, [
        process(python, entry_path(root, "fdpose_camera", "d415_node.py"), camera_flags, root),
        process(python, entry_path(root, "fdpose_perception", "node.py"), node_flags, root),
    ])