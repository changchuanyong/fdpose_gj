"""Desktop CLI: track a configured part with D415 or Gemini RGB-D via FoundationPose."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("FDPOSE_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / "src/fdpose_perception"))
from fdpose_perception.rgbd import open_d415
from fdpose_perception.adapters.foundationpose import draw_pose
from fdpose_perception.adapters.ros2 import PosePublisher
from fdpose_perception.frontend import (add_common_args, bootstrap, build_summary, make_validate,
                                        tracking_statistics, validate_inputs, write_summary)
from fdpose_perception.pipeline import PosePipeline, load_cad_from_target
from fdpose_perception.preview import make_preview
from fdpose_perception.recording import Recorder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, ROOT)
    parser.add_argument("--stream-width", type=int, default=640, help="D415 stream width")
    parser.add_argument("--stream-height", type=int, default=480, help="D415 stream height")
    parser.add_argument("--show", action="store_true",
                        help="Show live RGB and pose overlays on the Jetson desktop")
    parser.add_argument("--ros-pose-topic", help="Publish target poses as geometry_msgs/PoseStamped")
    args = parser.parse_args(argv)
    if args.class_id < 0:
        parser.error("--class-id must be nonnegative")
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if args.stream_width <= 0 or args.stream_height <= 0 or args.init_width <= 0:
        parser.error("--stream-width, --stream-height and --init-width must be positive")
    if args.est_refine_iter < 1:
        parser.error("--est-refine-iter must be positive")
    if args.ros_pose_topic is not None and not args.ros_pose_topic.startswith('/'):
        parser.error("--ros-pose-topic must be an absolute ROS topic")
    if args.show and not os.environ.get("DISPLAY"):
        parser.error("--show needs a desktop display. Run from a terminal on the Jetson desktop.")
    validate_inputs(parser, args)
    return args


def main():
    args = parse_args()
    cad_from_target = load_cad_from_target(args.geometry)
    model = bootstrap(args, ROOT)
    with ExitStack() as stack:
        if args.camera == "orbbec-ros":
            from fdpose_perception.orbbec_ros import open_orbbec_ros

            camera = stack.enter_context(open_orbbec_ros(args.serial, args.camera_namespace))
        else:
            camera = stack.enter_context(open_d415(args.serial, args.stream_width, args.stream_height))
        live_preview, preview = make_preview(camera.metadata["camera"], show=args.show,
                                             spawned=args.camera == "orbbec-ros")
        if preview:
            stack.callback(preview.close)
        K = camera.K
        frame_stamp_ns = None
        pose_publisher = None
        if args.ros_pose_topic:
            frame_id = camera.metadata.get("frame_id", "camera_color_optical_frame")
            pose_publisher = PosePublisher(args.ros_pose_topic, frame_id)
            stack.callback(pose_publisher.close)

        def read_frame():
            nonlocal frame_stamp_ns
            frame = camera.read_frame()
            if pose_publisher:
                frame_stamp_ns = camera.ros_stamp_ns() if camera.ros_stamp_ns else pose_publisher.now_ns()
            if live_preview:
                live_preview.source = frame[2:4]
            return frame
        print(json.dumps(camera.metadata), flush=True)

        publish = (lambda target_pose: pose_publisher.publish(target_pose, frame_stamp_ns)) if pose_publisher else None
        recorder = Recorder(args.output, K, cad_from_target, publish=publish)
        pipeline = PosePipeline(model.session, model.detector,
                                model.segmenter.segment_box if model.segmenter else None,
                                model.corners, K, make_validate(model.corners, model.estimator, K),
                                init_width=args.init_width, est_refine_iter=args.est_refine_iter,
                                mask_backend=args.mask_backend)
        ui = SimpleNamespace(
            read_frame=read_frame,
            present=lambda bgr, status: preview.present(bgr, status) if preview else True,
            emit=lambda obj: print(json.dumps(obj, flush=True)),
            log=lambda msg: print(msg, flush=True),
            draw=lambda bgr, cad_pose, target_pose: draw_pose(K, bgr, cad_pose, target_pose, model.mesh.bounds),
            interactive=bool(preview),
            closed=lambda: preview.closed if preview else False)
        pipeline.run(ui, recorder, args.frames)
        extra = {"requested_stream_fps": 30 if args.camera == "d415" else None}
        extra.update(tracking_statistics(recorder))
        if live_preview:
            extra.update(live_preview.measurements())
        write_summary(args.output, build_summary(args, camera.metadata, recorder, model.segmenter,
                                                 extra=extra))


if __name__ == "__main__":
    main()