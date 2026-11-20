"""Pose validation shared by live tracking and recorded-data replay."""

from fdonpose.domain.geometry import box_iou, projected_box


def validate_pose(corners, pose, detection_box, measured_depth, K, render_depth,
                  align_depth, minimum_iou=0.35):
    overlap = box_iou(projected_box(K, corners, pose), detection_box)
    result = {"accepted": False, "alignment_iou": overlap}
    if detection_box is None or overlap < minimum_iou:
        return result
    rendered_depth = render_depth(K, measured_depth)
    result.update(align_depth(rendered_depth, measured_depth))
    return result
