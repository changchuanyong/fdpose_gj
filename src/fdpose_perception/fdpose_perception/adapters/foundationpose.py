"""FoundationPose depth rendering, validation and pose overlays."""

from functools import partial


def render_depth(estimator, K, depth):
    import torch
    from estimater import nvdiffrast_render

    with torch.inference_mode():
        _, rendered, _ = nvdiffrast_render(
            K=K, H=depth.shape[0], W=depth.shape[1],
            ob_in_cams=estimator.pose_last.reshape(1, 4, 4), glctx=estimator.glctx,
            mesh_tensors=estimator.mesh_tensors, output_size=depth.shape,
        )
    return rendered[0].detach().cpu().numpy()


def make_validate(corners, estimator, default_K):
    from fdpose_perception.tracking import depth_alignment, validate_pose

    def validate(pose, detection_box, measured_depth, evaluation_K=None):
        evaluation_K = default_K if evaluation_K is None else evaluation_K
        return validate_pose(corners, pose, detection_box, measured_depth, evaluation_K,
                             partial(render_depth, estimator), depth_alignment)
    return validate


def draw_pose(K, bgr_frame, cad_pose, target_pose, mesh_bounds):
    """Overlay the 3D bounding box and target XYZ axes on a BGR frame."""
    import numpy as np
    from estimater import draw_posed_3d_box, draw_xyz_axis

    from fdpose_perception.tracking import projected_box

    boxed = draw_posed_3d_box(K, bgr_frame, cad_pose, mesh_bounds)
    scale = 0.04
    axes = np.vstack((np.zeros((1, 3)), np.eye(3) * scale))
    bounds = projected_box(K, axes, target_pose)
    if bounds is None:
        return boxed
    height, width = boxed.shape[:2]
    x0, y0 = np.maximum(np.floor(bounds[:2]).astype(int) - 4, 0)
    x1, y1 = np.minimum(np.ceil(bounds[2:]).astype(int) + 5, (width, height))
    if x1 > x0 and y1 > y0:
        local_K = K.copy()
        local_K[:2, 2] -= [x0, y0]
        crop = boxed[y0:y1, x0:x1]
        crop[:] = draw_xyz_axis(crop, target_pose, scale=scale, K=local_K)
    return boxed
