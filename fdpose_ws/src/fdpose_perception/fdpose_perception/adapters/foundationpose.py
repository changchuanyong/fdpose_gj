"""Render the depth of FoundationPose's current centered-mesh state."""


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
