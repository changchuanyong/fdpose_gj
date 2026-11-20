"""Select one target instance for live pose registration."""

from pathlib import Path
import sys

import numpy as np


def select_target(result, image_shape, class_id, confidence, *, require_mask):
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return None
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    candidates = np.flatnonzero((classes == class_id) & (scores >= confidence))
    if not len(candidates):
        return None
    index = int(candidates[np.argmax(scores[candidates])])
    mask = None
    if require_mask:
        if result.masks is None or len(result.masks.data) != len(boxes):
            raise RuntimeError("YOLO segmentation boxes and masks must match one-to-one")
        mask = result.masks.data[index].cpu().numpy() > 0.5
        if mask.shape != tuple(image_shape[:2]):
            raise RuntimeError(f"Mask {mask.shape} does not match RGB pixels {image_shape[:2]}")
        if not mask.any():
            return None
    return boxes.xyxy[index].cpu().numpy(), float(scores[index]), mask


class TargetDetector:
    def __init__(self, weights: Path, backend: str, class_id: int):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.require_mask = backend == "yolo-seg"
        expected = "segment" if self.require_mask else "detect"
        if self.model.task != expected:
            raise ValueError(f"{backend} needs {expected} weights, got {self.model.task!r}: {weights}")
        if class_id not in self.model.names:
            raise ValueError(f"Class ID {class_id} is not in the model: {self.model.names}")
        self.class_id = class_id

    def predict(self, bgr, confidence=0.50):
        options = {"imgsz": 640, "retina_masks": True} if self.require_mask else {}
        result = self.model.predict(bgr, classes=[self.class_id], conf=confidence,
                                    device=0, verbose=False, **options)[0]
        return select_target(result, bgr.shape, self.class_id, confidence,
                             require_mask=self.require_mask)


class Sam2Segmenter:
    """SAM2.1 box-prompted mask refinement; builds the predictor lazily on CUDA."""

    def __init__(self, checkpoint, sam2_root):
        sys.path.insert(0, str(sam2_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor(build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint),
            device="cuda", apply_postprocessing=False,
        ))

    def segment_box(self, image, box):
        import torch

        with torch.inference_mode():
            self.predictor.set_image(image)
            try:
                masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
            finally:
                self.predictor.reset_predictor()
        return masks[0].astype(bool), float(scores[0])
