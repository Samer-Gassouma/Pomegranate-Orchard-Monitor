"""
EigenCAM for YOLOv8 detection explainability.
EigenCAM works cleanly on single-stage detectors (YOLO) by
using the last convolutional layer's feature maps.
"""

import warnings

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _get_last_conv_layer(model):
    """Find the last Conv2d layer in YOLOv8 backbone."""
    inner = model.model if hasattr(model, "model") else model
    target_layer = None
    for module in reversed(list(inner.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    return target_layer


class _YOLOv8ForwardWrapper(torch.nn.Module):
    """Minimal wrapper so CAM library sees a normal nn.Module forward."""

    def __init__(self, yolo):
        super().__init__()
        self.inner = yolo.model if hasattr(yolo, "model") else yolo

    def forward(self, x):
        return self.inner(x)


def explain_yolo_detection(yolo_model, image_bgr, bbox_xyxy):
    """
    Generate EigenCAM heatmap for a specific YOLO detection.

    Args:
        yolo_model: Ultralytics YOLO model
        image_bgr: Original image in BGR (numpy array, HxWx3, 0-255)
        bbox_xyxy: [x1, y1, x2, y2] bounding box to explain

    Returns:
        heatmap_overlay: RGB image with heatmap overlay
        heatmap_raw: Raw heatmap (HxW, 0-1)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Fallback: return empty heatmap on any error so API doesn't crash
    try:
        target_layer = _get_last_conv_layer(yolo_model)
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in YOLO model")

        wrapper = _YOLOv8ForwardWrapper(yolo_model)
        cam = EigenCAM(model=wrapper, target_layers=[target_layer])

        # Resize to model input size
        img_resized = cv2.resize(image_rgb, (640, 640))
        tensor = (
            torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        if next(wrapper.parameters()).is_cuda:
            tensor = tensor.cuda()

        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]  # HxW

        # Resize back to original image size
        heatmap_raw = cv2.resize(grayscale_cam, (w, h))

        # Normalize to 0-1
        if heatmap_raw.max() > 0:
            heatmap_raw = heatmap_raw / heatmap_raw.max()

        # Create colored overlay
        heatmap_overlay = show_cam_on_image(
            image_rgb.astype(np.float32) / 255.0,
            heatmap_raw,
            use_rgb=True,
        )

        return heatmap_overlay, heatmap_raw

    except Exception as e:
        warnings.warn(f"EigenCAM failed: {e}. Returning empty heatmap.")
        heatmap_raw = np.zeros((h, w), dtype=np.float32)
        heatmap_overlay = image_rgb.copy()
        return heatmap_overlay, heatmap_raw
