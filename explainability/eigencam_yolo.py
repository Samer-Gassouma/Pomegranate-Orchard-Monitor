"""
EigenCAM for YOLOv8 detection explainability.
Custom lightweight implementation — hooks the last conv layer,
computes PCA on feature maps, no gradient required.
"""

import warnings

import cv2
import numpy as np
import torch


def _get_last_conv_layer(model):
    """Find the last Conv2d layer in YOLOv8 backbone."""
    inner = model.model if hasattr(model, "model") else model
    for module in reversed(list(inner.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None


class _FeatureHook:
    """Simple hook to capture layer output."""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        self.features = output.detach()


def _compute_eigencam(feature_map):
    """
    Compute EigenCAM from a (C, H, W) feature map.
    Returns (H, W) heatmap.
    """
    c, h, w = feature_map.shape
    # Flatten to (C, H*W)
    features = feature_map.reshape(c, -1)
    # Center
    features = features - features.mean(dim=1, keepdim=True)
    # Covariance
    cov = torch.matmul(features, features.t())
    # Eigendecomposition
    _, eigenvectors = torch.linalg.eigh(cov)
    # Principal component (largest eigenvalue = last column)
    pc = eigenvectors[:, -1]
    # Project features onto PC
    heatmap = torch.matmul(pc, features).reshape(h, w)
    # ReLU
    heatmap = torch.relu(heatmap)
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.cpu().numpy()


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

    try:
        target_layer = _get_last_conv_layer(yolo_model)
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in YOLO model")

        hook = _FeatureHook()
        handle = target_layer.register_forward_hook(hook)

        # Run forward pass
        img_resized = cv2.resize(image_rgb, (640, 640))
        tensor = (
            torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        device = next(yolo_model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            yolo_model.model(tensor) if hasattr(yolo_model, "model") else yolo_model(tensor)

        handle.remove()

        if hook.features is None:
            raise RuntimeError("Failed to capture feature maps")

        # features shape: (1, C, H, W)
        features = hook.features[0]  # (C, H, W)
        heatmap = _compute_eigencam(features)

        # Resize to original image size
        heatmap_raw = cv2.resize(heatmap, (w, h))

        # Create colored overlay (JET colormap)
        heatmap_color = cv2.applyColorMap((heatmap_raw * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0
        heatmap_overlay = (image_norm * 0.5 + heatmap_color.astype(np.float32) / 255.0 * 0.5)
        heatmap_overlay = np.clip(heatmap_overlay, 0, 1)

        return (heatmap_overlay * 255).astype(np.uint8), heatmap_raw

    except Exception as e:
        warnings.warn(f"EigenCAM failed: {e}. Returning empty heatmap.")
        heatmap_raw = np.zeros((h, w), dtype=np.float32)
        heatmap_overlay = image_rgb.copy()
        return heatmap_overlay, heatmap_raw
