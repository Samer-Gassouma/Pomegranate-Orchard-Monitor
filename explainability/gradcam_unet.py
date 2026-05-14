"""
Grad-CAM for U-Net fruit instance segmentation explainability.
Explains which pixels in the fruit crop drove the fruit boundary decision.
"""

import warnings

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def _get_last_conv_layer(model):
    """Find the last Conv2d layer in the U-Net decoder."""
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return [module]
    return None


def explain_unet_segmentation(unet_model, fruit_crop_rgb, device="cuda"):
    """
    Generate Grad-CAM heatmap for U-Net fruit segmentation.

    Args:
        unet_model: Trained smp.Unet model
        fruit_crop_rgb: Fruit crop image (RGB, HxWx3, 0-255)
        device: 'cuda' or 'cpu'

    Returns:
        heatmap_overlay: RGB image with heatmap overlay
        heatmap_raw: Raw heatmap (HxW, 0-1)
    """
    h, w = fruit_crop_rgb.shape[:2]

    try:
        target_layers = _get_last_conv_layer(unet_model)
        if target_layers is None:
            raise RuntimeError("No Conv2d layer found in U-Net")

        cam = GradCAM(model=unet_model, target_layers=target_layers)

        # Prepare input
        image_resized = cv2.resize(fruit_crop_rgb, (256, 256))
        input_tensor = (
            torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean) / std

        if device == "cuda" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        # Resize back to original crop size
        heatmap_raw = cv2.resize(grayscale_cam, (w, h))
        if heatmap_raw.max() > 0:
            heatmap_raw = heatmap_raw / heatmap_raw.max()

        image_normalized = fruit_crop_rgb.astype(np.float32) / 255.0
        heatmap_overlay = show_cam_on_image(image_normalized, heatmap_raw, use_rgb=True)

        return heatmap_overlay, heatmap_raw

    except Exception as e:
        warnings.warn(f"Grad-CAM failed: {e}. Returning empty heatmap.")
        heatmap_raw = np.zeros((h, w), dtype=np.float32)
        heatmap_overlay = fruit_crop_rgb.copy()
        return heatmap_overlay, heatmap_raw
