"""
Grad-CAM for U-Net fruit instance segmentation explainability.
Custom lightweight implementation — hooks target conv layer,
computes gradients w.r.t. output sum, no external library needed.
"""

import warnings

import cv2
import numpy as np
import torch


class _GradCAMHook:
    """Hook to capture features and gradients."""

    def __init__(self):
        self.features = None
        self.gradients = None

    def forward_hook(self, module, input, output):
        self.features = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


def _compute_gradcam(features, gradients):
    """
    Compute Grad-CAM from feature maps and gradients.
    features: (1, C, H, W)
    gradients: (1, C, H, W)
    Returns (H, W) heatmap.
    """
    # Global average pool gradients -> channel weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    # Weighted sum of features
    cam = (weights * features).sum(dim=1).squeeze(0)  # (H, W)
    # ReLU
    cam = torch.relu(cam)
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def _get_last_conv_layer(model):
    """Find the last Conv2d layer in the U-Net decoder."""
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
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
        target_layer = _get_last_conv_layer(unet_model)
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in U-Net")

        hook = _GradCAMHook()
        f_handle = target_layer.register_forward_hook(hook.forward_hook)
        b_handle = target_layer.register_full_backward_hook(hook.backward_hook)

        # Prepare input
        image_resized = cv2.resize(fruit_crop_rgb, (256, 256))
        input_tensor = (
            torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean) / std

        model_device = next(unet_model.parameters()).device
        input_tensor = input_tensor.to(model_device)

        # Forward + backward
        unet_model.zero_grad()
        output = unet_model(input_tensor)
        loss = output.sum()
        loss.backward()

        f_handle.remove()
        b_handle.remove()

        if hook.features is None or hook.gradients is None:
            raise RuntimeError("Failed to capture features or gradients")

        heatmap = _compute_gradcam(hook.features, hook.gradients)

        # Resize back to original crop size
        heatmap_raw = cv2.resize(heatmap, (w, h))
        if heatmap_raw.max() > 0:
            heatmap_raw = heatmap_raw / heatmap_raw.max()

        # Create colored overlay (JET colormap)
        heatmap_color = cv2.applyColorMap((heatmap_raw * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        image_norm = fruit_crop_rgb.astype(np.float32) / 255.0
        heatmap_overlay = (image_norm * 0.5 + heatmap_color.astype(np.float32) / 255.0 * 0.5)
        heatmap_overlay = np.clip(heatmap_overlay, 0, 1)

        return (heatmap_overlay * 255).astype(np.uint8), heatmap_raw

    except Exception as e:
        warnings.warn(f"Grad-CAM failed: {e}. Returning empty heatmap.")
        heatmap_raw = np.zeros((h, w), dtype=np.float32)
        heatmap_overlay = fruit_crop_rgb.copy()
        return heatmap_overlay, heatmap_raw
