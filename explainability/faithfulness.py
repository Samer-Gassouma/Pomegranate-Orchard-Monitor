"""
Faithfulness Analysis:
Tests whether Grad-CAM heatmaps genuinely point to decision-critical pixels
by masking hotspot regions and measuring the change in U-Net fruit segmentation output.
"""

import cv2
import numpy as np
import torch


def compute_faithfulness(
    unet_model,
    fruit_crop_rgb,
    gradcam_heatmap,
    device="cuda",
    top_percentile=20,
):
    """
    Compute faithfulness score for a U-Net fruit segmentation prediction.

    Methodology:
    1. Get baseline prediction on original fruit crop
    2. Mask out the top-K% Grad-CAM hotspot pixels (set to black)
    3. Re-run prediction on masked image
    4. Measure delta in fruit coverage probability

    A large drop in confidence = Grad-CAM was pointing at genuinely
    decision-critical pixels = HIGH faithfulness.

    Args:
        unet_model: Trained smp.Unet
        fruit_crop_rgb: Original fruit crop (HxWx3, 0-255)
        gradcam_heatmap: Grad-CAM heatmap (HxW, 0-1)
        device: 'cuda' or 'cpu'
        top_percentile: Top-K% of heatmap pixels to mask (default 20)

    Returns:
        dict with:
            - baseline_confidence: original fruit coverage probability
            - masked_confidence: coverage probability after masking
            - confidence_drop: baseline - masked
            - faithfulness_score: 1.0 = high faithfulness (large drop)
    """
    model_device = next(unet_model.parameters()).device

    # Resize heatmap to match model input size (256x256)
    image_resized = cv2.resize(fruit_crop_rgb, (256, 256))
    heatmap_resized = cv2.resize(gradcam_heatmap, (256, 256))
    input_tensor = (
        torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.to(model_device)

    # Baseline prediction
    with torch.no_grad():
        baseline_pred = unet_model(input_tensor)
    baseline_conf = baseline_pred.mean().item()

    # Create masked image: black out top-K% heatmap pixels
    threshold = np.percentile(heatmap_resized, 100 - top_percentile)
    mask = heatmap_resized > threshold
    masked_image = image_resized.copy()
    masked_image[mask] = 0  # black out

    # Masked prediction
    masked_tensor = (
        torch.from_numpy(masked_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )
    masked_tensor = (masked_tensor - mean) / std
    masked_tensor = masked_tensor.to(model_device)

    with torch.no_grad():
        masked_pred = unet_model(masked_tensor)
    masked_conf = masked_pred.mean().item()

    confidence_drop = baseline_conf - masked_conf

    try:
        return {
            "baseline_confidence": round(baseline_conf, 4),
            "masked_confidence": round(masked_conf, 4),
            "confidence_drop": round(confidence_drop, 4),
            "faithfulness_score": round(
                min(confidence_drop / (baseline_conf + 1e-7), 1.0), 4
            ),
            "n_pixels_masked": int(mask.sum()),
            "mask_fraction": round(mask.sum() / (h * w), 4),
        }
    except Exception:
        return {
            "baseline_confidence": round(baseline_conf, 4),
            "masked_confidence": round(masked_conf, 4),
            "confidence_drop": 0.0,
            "faithfulness_score": 0.0,
            "n_pixels_masked": 0,
            "mask_fraction": 0.0,
        }
