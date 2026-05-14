"""
Cross-Model Attention Alignment:
Measures whether YOLO's EigenCAM and U-Net's Grad-CAM agree
on which pixels are important for fruit detection and segmentation.

High alignment = both independently-trained models focus on the same
region = stronger trust in the overall system.
"""

import cv2
import numpy as np


def compute_cross_model_alignment(
    eigencam_heatmap,
    gradcam_heatmap,
    bbox_xyxy,
    image_shape,
    threshold=0.5,
):
    """
    Compute IoU between thresholded YOLO EigenCAM and U-Net Grad-CAM
    within a detected fruit's bounding box.

    Args:
        eigencam_heatmap: YOLO EigenCAM heatmap (H_img x W_img, 0-1)
        gradcam_heatmap: U-Net Grad-CAM heatmap (H_crop x W_crop, 0-1)
        bbox_xyxy: [x1, y1, x2, y2] of detected fruit in image coords
        image_shape: (H, W) of full image
        threshold: binarization threshold for heatmaps

    Returns:
        dict with:
            - alignment_iou: IoU between binary YOLO-CAM and U-Net-CAM regions
            - yolo_hotspot_fraction: fraction of bbox identified as important by YOLO
            - unet_hotspot_fraction: fraction of bbox identified as important by U-Net
    """
    h_img, w_img = image_shape
    x1, y1, x2, y2 = map(int, bbox_xyxy)

    # Crop EigenCAM to bbox region
    eigencam_crop = eigencam_heatmap[y1:y2, x1:x2]

    # Resize U-Net Grad-CAM to match crop size
    crop_h, crop_w = y2 - y1, x2 - x1
    gradcam_resized = cv2.resize(gradcam_heatmap, (crop_w, crop_h))

    # Binarize both
    eigencam_binary = (eigencam_crop > threshold).astype(np.uint8)
    gradcam_binary = (gradcam_resized > threshold).astype(np.uint8)

    # Compute IoU
    intersection = np.logical_and(eigencam_binary, gradcam_binary).sum()
    union = np.logical_or(eigencam_binary, gradcam_binary).sum()

    iou = intersection / (union + 1e-7)

    try:
        return {
            "alignment_iou": round(float(iou), 4),
            "yolo_hotspot_fraction": round(
                eigencam_binary.sum() / (crop_h * crop_w + 1e-7), 4
            ),
            "unet_hotspot_fraction": round(
                gradcam_binary.sum() / (crop_h * crop_w + 1e-7), 4
            ),
        }
    except Exception:
        return {
            "alignment_iou": 0.0,
            "yolo_hotspot_fraction": 0.0,
            "unet_hotspot_fraction": 0.0,
        }
