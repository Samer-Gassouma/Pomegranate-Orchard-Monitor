"""
Visualization utilities for heatmap overlays on images.
"""

import cv2
import numpy as np


def overlay_heatmap(image_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay a heatmap on an RGB image.

    Args:
        image_rgb: RGB image (HxWx3, 0-255 or 0-1)
        heatmap: Heatmap (HxW, 0-1)
        alpha: Blend factor (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap

    Returns:
        Blended RGB image (HxWx3, 0-255)
    """
    # Ensure image is 0-255
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    # Resize heatmap to match image
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Colorize
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return blended


def draw_bboxes(image_rgb, detections, class_names, class_colors=None):
    """
    Draw bounding boxes with class labels on an image.

    Args:
        image_rgb: RGB image (HxWx3)
        detections: list of dicts with 'bbox' [x1,y1,x2,y2], 'class_id' (int), 'confidence' (float)
        class_names: dict mapping class_id -> name
        class_colors: dict mapping class_id -> (R,G,B) tuple

    Returns:
        Annotated RGB image
    """
    image = image_rgb.copy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det.get("class_id", det.get("class", 0))
        conf = det.get("confidence", det.get("conf", 0.0))
        name = class_names.get(cls_id, f"cls_{cls_id}")
        color = class_colors.get(cls_id, (0, 255, 0)) if class_colors else (0, 255, 0)

        # Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image
