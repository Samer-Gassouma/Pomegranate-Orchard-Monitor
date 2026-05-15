"""
Model loading and inference logic for FastAPI backend.
Loads configuration from config.yaml for all paths and hyperparameters.
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from ultralytics import YOLO

from explainability.alignment import compute_cross_model_alignment
from explainability.eigencam_yolo import explain_yolo_detection
from explainability.faithfulness import compute_faithfulness
from explainability.gradcam_unet import explain_unet_segmentation

# ─── Load Config ──────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()
DEVICE_CFG = CONFIG["inference"]["device"]
if DEVICE_CFG == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = DEVICE_CFG

YOLO_PATH = CONFIG["paths"]["yolo_weights"]
UNET_PATH = CONFIG["paths"]["unet_weights"]
YOLO_IMGSZ = CONFIG["inference"]["yolo_imgsz"]
UNET_IMGSZ = CONFIG["inference"]["unet_imgsz"]
NORMALIZE_MEAN = torch.tensor(CONFIG["inference"]["normalize_mean"]).view(1, 3, 1, 1)
NORMALIZE_STD = torch.tensor(CONFIG["inference"]["normalize_std"]).view(1, 3, 1, 1)

CLASS_NAMES = CONFIG["classes"]["names"]
CLASS_COLORS = {int(k): tuple(v) for k, v in CONFIG["classes"]["colors"].items()}
HEALTHY_CLASSES = set(CONFIG["classes"].get("healthy", []))
FAITH_TOP_PCT = CONFIG["xai"]["faithfulness_top_percentile"]
CAM_THRESHOLD = CONFIG["xai"]["cam_threshold"]

# ─── Model Loading ────────────────────────────────────
yolo_model = None
unet_model = None


def load_models():
    global yolo_model, unet_model

    if os.path.exists(YOLO_PATH):
        yolo_model = YOLO(YOLO_PATH)
        print(f"[INFO] YOLO loaded on {DEVICE}")
    else:
        print(f"[WARNING] YOLO weights not found: {YOLO_PATH}")

    unet_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    if os.path.exists(UNET_PATH):
        unet_model.load_state_dict(
            torch.load(UNET_PATH, map_location=DEVICE, weights_only=True)
        )
        print(f"[INFO] U-Net loaded on {DEVICE}")
    else:
        print(f"[WARNING] U-Net weights not found: {UNET_PATH}")

    unet_model.to(DEVICE)
    unet_model.eval()


# ─── Helpers ──────────────────────────────────────────
def image_to_base64(image_rgb):
    """Convert RGB numpy array to base64 PNG string."""
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def read_image(file_bytes):
    """Read image from uploaded bytes, return RGB numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image from uploaded bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─── Inference Functions ──────────────────────────────
def run_detection(image_rgb, conf_threshold=0.25):
    """Run YOLO detection. Returns list of DetectionResult dicts."""
    if yolo_model is None:
        raise RuntimeError("YOLO model not loaded")

    results = yolo_model(image_rgb, conf=conf_threshold, device=DEVICE)[0]
    detections = []

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(classes[i])
            detections.append(
                {
                    "bbox": boxes[i].tolist(),
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
                    "confidence": float(confs[i]),
                    "healthy": cls_id in HEALTHY_CLASSES,
                }
            )

    return detections


def run_segmentation(image_rgb, detections):
    """Run U-Net fruit instance segmentation on each detected crop.
    Returns list of SegmentResult dicts with fruit masks."""
    if unet_model is None:
        raise RuntimeError("U-Net model not loaded")

    segments = []
    h, w = image_rgb.shape[:2]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_w, crop_h = x2 - x1, y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            segments.append(
                {
                    "fruit_index": i,
                    "bbox": det["bbox"],
                    "fruit_mask_base64": None,
                    "fruit_coverage": 0.0,
                    "mask_width": 0,
                    "mask_height": 0,
                }
            )
            continue

        crop = image_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (UNET_IMGSZ, UNET_IMGSZ))

        tensor = (
            torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        tensor = (tensor - NORMALIZE_MEAN) / NORMALIZE_STD
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            pred = unet_model(tensor)[0, 0].cpu().numpy()

        # Coverage = fraction of pixels classified as fruit
        fruit_coverage = float(pred.mean())

        # Resize mask back to original crop size for client overlay
        pred_resized = cv2.resize(pred, (crop_w, crop_h))
        mask_binary = (pred_resized > 0.5).astype(np.uint8) * 255

        segments.append(
            {
                "fruit_index": i,
                "bbox": det["bbox"],
                "fruit_mask_base64": image_to_base64(
                    cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)
                ),
                "fruit_coverage": fruit_coverage,
                "mask_width": crop_w,
                "mask_height": crop_h,
            }
        )

    return segments


def run_explain(image_rgb, detections, segments):
    """Run dual explainability (EigenCAM + Grad-CAM) with faithfulness."""
    if yolo_model is None or unet_model is None:
        raise RuntimeError("Models not loaded")

    explanations = []

    for i, (det, seg) in enumerate(zip(detections, segments)):
        x1, y1, x2, y2 = map(int, det["bbox"])

        # YOLO EigenCAM — explains what the detector looked at
        eigencam_overlay, eigencam_raw = explain_yolo_detection(
            yolo_model,
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            [x1, y1, x2, y2],
        )

        # U-Net Grad-CAM — explains fruit boundary decision
        crop = image_rgb[y1:y2, x1:x2]
        gradcam_overlay, gradcam_raw = explain_unet_segmentation(
            unet_model,
            crop,
            device=DEVICE,
        )

        # Faithfulness — mask top CAM pixels and re-run
        faithfulness = compute_faithfulness(
            unet_model,
            crop,
            gradcam_raw,
            device=DEVICE,
            top_percentile=FAITH_TOP_PCT,
        )

        # Cross-model alignment
        alignment = compute_cross_model_alignment(
            eigencam_raw,
            gradcam_raw,
            [x1, y1, x2, y2],
            image_rgb.shape[:2],
            threshold=CAM_THRESHOLD,
        )

        explanations.append(
            {
                "fruit_index": i,
                "eigencam_base64": image_to_base64(eigencam_overlay),
                "gradcam_base64": image_to_base64(gradcam_overlay),
                "faithfulness": faithfulness,
                "cross_model_alignment": alignment,
            }
        )

    return explanations
