"""
Streamlit Frontend for Pomegranate Orchard Monitor.
Interactive UI: upload, visualize detections, toggle explanations, inspect per-fruit XAI.
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import yaml
from PIL import Image, ImageDraw

# ─── Load Config ──────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

API_URL = CONFIG["frontend"]["api_url"]
DEFAULT_CONF = CONFIG["frontend"]["default_conf_threshold"]
DEFAULT_OPACITY = CONFIG["frontend"]["default_mask_opacity"]
CLASS_NAMES = CONFIG["classes"]["names"]
CLASS_COLORS = {int(k): tuple(v) for k, v in CONFIG["classes"]["colors"].items()}

st.set_page_config(
    page_title=CONFIG["frontend"]["page_title"],
    page_icon=":material/agriculture:",
    layout="wide",
)

st.title("Pomegranate Orchard Monitor")
st.caption("Dual-Explainable Object Detection & Fruit Instance Segmentation")

# ─── Sidebar Controls ─────────────────────────────────
with st.sidebar:
    st.header("Controls")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, DEFAULT_CONF, 0.05)
    mask_opacity = st.slider("Fruit Mask Opacity", 0.0, 1.0, DEFAULT_OPACITY, 0.05)
    show_explanations = st.checkbox("Show Explanations (CAM)", value=True)
    st.divider()
    st.caption("API: " + API_URL)

# ─── Helpers ──────────────────────────────────────────
def _draw_bboxes(pil_image, detections):
    """Draw bounding boxes with class labels on a PIL image."""
    draw = ImageDraw.Draw(pil_image)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det.get("class_id", 0)
        conf = det.get("confidence", 0.0)
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        color = tuple(CLASS_COLORS.get(cls_id, (0, 255, 0)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{name} {conf:.2f}"
        # Draw label background
        text_w = len(label) * 6 + 4
        draw.rectangle([x1, y1 - 14, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - 13), label, fill=(255, 255, 255))
    return pil_image


def _overlay_masks(pil_image, segments, opacity):
    """Overlay fruit masks on the image with given opacity."""
    if not segments or opacity <= 0:
        return pil_image

    img_arr = np.array(pil_image).astype(np.float32)
    h, w = img_arr.shape[:2]

    for seg in segments:
        if not seg.get("fruit_mask_base64"):
            continue
        mask_bytes = base64.b64decode(seg["fruit_mask_base64"])
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        bx1, by1, bx2, by2 = map(int, seg["bbox"])
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            continue
        mask_img = mask_img.resize((bw, bh), Image.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0

        # Colorize mask (green for fruit)
        color = np.array([0, 255, 0], dtype=np.float32)
        for c in range(3):
            img_arr[by1:by2, bx1:bx2, c] = (
                img_arr[by1:by2, bx1:bx2, c] * (1 - opacity * mask_arr)
                + color[c] * opacity * mask_arr
            )

    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


# ─── File Upload ──────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an orchard image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Display original
    original_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(original_image, use_container_width=True)

    # ─── Call API ──────────────────────────────────
    with st.spinner("Analyzing..."):
        files = {"file": uploaded_file.getvalue()}
        params = {"conf": conf_threshold, "explain": show_explanations}

        try:
            resp = requests.post(f"{API_URL}/analyze", files=files, params=params, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Is the FastAPI server running?")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("API request timed out. Try a smaller image or disable explanations.")
            st.stop()

    # ─── Results ────────────────────────────────────
    detections = result.get("detections", [])
    segments = result.get("segments", [])
    explanations = result.get("explanations", [])

    # Build annotated image client-side
    annotated = original_image.copy()
    if detections:
        annotated = _draw_bboxes(annotated, detections)
    if segments and mask_opacity > 0:
        annotated = _overlay_masks(annotated, segments, mask_opacity)

    with col2:
        st.subheader(f"Detections ({len(detections)})")
        if detections:
            st.image(annotated, use_container_width=True)

            # Metrics
            healthy = sum(1 for d in detections if d.get("healthy", False))
            defective = len(detections) - healthy
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Fruits", len(detections))
            m2.metric("Healthy", healthy)
            m3.metric("Defective", defective, delta=f"{defective}" if defective > 0 else None)
        else:
            st.warning("No fruits detected. Try lowering the confidence threshold.")

    # ─── Download Result ────────────────────────────
    if detections:
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            label="Download Annotated Image",
            data=buf.getvalue(),
            file_name="annotated_result.png",
            mime="image/png",
        )

    # ─── Per-Fruit Details ────────────────────────
    if detections:
        st.divider()
        st.subheader("Per-Fruit Analysis")

        for i, (det, seg) in enumerate(zip(detections, segments)):
            exp = explanations[i] if i < len(explanations) else None
            status = "[OK]" if det.get("healthy", False) else "[DEF]"

            with st.expander(
                f"{status} Fruit #{i + 1}: {det['class_name']} ({det['confidence']:.2f})"
            ):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write("**Detection**")
                    st.write(f"Class: `{det['class_name']}`")
                    st.write(f"Confidence: `{det['confidence']:.3f}`")
                    st.write(f"Healthy: `{'Yes' if det.get('healthy') else 'No'}`")
                    bbox = det["bbox"]
                    st.write(
                        f"BBox: `[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]`"
                    )

                with c2:
                    st.write("**Fruit Segmentation**")
                    st.write(f"Fruit Coverage: `{seg.get('fruit_coverage', 0):.3f}`")
                    if seg.get("fruit_mask_base64"):
                        mask_bytes = base64.b64decode(seg["fruit_mask_base64"])
                        mask_img = Image.open(io.BytesIO(mask_bytes))
                        st.image(mask_img, caption="Fruit Mask", width=128)

                with c3:
                    if exp:
                        st.write("**Explainability**")
                        if exp.get("faithfulness"):
                            faith = exp["faithfulness"]
                            st.metric(
                                "Faithfulness", f"{faith['faithfulness_score']:.2f}"
                            )
                            st.metric(
                                "Coverage Drop", f"{faith['confidence_drop']:.3f}"
                            )

                        if exp.get("cross_model_alignment"):
                            align = exp["cross_model_alignment"]
                            st.metric(
                                "CAM Alignment (IoU)", f"{align['alignment_iou']:.2f}"
                            )

                # CAM images
                if exp:
                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        if exp.get("eigencam_base64"):
                            st.caption("YOLO EigenCAM")
                            cam_bytes = base64.b64decode(exp["eigencam_base64"])
                            st.image(Image.open(io.BytesIO(cam_bytes)))
                    with cam_col2:
                        if exp.get("gradcam_base64"):
                            st.caption("U-Net Grad-CAM")
                            cam_bytes = base64.b64decode(exp["gradcam_base64"])
                            st.image(Image.open(io.BytesIO(cam_bytes)))

    # ─── Session Summary ──────────────────────────────
    st.divider()
    st.subheader("Session Summary")
    st.write(f"Total fruits detected: {len(detections)}")
    if segments:
        avg_coverage = np.mean([s.get("fruit_coverage", 0) for s in segments])
        st.metric("Avg. Fruit Coverage", f"{avg_coverage:.3f}")
else:
    st.info("Upload an orchard image to get started.")
