# Project Brief: Apple Orchard Monitoring System

## Objective

Build an end-to-end deep learning system for real-time apple detection and instance segmentation in orchard images, with explainable AI (XAI) validation and a production-ready web interface.

## Scope

- **Detection:** YOLOv8n for apple detection in orchard images
- **Segmentation:** U-Net (ResNet-34) for precise apple boundary masks
- **XAI:** EigenCAM (YOLO) + Grad-CAM (U-Net) with faithfulness and alignment metrics
- **Deployment:** FastAPI backend + Streamlit frontend + Docker

## Design Decisions

### Why YOLOv8n?
- Lightweight (3.2M parameters) -> fast inference on CPU
- Strong COCO pre-training -> good transfer learning
- Single-class detection simplifies the problem

### Why U-Net with SAM Masks?
- Roboflow dataset only had bounding boxes (no polygons)
- SAM generates high-quality apple-shaped masks from bboxes
- U-Net learns precise boundaries for per-fruit coverage analysis

### Why Custom XAI?
- pytorch-grad-cam had compatibility issues with YOLOv8 tuple outputs
- Custom EigenCAM hooks YOLO directly, no external dependency
- Custom Grad-CAM wraps U-Net scalar output for gradient flow

## Benchmarks

| Model | Metric | Value |
|-------|--------|-------|
| YOLOv8n | mAP@0.5 | **98.6%** |
| YOLOv8n | mAP@0.5:0.95 | **82.7%** |
| U-Net | Best Val Loss | **0.324** |
| U-Net | Training Epochs | 14 (early stopping) |
| Pipeline | CPU Throughput | **12.2 FPS** |
| XAI | Faithfulness | **0.302** |

## Dataset

- Merged Roboflow datasets: `paramee/apple-tiyxx` + `l61l/apple-desti`
- ~1,500 total images for detection
- 1,040 SAM-generated masks for segmentation
- 85/15 train/val split

## Architecture

```
Orchard Image -> YOLOv8n -> BBoxes -> U-Net -> Masks
                    |                      |
                EigenCAM              Grad-CAM
                    |                      |
                    +-----> Alignment <----+
```

## Technology Stack

- Python, PyTorch, OpenCV, NumPy
- Ultralytics YOLOv8, segmentation-models-pytorch
- FastAPI, Uvicorn, Streamlit
- Docker, YAML config

## Timeline

- Dataset merging and YOLO training: May 2026
- SAM mask generation and U-Net training: May 2026
- XAI implementation and integration: May 2026
- Deployment and benchmarking: May 2026

## Future Work

- Multi-orchard dataset for domain generalization
- Pixel-level defect segmentation when annotations available
- ONNX/TensorRT edge deployment
- Video processing pipeline
