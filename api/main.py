"""
FastAPI Backend for Apple Orchard Monitoring System.
Endpoints: /health, /detect, /segment, /analyze, /explain
"""

from contextlib import asynccontextmanager

import api.inference as inference
from api.inference import (
    DEVICE,
    image_to_base64,
    load_models,
    read_image,
    run_detection,
    run_explain,
    run_segmentation,
)
from api.schemas import AnalyzeResponse, HealthResponse
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Apple Orchard Monitor",
    description="Dual-Explainable Object Detection & Fruit Instance Segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        yolo_loaded=inference.yolo_model is not None,
        unet_loaded=inference.unet_model is not None,
        device=DEVICE,
    )


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
):
    """Run YOLO object detection on uploaded image."""
    image_rgb = read_image(await file.read())
    detections = run_detection(image_rgb, conf_threshold=conf)
    return {"detections": detections, "count": len(detections)}


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
):
    """Run detection + fruit instance segmentation pipeline."""
    image_rgb = read_image(await file.read())
    detections = run_detection(image_rgb, conf_threshold=conf)
    segments = run_segmentation(image_rgb, detections)
    return {"segments": segments, "count": len(segments)}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    explain: bool = Query(True),
):
    """Full pipeline: detect + fruit segment + explain."""
    image_rgb = read_image(await file.read())

    detections = run_detection(image_rgb, conf_threshold=conf)
    segments = run_segmentation(image_rgb, detections)

    explanations = []
    if explain:
        explanations = run_explain(image_rgb, detections, segments)

    return AnalyzeResponse(
        detections=detections,
        segments=segments,
        explanations=explanations,
        annotated_image_base64=image_to_base64(image_rgb),
    )


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    bbox_x1: float = Query(...),
    bbox_y1: float = Query(...),
    bbox_x2: float = Query(...),
    bbox_y2: float = Query(...),
):
    """Run explainability on a specific fruit detection."""
    image_rgb = read_image(await file.read())
    det = {"bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2]}
    seg = {"fruit_index": 0, "bbox": det["bbox"]}

    explanations = run_explain(image_rgb, [det], [seg])
    return explanations[0] if explanations else {}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
