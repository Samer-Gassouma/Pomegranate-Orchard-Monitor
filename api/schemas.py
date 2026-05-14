"""Pydantic models for FastAPI request/response schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] in pixel coordinates")
    class_id: int
    class_name: str
    confidence: float
    healthy: bool = Field(..., description="True if class is crown_good or surface_good")


class SegmentResult(BaseModel):
    fruit_index: int
    bbox: List[float]
    fruit_mask_base64: Optional[str] = None
    fruit_coverage: float = Field(..., ge=0.0, le=1.0, description="Fraction of crop pixels classified as fruit")
    mask_width: int
    mask_height: int


class ExplainResult(BaseModel):
    fruit_index: int
    eigencam_base64: Optional[str] = None
    gradcam_base64: Optional[str] = None
    faithfulness: Optional[dict] = None
    cross_model_alignment: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    detections: List[DetectionResult]
    segments: List[SegmentResult]
    explanations: List[ExplainResult]
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    unet_loaded: bool
    device: str
