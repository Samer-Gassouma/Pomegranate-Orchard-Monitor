# Pomegranate Orchard Monitor

Master's project: End-to-end Deep Learning computer vision system for Smart Agriculture.

## Overview

This system performs **object detection** and **fruit instance segmentation** on pomegranate orchard images using:
- **YOLOv8m** for fruit detection and defect classification (6 classes)
- **U-Net (ResNet-34)** for precise fruit boundary segmentation
- **Dual XAI**: EigenCAM (YOLO) + Grad-CAM (U-Net) with faithfulness and cross-model alignment metrics
- **FastAPI** backend + **Streamlit** frontend for deployment

## Project Structure

```
.
├── api/                       # FastAPI backend
│   ├── main.py               # API endpoints
│   ├── inference.py          # Model loading & inference
│   └── schemas.py            # Pydantic request/response models
├── explainability/           # XAI modules
│   ├── eigencam_yolo.py
│   ├── gradcam_unet.py
│   ├── faithfulness.py
│   ├── alignment.py
│   └── visualize.py
├── frontend/                 # Streamlit UI
│   └── app.py
├── training/                 # Training & evaluation scripts
│   └── evaluate.py
├── pomegranate_models/       # Trained model weights
│   ├── yolov8_pomegranate.pt
│   └── unet_defect.pth
├── presentation/           # Slidev slides for defense
│   ├── slides.md
│   └── package.json
├── config.yaml               # Central configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── README.md                 # This file
├── REPORT.md                 # Technical report
└── PROJECT_BRIEF.md        # Project scope and design decisions
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 3. Start Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Open: http://localhost:8501

### 4. Run Evaluation

```bash
python training/evaluate.py --test_dir path/to/test/images --save_viz
```

## Training

Model training was performed on Kaggle using the Makinay Pomegranate Dataset. The training notebook (`pomegranate-orchard-monitor-yolo-u-net-xai.ipynb`) contains the full pipeline: dataset download, YOLOv8m training, U-Net training, and weight export.

**Notebook:** `pomegranate-orchard-monitor-yolo-u-net-xai.ipynb`

**Kaggle link:** https://www.kaggle.com/code/samergassouma/pomegranate-orchard-monitor-yolo-u-net-xai

## Report

A formal project report is available as a compiled PDF:

**[View Report (PDF)](./latex_report/main.pdf)**

The LaTeX source is in `latex_report/main.tex`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server & model status |
| `/detect` | POST | YOLO detection only |
| `/segment` | POST | Detection + fruit segmentation |
| `/analyze` | POST | Full pipeline + XAI |
| `/explain` | POST | Explainability for a specific bbox |

## Configuration

All paths and hyperparameters are in `config.yaml`:
- Model weights paths
- Confidence thresholds
- Image sizes
- Class names & colors
- XAI parameters

## Classes

| ID | Name | Healthy? |
|----|------|----------|
| 0 | crown_damaged | No |
| 1 | crown_good | Yes |
| 2 | surface_damaged | No |
| 3 | surface_good | Yes |
| 4 | surface_burnt | No |
| 5 | surface_cracked | No |

## Docker

```bash
docker build -t pom-monitor .
docker run -p 8000:8000 pom-monitor
```



## License

Academic use only -- Master's project submission.
