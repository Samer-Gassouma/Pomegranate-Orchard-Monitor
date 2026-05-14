# Pomegranate Orchard Monitor — Docker Image
# Usage:
#   docker build -t pom-monitor .
#   docker run -p 8000:8000 pom-monitor
#
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.yaml .
COPY api/ ./api/
COPY explainability/ ./explainability/
COPY frontend/ ./frontend/
COPY training/ ./training/
COPY pomegranate_models/ ./pomegranate_models/

# Create results directories
RUN mkdir -p results/figures results/metrics

# Expose FastAPI port
EXPOSE 8000

# Default: run FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
