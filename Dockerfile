FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system libs. opencv-python-headless bundles its own ffmpeg libs,
# so no system ffmpeg package is needed. libgl1/libglib2.0-0 satisfy OpenCV's
# shared-lib dependencies; curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch/torchvision: far smaller than the default CUDA wheels and the container has no GPU anyway. Unpinned so pip picks an available CPU wheel.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

ENV WEIGHTS_PATH=models/violence_classifier.safetensors \
    NUM_FRAMES=16 \
    THRESHOLD=0.65 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
