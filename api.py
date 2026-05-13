# FastAPI inference service for the violence classifier
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from safetensors.torch import load_file

from dataset import build_transforms, sample_frames
from model import ViolenceClassifier

# Configurable parameters (via environment variables)
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "models/violence_classifier.safetensors")
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "16"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.65"))
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

LABELS = {0: "NonViolence", 1: "Violence"}


# Global state for model and device (loaded once at startup)
STATE: Dict[str, Any] = {"model": None, "transform": None, "device": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceClassifier(num_frames=NUM_FRAMES).to(device)

    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"Weights not found at {WEIGHTS_PATH}")

    state_dict = load_file(WEIGHTS_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    STATE["model"] = model
    STATE["transform"] = build_transforms(train=False)
    STATE["device"] = device

    print(f"[api] model loaded from {WEIGHTS_PATH} on {device}")
    yield


app = FastAPI(
    title="Violence Detector API",
    version="1.0.0",
    description="Binary video classifier (Violence / NonViolence) — EfficientNet-B0 + temporal mean pooling.",
    lifespan=lifespan,
)

def _run_inference(video_path: str, threshold: float) -> Dict[str, Any]:
    model = STATE["model"]
    transform = STATE["transform"]
    device = STATE["device"]

    frames = sample_frames(video_path, NUM_FRAMES)
    if not frames:
        raise HTTPException(status_code=422, detail="Could not decode video (no frames sampled).")

    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1)[0]

    violence_prob = probs[1].item()
    pred = 1 if violence_prob >= threshold else 0

    return {
        "prediction": LABELS[pred],
        "threshold_used": threshold,
        "confidence": f"{probs[pred].item():.4f}",
        "violence_prob": f"{violence_prob:.4f}",
        "non_violence_prob": f"{probs[0].item():.4f}",
    }


# API endpoints
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Liveness / readiness check."""
    return {
        "status": "ok" if STATE["model"] is not None else "loading",
        "device": str(STATE["device"]) if STATE["device"] is not None else None,
        "num_frames": NUM_FRAMES,
        "weights": WEIGHTS_PATH,
    }


@app.post("/predict")
async def predict(
    video: UploadFile = File(..., description="Video file (mp4/avi/mov/mkv/webm)"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """Run the classifier on an uploaded video file."""
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    # Save uploaded file to a temporary location for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        try:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name
        finally:
            video.file.close()

    try:
        result = _run_inference(tmp_path, threshold)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    result["filename"] = video.filename
    return result
