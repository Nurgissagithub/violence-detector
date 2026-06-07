"""
Contract tests for the Violence Detector FastAPI service.
These verify the HTTP interface like endpoint presence, validation rules,
and response shape without depending on real model weights or video
decoding. Inference internals are mocked so CI runs in seconds on CPU.
"""
import io
from unittest import mock

import torch

import api


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["num_frames"] == api.NUM_FRAMES
    assert body["weights"] == api.WEIGHTS_PATH
    assert "device" in body


def test_predict_rejects_bad_extension(client):
    files = {"video": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    assert "Unsupported file extension" in r.json()["detail"]


def test_predict_threshold_out_of_range(client, fake_video):
    files = {"video": fake_video}
    r = client.post("/predict?threshold=1.5", files=files)
    # FastAPI Query(ge=0, le=1) -> 422 validation error before inference
    assert r.status_code == 422


def test_predict_no_frames_returns_422(client, fake_video):
    files = {"video": fake_video}
    with mock.patch.object(api, "sample_frames", return_value=[]):
        r = client.post("/predict", files=files)
    assert r.status_code == 422
    assert "decode" in r.json()["detail"].lower()


def test_predict_violence_path(client, fake_video):
    files = {"video": fake_video}
    # One fake frame; transform/model are stubbed so content is irrelevant.
    with mock.patch.object(api, "sample_frames", return_value=["frame"]), \
         mock.patch.object(api.STATE["transform"].__class__, "__call__",
                           lambda self, f: torch.zeros(3, 224, 224)):
        api.STATE["model"] = lambda clip: torch.tensor([[0.1, 5.0]])  # high violence logit
        r = client.post("/predict?threshold=0.65", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Violence"
    assert body["threshold_used"] == 0.65
    assert body["filename"] == "clip.mp4"
    assert 0.0 <= float(body["violence_prob"]) <= 1.0


def test_predict_nonviolence_path(client, fake_video):
    files = {"video": fake_video}
    with mock.patch.object(api, "sample_frames", return_value=["frame"]), \
         mock.patch.object(api.STATE["transform"].__class__, "__call__",
                           lambda self, f: torch.zeros(3, 224, 224)):
        api.STATE["model"] = lambda clip: torch.tensor([[5.0, 0.1]])  # high non-violence logit
        r = client.post("/predict?threshold=0.65", files=files)
    assert r.status_code == 200
    assert r.json()["prediction"] == "NonViolence"


def test_metrics_endpoint_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus exposition format and our custom counter are present.
    assert "http_requests_total" in r.text
    assert "violence_predictions_total" in r.text

