import io
import sys
import importlib
from unittest import mock

import pytest


@pytest.fixture(scope="session")
def client():
    """
    TestClient for api:app with the heavy startup neutralized.
    api.py's lifespan handler builds a ViolenceClassifier, checks the
    weights file exists, and loads it with safetensors.load_file. None of
    that should run in CI, so we patch:
      - api.os.path.exists -> True (skip the weights-missing guard)
      - api.load_file -> empty dict (no real safetensors read)
      - api.ViolenceClassifier -> a light stub whose forward is overridable
    """
    import api

    class _StubModel:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def load_state_dict(self, *args, **kwargs):
            return None

        def eval(self):
            return self

        def __call__(self, clip):
            # Overridden per-test via api.STATE["model"]; default benign.
            import torch
            return torch.tensor([[2.0, 0.5]])

    with mock.patch.object(api, "ViolenceClassifier", _StubModel), \
         mock.patch.object(api.os.path, "exists", return_value=True), \
         mock.patch.object(api, "load_file", return_value={}):
        from fastapi.testclient import TestClient
        # TestClient context manager triggers the lifespan startup.
        with TestClient(api.app) as test_client:
            yield test_client


@pytest.fixture
def fake_video():
    """A minimal fake mp4 upload (filename + bytes)."""
    return ("clip.mp4", io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 256), "video/mp4")
