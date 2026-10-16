from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

import torch
import torch.nn as nn

from .config import (
    DEFAULT_MODEL_PATH,
    MODEL_DOWNLOAD_URL,
    MODEL_SHA256,
    NUM_CLASSES,
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    return cleaned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_default_checkpoint(
    model_path: Path = DEFAULT_MODEL_PATH,
    download_url: str = MODEL_DOWNLOAD_URL,
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        if _sha256(model_path) != MODEL_SHA256:
            raise ValueError(
                "Existing checkpoint hash does not match the verified final model."
            )
        return model_path

    temp_path = model_path.with_suffix(model_path.suffix + ".download")
    request = Request(
        download_url,
        headers={"User-Agent": "wafer-map-defect-classification-streamlit"},
    )

    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        if _sha256(temp_path) != MODEL_SHA256:
            raise ValueError(
                "Downloaded checkpoint hash does not match the verified final model."
            )

        temp_path.replace(model_path)
        return model_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def load_checkpoint_model(
    device: torch.device,
    checkpoint_upload: BinaryIO | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[SimpleCNN | None, str]:
    model = SimpleCNN().to(device)

    try:
        if checkpoint_upload is not None:
            checkpoint_upload.seek(0)
            checkpoint = torch.load(checkpoint_upload, map_location=device)
            source = "uploaded checkpoint"
        else:
            checkpoint_path = download_default_checkpoint(model_path=model_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            source = "verified release checkpoint"

        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.eval()
        return model, f"Model loaded from: {source}"

    except Exception as exc:
        return None, f"Checkpoint load failed: {exc}"
