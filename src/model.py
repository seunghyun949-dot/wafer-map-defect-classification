from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn as nn

from .config import DEFAULT_MODEL_PATH, NUM_CLASSES


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
        elif model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            source = str(model_path)
        else:
            return None, (
                "No checkpoint loaded. Upload "
                "final_tempered_cnn_earlystop.pt or place it at "
                f"{model_path}."
            )

        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.eval()
        return model, f"Model loaded from: {source}"

    except Exception as exc:
        return None, f"Checkpoint load failed: {exc}"
