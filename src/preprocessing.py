from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .config import INPUT_SIZE


def resize_wafer(wafer: np.ndarray) -> np.ndarray:
    return cv2.resize(
        wafer.astype(np.uint8),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def validate_wafer(wafer: np.ndarray) -> None:
    if wafer.ndim != 2:
        raise ValueError(
            f"Expected a 2D wafer map, but received shape {wafer.shape}."
        )

    if not np.isfinite(wafer).all():
        raise ValueError("Input contains NaN or infinite values.")

    unique_values = np.unique(wafer)
    if not np.all(np.isin(unique_values, [0, 1, 2])):
        raise ValueError(
            "Wafer array must contain only categorical values 0, 1, and 2. "
            f"Found values such as: {unique_values[:10]}"
        )


def preprocess_wafer(wafer: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    wafer = np.asarray(wafer)
    validate_wafer(wafer)

    resized = resize_wafer(wafer)
    normalized = resized.astype(np.float32) / 2.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    return resized, tensor


def load_array_file(uploaded_file) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".npy":
        uploaded_file.seek(0)
        return np.load(uploaded_file)

    if suffix == ".csv":
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None).values

    raise ValueError("Supported array formats are .npy and .csv.")


def image_to_wafer(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("L")
    array = np.asarray(image)
    unique = np.unique(array)

    if len(unique) <= 3:
        mapping = {value: idx for idx, value in enumerate(sorted(unique))}
        return np.vectorize(mapping.get)(array).astype(np.uint8)

    wafer = np.zeros_like(array, dtype=np.uint8)
    wafer[array >= 85] = 1
    wafer[array >= 170] = 2
    return wafer


def make_synthetic_sample(kind: str, size: int = 48) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = size * 0.44

    wafer = np.zeros((size, size), dtype=np.uint8)
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    wafer[mask] = 1

    if kind == "Center":
        defect = (
            (x - center) ** 2 + (y - center) ** 2
            <= (size * 0.10) ** 2
        )
        wafer[defect & mask] = 2

    elif kind == "Edge-Ring":
        r2 = (x - center) ** 2 + (y - center) ** 2
        outer = radius ** 2
        inner = (radius - size * 0.06) ** 2
        wafer[(r2 <= outer) & (r2 >= inner)] = 2

    elif kind == "Scratch":
        for offset in range(-1, 2):
            rr = np.arange(size)
            cc = np.clip(
                (0.65 * rr + size * 0.12 + offset).astype(int),
                0,
                size - 1,
            )
            valid = mask[rr, cc]
            wafer[rr[valid], cc[valid]] = 2

    elif kind == "Loc":
        rr, cc = np.ogrid[:size, :size]
        loc = (
            (rr - size * 0.35) ** 2
            + (cc - size * 0.68) ** 2
            <= (size * 0.08) ** 2
        )
        wafer[loc & mask] = 2

    return wafer
