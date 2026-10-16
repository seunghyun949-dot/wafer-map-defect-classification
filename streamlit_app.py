from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]

DEFAULT_MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/final_tempered_cnn_earlystop.pt")
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 9):
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

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def resize_wafer(wafer: np.ndarray) -> np.ndarray:
    return cv2.resize(
        wafer.astype(np.uint8),
        (64, 64),
        interpolation=cv2.INTER_NEAREST,
    )


def preprocess_wafer(wafer: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    wafer = np.asarray(wafer)

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
        mapping = {
            value: idx
            for idx, value in enumerate(sorted(unique))
        }
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
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    return cleaned


def load_model_from_source(device, checkpoint_upload=None):
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)

    try:
        if checkpoint_upload is not None:
            checkpoint_upload.seek(0)
            checkpoint = torch.load(
                checkpoint_upload,
                map_location=device,
            )
            source = "uploaded checkpoint"

        elif DEFAULT_MODEL_PATH.exists():
            checkpoint = torch.load(
                DEFAULT_MODEL_PATH,
                map_location=device,
            )
            source = str(DEFAULT_MODEL_PATH)

        else:
            return None, (
                "No checkpoint loaded. Upload final_tempered_cnn_earlystop.pt "
                "in the sidebar or place it at "
                f"{DEFAULT_MODEL_PATH}."
            )

        model.load_state_dict(
            extract_state_dict(checkpoint),
            strict=True,
        )
        model.eval()

        return model, f"Model loaded from: {source}"

    except Exception as exc:
        return None, f"Checkpoint load failed: {exc}"


def run_inference(model, input_tensor, device):
    x = input_tensor.to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probabilities.argmax().item())

    return pred_idx, probabilities.cpu().numpy()


def compute_gradcam(model, input_tensor, target_idx, device):
    x = input_tensor.to(device)
    x.requires_grad_(True)

    activations = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.retain_grad()

    handle = model.features[8].register_forward_hook(forward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, target_idx].backward()

    feature_map = activations["value"]
    gradients = feature_map.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    cam = (
        weights * feature_map
    ).sum(dim=1, keepdim=True)

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    cam = cam.detach().cpu().numpy()

    if cam.max() > cam.min():
        cam = (
            (cam - cam.min())
            / (cam.max() - cam.min())
        )
    else:
        cam = np.zeros_like(cam)

    handle.remove()

    return cam


def plot_wafer(wafer, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wafer, cmap="gray", vmin=0, vmax=2)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_gradcam(normalized_wafer, cam, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(normalized_wafer, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="WM-811K Wafer Defect Classifier",
    page_icon="🧩",
    layout="wide",
)

st.title("WM-811K Wafer Map Defect Classifier")
st.caption(
    "Final Tempered CNN · 9 classes · 64×64 nearest-neighbor preprocessing · Grad-CAM"
)

st.info(
    "This demo reproduces the project inference pipeline. "
    "The trained checkpoint is intentionally not committed to GitHub."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.header("Model")
    st.write(f"Device: **{device}**")

    checkpoint_upload = st.file_uploader(
        "Upload final checkpoint (.pt)",
        type=["pt", "pth"],
        help=(
            "Use final_tempered_cnn_earlystop.pt. "
            "Model files are excluded from GitHub."
        ),
    )

    model, model_message = load_model_from_source(
        device=device,
        checkpoint_upload=checkpoint_upload,
    )

    if model is None:
        st.warning(model_message)
    else:
        st.success(model_message)

    st.divider()

    st.header("Input")
    input_mode = st.radio(
        "Choose input source",
        [
            "Upload wafer array",
            "Upload image",
            "Synthetic sample",
        ],
    )


wafer = None
input_note = None

if input_mode == "Upload wafer array":
    uploaded = st.file_uploader(
        "Upload .npy or .csv wafer map",
        type=["npy", "csv"],
        help="Recommended format. Values must be 0, 1, or 2.",
    )

    if uploaded is not None:
        try:
            wafer = load_array_file(uploaded)
        except Exception as exc:
            st.error(str(exc))

elif input_mode == "Upload image":
    uploaded = st.file_uploader(
        "Upload PNG/JPG image",
        type=["png", "jpg", "jpeg"],
    )

    st.warning(
        "Image upload is approximate unless the image directly encodes "
        "the original 0/1/2 wafer categories. For reproducible inference, "
        "use .npy or .csv."
    )

    if uploaded is not None:
        try:
            wafer = image_to_wafer(uploaded)
            input_note = (
                "Image was converted to categorical 0/1/2 values "
                "before preprocessing."
            )
        except Exception as exc:
            st.error(str(exc))

else:
    synthetic_kind = st.selectbox(
        "Synthetic pattern",
        ["Center", "Edge-Ring", "Loc", "Scratch"],
    )

    wafer = make_synthetic_sample(synthetic_kind)
    input_note = (
        "Synthetic samples are for UI/preprocessing checks only. "
        "They are not WM-811K test samples and should not be used "
        "to evaluate model performance."
    )


if wafer is None:
    st.write("Upload a wafer map to begin.")
    st.stop()


try:
    resized, input_tensor = preprocess_wafer(wafer)
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()


if input_note:
    st.caption(input_note)


left, middle = st.columns(2)

with left:
    st.subheader("Original input")
    st.write(f"Shape: **{np.asarray(wafer).shape}**")
    st.pyplot(plot_wafer(np.asarray(wafer), "Original wafer map"))

with middle:
    st.subheader("Model input")
    st.write("Shape: **64 × 64**")
    st.pyplot(plot_wafer(resized, "64×64 nearest resize"))


st.divider()

if model is None:
    st.warning(
        "Preprocessing is working, but prediction/Grad-CAM requires "
        "the final model checkpoint. Upload the .pt file in the sidebar."
    )
    st.stop()


pred_idx, probabilities = run_inference(
    model,
    input_tensor,
    device,
)

pred_class = CLASS_NAMES[pred_idx]
confidence = float(probabilities[pred_idx])

metric_col1, metric_col2 = st.columns(2)

metric_col1.metric("Predicted class", pred_class)
metric_col2.metric("Confidence", f"{confidence:.2%}")


st.subheader("Class probabilities")

prob_df = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Probability": probabilities,
    }
).set_index("Class")

st.bar_chart(prob_df)


st.subheader("Grad-CAM")

try:
    cam = compute_gradcam(
        model,
        input_tensor,
        pred_idx,
        device,
    )

    normalized = resized.astype(np.float32) / 2.0

    grad_col1, grad_col2 = st.columns(2)

    with grad_col1:
        st.pyplot(
            plot_gradcam(
                normalized,
                cam,
                f"Grad-CAM: {pred_class}",
            )
        )

    with grad_col2:
        st.write(
            "**Interpretation note**\n\n"
            "Grad-CAM visualizes spatial regions associated with "
            "the model prediction. It does **not** establish a "
            "physical semiconductor-process root cause."
        )

        st.write(
            f"CAM range: **{cam.min():.4f} – {cam.max():.4f}**"
        )

        if np.allclose(cam, 0):
            st.warning(
                "This sample produced a near-zero Grad-CAM. "
                "The project also observed that Grad-CAM is not "
                "equally informative for every wafer/sample."
            )

except Exception as exc:
    st.error(f"Grad-CAM failed: {exc}")


st.divider()
st.caption(
    "Project final test: Accuracy 97.15% · Balanced Accuracy 81.37% · "
    "Macro-F1 84.34% · Scratch Recall 45.20%"
)
