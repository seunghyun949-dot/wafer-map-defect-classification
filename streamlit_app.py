from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import CLASS_NAMES, INPUT_SIZE
from src.gradcam import compute_gradcam
from src.inference import run_inference
from src.model import load_checkpoint_model
from src.preprocessing import (
    image_to_wafer,
    load_array_file,
    make_synthetic_sample,
    preprocess_wafer,
)
from src.visualization import plot_gradcam, plot_wafer


st.set_page_config(
    page_title="WM-811K Wafer Defect Classifier",
    page_icon="🧩",
    layout="wide",
)

st.title("WM-811K Wafer Map Defect Classifier")
st.caption(
    "Final Tempered CNN · 9 classes · "
    "64×64 nearest-neighbor preprocessing · Grad-CAM"
)

st.info(
    "This demo reproduces the project inference pipeline. "
    "The verified final checkpoint is loaded automatically from the GitHub Release."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with st.sidebar:
    st.header("Model")
    st.write(f"Device: **{device}**")
    st.caption("Build: v1.0.0 · auto checkpoint")

    checkpoint_upload = st.file_uploader(
        "Fallback: upload checkpoint (.pt)",
        type=["pt", "pth"],
        help=(
            "Normally no upload is needed. Use this only if automatic "
            "checkpoint download is temporarily unavailable."
        ),
    )

    model, model_message = load_checkpoint_model(
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
        "the original 0/1/2 wafer categories. "
        "For reproducible inference, use .npy or .csv."
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


left, right = st.columns(2)

with left:
    st.subheader("Original input")
    st.write(f"Shape: **{np.asarray(wafer).shape}**")
    fig = plot_wafer(np.asarray(wafer), "Original wafer map")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Model input")
    st.write(f"Shape: **{INPUT_SIZE} × {INPUT_SIZE}**")
    fig = plot_wafer(resized, "64×64 nearest resize")
    st.pyplot(fig)
    plt.close(fig)


st.divider()

if model is None:
    st.warning(
        "Preprocessing is working, but the automatic checkpoint could not "
        "be loaded. Use the fallback checkpoint uploader in the sidebar."
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

    grad_left, grad_right = st.columns(2)

    with grad_left:
        fig = plot_gradcam(
            normalized,
            cam,
            f"Grad-CAM: {pred_class}",
        )
        st.pyplot(fig)
        plt.close(fig)

    with grad_right:
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
    "Project final test: Accuracy 97.15% · "
    "Balanced Accuracy 81.37% · "
    "Macro-F1 84.34% · Scratch Recall 45.20%"
)
