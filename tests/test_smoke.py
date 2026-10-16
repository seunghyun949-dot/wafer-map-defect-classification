import numpy as np
import torch

from src.config import CLASS_NAMES, INPUT_SIZE, NUM_CLASSES
from src.gradcam import compute_gradcam
from src.inference import run_inference
from src.model import SimpleCNN
from src.preprocessing import make_synthetic_sample, preprocess_wafer


def test_preprocessing_shape_and_values():
    wafer = make_synthetic_sample("Center")
    resized, tensor = preprocess_wafer(wafer)

    assert resized.shape == (INPUT_SIZE, INPUT_SIZE)
    assert tensor.shape == (1, 1, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_model_forward_and_inference():
    model = SimpleCNN()
    model.eval()

    wafer = make_synthetic_sample("Scratch")
    _, tensor = preprocess_wafer(wafer)

    pred_idx, probabilities = run_inference(
        model,
        tensor,
        torch.device("cpu"),
    )

    assert 0 <= pred_idx < NUM_CLASSES
    assert probabilities.shape == (NUM_CLASSES,)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    assert len(CLASS_NAMES) == NUM_CLASSES


def test_gradcam_output_shape():
    model = SimpleCNN()
    model.eval()

    wafer = make_synthetic_sample("Loc")
    _, tensor = preprocess_wafer(wafer)

    cam = compute_gradcam(
        model,
        tensor,
        target_idx=0,
        device=torch.device("cpu"),
    )

    assert cam.shape == (INPUT_SIZE, INPUT_SIZE)
    assert np.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
