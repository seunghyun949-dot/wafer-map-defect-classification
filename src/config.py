from pathlib import Path

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

INPUT_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)

MODEL_FILENAME = "final_tempered_cnn_earlystop.pt"
DEFAULT_MODEL_PATH = Path("models") / MODEL_FILENAME
MODEL_DOWNLOAD_URL = (
    "https://github.com/seunghyun949-dot/wafer-map-defect-classification/"
    "releases/latest/download/final_tempered_cnn_earlystop.pt"
)
MODEL_SHA256 = "b9a17df06f85174f4712cda2d42bd9a97f3d527e08b2984567068b55801f848e"

GRADCAM_TARGET_LAYER_INDEX = 8
