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
DEFAULT_MODEL_PATH = Path("models/final_tempered_cnn_earlystop.pt")
GRADCAM_TARGET_LAYER_INDEX = 8
