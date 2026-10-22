import os
from pathlib import Path
import torch

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent

# Centralized Dataset Configuration
DATASET_ROOT = r"C:\Users\Narasimha\OneDrive\Documents\zip\zip"
REAL_DIR = os.path.join(DATASET_ROOT, "real")
FAKE_DIR = os.path.join(DATASET_ROOT, "fake")
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata.csv")

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTION_HISTORY_DIR = PROJECT_ROOT / "prediction_history"

# Output files
DATASET_SUMMARY_PATH = DATA_DIR / "dataset_summary.json"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
FOLD_METRICS_JSON = RESULTS_DIR / "fold_metrics.json"
TRAINING_HISTORY_CSV = RESULTS_DIR / "training_history.csv"
PREDICTION_HISTORY_CSV = PREDICTION_HISTORY_DIR / "history.csv"

# Label Mapping - consistent across pipeline
CLASS_TO_IDX = {
    "real": 0,
    "fake": 1
}

IDX_TO_CLASS = {
    0: "REAL",
    1: "MORPHED"
}

# Image parameters
IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Normalization constants (ImageNet defaults)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.0001
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 42
NUM_FOLDS = 5
WEIGHT_DECAY = 1e-4

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
NUM_WORKERS = 2 if os.name != 'nt' else 0  # 0 on Windows to prevent multi-processing deadlock

# Ensure directories exist
for p in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, PREDICTION_HISTORY_DIR]:
    p.mkdir(parents=True, exist_ok=True)
