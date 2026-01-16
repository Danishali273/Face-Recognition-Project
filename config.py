"""
Configuration module for Face Recognition Project.

Centralizes all configuration constants, file paths, and model parameters.
Supports environment variable overrides for flexible deployment.
"""

import os
from pathlib import Path

# ============== File Paths ==============
# Base directory for data files
BASE_DIR = Path(__file__).parent

# Data files
DATA_FILE = os.getenv("FACE_DATA_FILE", str(BASE_DIR / "face_encodings.csv"))
MODEL_FILE = os.getenv("FACE_MODEL_FILE", str(BASE_DIR / "face_model.pkl"))

# ============== Recognition Settings ==============
# Confidence threshold for face recognition (0.0 to 1.0)
# Faces with confidence below this are marked as "Unknown"
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.4"))

# ============== Face Capture Settings ==============
# Number of face samples to capture per person
CAPTURE_MAX_SAMPLES = int(os.getenv("CAPTURE_MAX_SAMPLES", "15"))

# Minimum samples required for training
MIN_SAMPLES_FOR_TRAINING = int(os.getenv("MIN_SAMPLES_FOR_TRAINING", "3"))

# ============== ArcFace Model Settings ==============
# InsightFace model name ('buffalo_l' is recommended for accuracy)
ARCFACE_MODEL_NAME = os.getenv("ARCFACE_MODEL_NAME", "buffalo_l")

# Detection size for ArcFace (larger = more accurate but slower)
ARCFACE_DET_SIZE = tuple(map(int, os.getenv("ARCFACE_DET_SIZE", "640,640").split(",")))

# Execution provider for ONNX runtime
# Options: 'CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'
ARCFACE_PROVIDERS = [os.getenv("ARCFACE_PROVIDER", "CPUExecutionProvider")]

# ============== SVM Model Settings ==============
# SVM kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
SVM_KERNEL = os.getenv("SVM_KERNEL", "rbf")

# SVM regularization parameter (higher = stricter margin)
SVM_C = float(os.getenv("SVM_C", "1.0"))

# SVM gamma parameter ('scale', 'auto', or float value)
SVM_GAMMA = os.getenv("SVM_GAMMA", "scale")

# Enable probability estimates (required for confidence scores)
SVM_PROBABILITY = True

# Random state for reproducibility
SVM_RANDOM_STATE = int(os.getenv("SVM_RANDOM_STATE", "42"))

# ============== Application Settings ==============
# Window dimensions
APP_WIDTH = int(os.getenv("APP_WIDTH", "1200"))
APP_HEIGHT = int(os.getenv("APP_HEIGHT", "750"))
APP_MIN_WIDTH = int(os.getenv("APP_MIN_WIDTH", "1000"))
APP_MIN_HEIGHT = int(os.getenv("APP_MIN_HEIGHT", "650"))

# CustomTkinter appearance
APP_APPEARANCE_MODE = os.getenv("APP_APPEARANCE_MODE", "dark")  # 'light', 'dark', 'system'
APP_COLOR_THEME = os.getenv("APP_COLOR_THEME", "blue")  # 'blue', 'green', 'dark-blue'

# ============== Camera Settings ==============
# Default camera index (0 for first webcam)
DEFAULT_CAMERA_INDEX = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))

# Video frame processing settings
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "0"))  # Skip every N frames for performance

# ============== Logging Settings ==============
# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Log file path (None for console only)
LOG_FILE = os.getenv("LOG_FILE", None)

# ============== Validation ==============
def validate_config():
    """
    Validate configuration values.
    
    Raises:
        ValueError: If any configuration value is invalid.
    """
    if not 0.0 <= RECOGNITION_THRESHOLD <= 1.0:
        raise ValueError(f"RECOGNITION_THRESHOLD must be between 0.0 and 1.0, got {RECOGNITION_THRESHOLD}")
    
    if CAPTURE_MAX_SAMPLES < 1:
        raise ValueError(f"CAPTURE_MAX_SAMPLES must be at least 1, got {CAPTURE_MAX_SAMPLES}")
    
    if MIN_SAMPLES_FOR_TRAINING < 1:
        raise ValueError(f"MIN_SAMPLES_FOR_TRAINING must be at least 1, got {MIN_SAMPLES_FOR_TRAINING}")
    
    if SVM_KERNEL not in ['rbf', 'linear', 'poly', 'sigmoid']:
        raise ValueError(f"SVM_KERNEL must be one of ['rbf', 'linear', 'poly', 'sigmoid'], got {SVM_KERNEL}")
    
    if APP_APPEARANCE_MODE not in ['light', 'dark', 'system']:
        raise ValueError(f"APP_APPEARANCE_MODE must be one of ['light', 'dark', 'system'], got {APP_APPEARANCE_MODE}")


# Validate on import
validate_config()
