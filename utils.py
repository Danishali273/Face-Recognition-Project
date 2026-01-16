"""
Utility functions for Face Recognition Project.

Provides common operations for:
- ArcFace model initialization and management
- Model loading and saving
- CSV data operations
- Face embedding validation
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from insightface.app import FaceAnalysis
from sklearn.svm import SVC

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global ArcFace model instance (initialized lazily)
_face_app: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    """
    Get or initialize the ArcFace model instance.
    
    Uses singleton pattern to ensure only one instance is created.
    The model is initialized lazily on first access.
    
    Returns:
        FaceAnalysis: Initialized ArcFace model instance.
        
    Raises:
        RuntimeError: If model initialization fails.
    """
    global _face_app
    
    if _face_app is None:
        try:
            logger.info(f"Initializing ArcFace model '{config.ARCFACE_MODEL_NAME}'...")
            _face_app = FaceAnalysis(
                name=config.ARCFACE_MODEL_NAME,
                providers=config.ARCFACE_PROVIDERS
            )
            _face_app.prepare(ctx_id=0, det_size=config.ARCFACE_DET_SIZE)
            logger.info("ArcFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace model: {e}")
            raise RuntimeError(f"ArcFace model initialization failed: {e}") from e
    
    return _face_app


def load_model(model_path: Optional[str] = None) -> Optional[SVC]:
    """
    Load a trained SVM model from a pickle file.
    
    Args:
        model_path: Path to the model file. If None, uses config.MODEL_FILE.
        
    Returns:
        SVC: Loaded SVM model, or None if file doesn't exist or loading fails.
        
    Example:
        >>> model = load_model()
        >>> if model is not None:
        ...     print(f"Model can recognize: {model.classes_}")
    """
    if model_path is None:
        model_path = config.MODEL_FILE
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        if not isinstance(model, SVC):
            logger.warning(f"Loaded model is not an SVC instance: {type(model)}")
            return None
        
        logger.info(f"Model loaded successfully. Classes: {model.classes_}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def save_model(model: SVC, model_path: Optional[str] = None) -> bool:
    """
    Save a trained SVM model to a pickle file.
    
    Args:
        model: Trained SVC model to save.
        model_path: Path to save the model. If None, uses config.MODEL_FILE.
        
    Returns:
        bool: True if save was successful, False otherwise.
        
    Example:
        >>> model = SVC()
        >>> model.fit(X, y)
        >>> save_model(model)
    """
    if model_path is None:
        model_path = config.MODEL_FILE
    
    try:
        logger.info(f"Saving model to {model_path}...")
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved successfully to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False


def load_face_data(data_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load face embeddings and labels from CSV file.
    
    Args:
        data_file: Path to the CSV file. If None, uses config.DATA_FILE.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (embeddings, labels) arrays.
            Returns empty arrays if file doesn't exist or is invalid.
            
    Example:
        >>> X, y = load_face_data()
        >>> print(f"Loaded {len(X)} embeddings for {len(np.unique(y))} people")
    """
    if data_file is None:
        data_file = config.DATA_FILE
    
    if not os.path.exists(data_file):
        logger.warning(f"Data file not found: {data_file}")
        return np.array([]), np.array([])
    
    try:
        logger.debug(f"Loading face data from {data_file}...")
        df = pd.read_csv(data_file, index_col=0)
        
        if "name" not in df.columns:
            logger.error("CSV file missing 'name' column")
            return np.array([]), np.array([])
        
        # Extract embeddings (all columns except 'name')
        X = df.drop("name", axis=1).values
        y = df["name"].values
        
        logger.info(f"Loaded {len(X)} embeddings for {len(np.unique(y))} people")
        return X, y
    except Exception as e:
        logger.error(f"Failed to load face data from {data_file}: {e}")
        return np.array([]), np.array([])


def save_face_encodings(name: str, encodings: np.ndarray, data_file: Optional[str] = None) -> bool:
    """
    Save face encodings to CSV file.
    
    Args:
        name: Person's name (label for the encodings).
        encodings: Array of face embeddings (shape: [n_samples, 512]).
        data_file: Path to the CSV file. If None, uses config.DATA_FILE.
        
    Returns:
        bool: True if save was successful, False otherwise.
        
    Raises:
        ValueError: If encodings are invalid or name is empty.
        
    Example:
        >>> encodings = np.array([[0.1, 0.2, ...], ...])  # 512-dim embeddings
        >>> save_face_encodings("John Doe", encodings)
    """
    if data_file is None:
        data_file = config.DATA_FILE
    
    # Validation
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")
    
    if encodings is None or len(encodings) == 0:
        raise ValueError("Encodings cannot be empty")
    
    if encodings.ndim != 2 or encodings.shape[1] != 512:
        raise ValueError(f"Encodings must be 2D array with 512 features, got shape {encodings.shape}")
    
    try:
        # Create DataFrame with embeddings
        embedding_cols = [f"encoding_{i}" for i in range(512)]
        new_data = pd.DataFrame(encodings, columns=embedding_cols)
        new_data["name"] = name.strip()
        
        # Append to existing file or create new
        if os.path.exists(data_file):
            logger.debug(f"Appending to existing data file: {data_file}")
            df = pd.read_csv(data_file, index_col=0)
            df = pd.concat([df, new_data], ignore_index=True, sort=False)
        else:
            logger.debug(f"Creating new data file: {data_file}")
            df = new_data
        
        df.to_csv(data_file)
        logger.info(f"Saved {len(encodings)} encodings for '{name}' to {data_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save face encodings: {e}")
        return False


def get_stored_names(data_file: Optional[str] = None) -> List[str]:
    """
    Get list of all stored person names from the database.
    
    Args:
        data_file: Path to the CSV file. If None, uses config.DATA_FILE.
        
    Returns:
        List[str]: List of unique person names, empty list if file doesn't exist.
        
    Example:
        >>> names = get_stored_names()
        >>> print(f"Stored people: {', '.join(names)}")
    """
    if data_file is None:
        data_file = config.DATA_FILE
    
    if not os.path.exists(data_file):
        return []
    
    try:
        df = pd.read_csv(data_file, index_col=0)
        if "name" not in df.columns:
            return []
        return df["name"].unique().tolist()
    except Exception as e:
        logger.error(f"Failed to get stored names: {e}")
        return []


def get_sample_counts(data_file: Optional[str] = None) -> Dict[str, int]:
    """
    Get sample count for each person in the database.
    
    Args:
        data_file: Path to the CSV file. If None, uses config.DATA_FILE.
        
    Returns:
        Dict[str, int]: Dictionary mapping person names to sample counts.
        
    Example:
        >>> counts = get_sample_counts()
        >>> for name, count in counts.items():
        ...     print(f"{name}: {count} samples")
    """
    if data_file is None:
        data_file = config.DATA_FILE
    
    if not os.path.exists(data_file):
        return {}
    
    try:
        df = pd.read_csv(data_file, index_col=0)
        if "name" not in df.columns:
            return {}
        return df["name"].value_counts().to_dict()
    except Exception as e:
        logger.error(f"Failed to get sample counts: {e}")
        return {}


def remove_face_from_db(name: str, data_file: Optional[str] = None) -> bool:
    """
    Remove all data for a person from the database.
    
    Args:
        name: Name of the person to remove.
        data_file: Path to the CSV file. If None, uses config.DATA_FILE.
        
    Returns:
        bool: True if removal was successful, False otherwise.
        
    Example:
        >>> if remove_face_from_db("John Doe"):
        ...     print("Successfully removed John Doe")
    """
    if data_file is None:
        data_file = config.DATA_FILE
    
    if not os.path.exists(data_file):
        logger.warning(f"Data file not found: {data_file}")
        return False
    
    try:
        df = pd.read_csv(data_file, index_col=0)
        if "name" not in df.columns:
            logger.error("CSV file missing 'name' column")
            return False
        
        initial_count = len(df)
        df = df[df["name"] != name]
        removed_count = initial_count - len(df)
        
        if removed_count == 0:
            logger.warning(f"No data found for '{name}'")
            return False
        
        df.to_csv(data_file)
        logger.info(f"Removed {removed_count} samples for '{name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to remove face from database: {e}")
        return False


def validate_embedding(embedding: np.ndarray) -> bool:
    """
    Validate that an embedding has the correct shape and values.
    
    Args:
        embedding: Face embedding array to validate.
        
    Returns:
        bool: True if embedding is valid, False otherwise.
        
    Example:
        >>> if validate_embedding(face.embedding):
        ...     print("Embedding is valid")
    """
    if embedding is None:
        return False
    
    if not isinstance(embedding, np.ndarray):
        return False
    
    # Check shape: should be 1D with 512 elements
    if embedding.ndim == 1:
        if embedding.shape[0] != 512:
            return False
    elif embedding.ndim == 2:
        if embedding.shape != (1, 512):
            return False
        embedding = embedding.flatten()
    else:
        return False
    
    # Check for NaN or Inf values
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False
    
    return True
