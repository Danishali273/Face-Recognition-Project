"""
Model Training Script
---------------------
Trains an SVM classifier for face recognition using stored face embeddings.

Loads face embeddings from CSV file, trains an SVM model with RBF kernel,
and saves the trained model for use in recognition.
"""

import logging
import sys

from sklearn.svm import SVC

import config
import utils

# Configure logging
logger = logging.getLogger(__name__)


def train():
    """
    Train the SVM model for face recognition.
    
    Loads face embeddings and labels from CSV, trains an SVM classifier,
    and saves the trained model to a pickle file.
    
    Returns:
        bool: True if training was successful, False otherwise.
        
    Example:
        >>> if train():
        ...     print("Model trained successfully!")
    """
    # Load the dataset
    logger.info("Loading face data from CSV...")
    try:
        X, y = utils.load_face_data()
    except Exception as e:
        error_msg = f"Error loading data: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        return False
    
    if len(X) == 0 or len(y) == 0:
        error_msg = "Error: No data found. Run manage_faces.py first to collect data."
        print(error_msg)
        logger.error(error_msg)
        return False

    print(f"Dataset loaded. Shape: {X.shape}")
    logger.info(f"Dataset loaded: {len(X)} samples, {len(set(y))} classes")

    # Validate we have enough samples
    unique_classes = len(set(y))
    if unique_classes < 1:
        error_msg = "Error: Need at least 1 person in the database."
        print(error_msg)
        logger.error(error_msg)
        return False
    
    min_samples = min([sum(y == cls) for cls in set(y)])
    if min_samples < config.MIN_SAMPLES_FOR_TRAINING:
        warning_msg = (
            f"Warning: Some people have fewer than {config.MIN_SAMPLES_FOR_TRAINING} samples. "
            "Model accuracy may be reduced."
        )
        print(warning_msg)
        logger.warning(warning_msg)

    # Train the Model using SVM (Support Vector Machine)
    # SVM works well for face recognition with high-dimensional embeddings
    # Using RBF kernel which is effective for non-linear classification
    try:
        logger.info("Training SVM model...")
        print("Training SVM model with RBF kernel...")
        
        # Configure gamma parameter
        gamma_value = config.SVM_GAMMA
        if isinstance(gamma_value, str):
            # 'scale' or 'auto' - pass as string
            pass
        else:
            # Numeric value - convert to float
            gamma_value = float(gamma_value)
        
        model = SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=gamma_value,
            probability=config.SVM_PROBABILITY,
            random_state=config.SVM_RANDOM_STATE
        )
        
        model.fit(X, y)
        logger.info("Model training completed successfully")
        
    except Exception as e:
        error_msg = f"Error training model: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        return False

    # Save the trained model
    if utils.save_model(model):
        print("Model trained and saved as 'face_model.pkl'!")
        print(f"Classes (People) learned: {model.classes_}")
        logger.info(f"Model saved. Classes: {model.classes_}")
        return True
    else:
        error_msg = "Failed to save trained model"
        print(f"Error: {error_msg}")
        logger.error(error_msg)
        return False


if __name__ == "__main__":
    try:
        success = train()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        logger.info("Training interrupted by keyboard")
        sys.exit(1)
    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(f"Error: {error_msg}")
        logger.critical(error_msg, exc_info=True)
        sys.exit(1)
