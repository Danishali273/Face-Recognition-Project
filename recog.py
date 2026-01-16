"""
Face Recognition CLI Tool
--------------------------
Command-line tool for real-time face recognition from webcam, video files, or IP cameras.

Uses ArcFace embeddings and trained SVM model to recognize faces in real-time.
Displays recognition results with confidence scores and statistics.
"""

import cv2
import numpy as np
import logging
import sys
import time

import config
import utils

# Configure logging
logger = logging.getLogger(__name__)


def main():
    """
    Main function for face recognition.
    
    Loads trained model, initializes camera, and performs real-time
    face recognition with statistics display.
    """
    # Load the trained ML model
    logger.info("Loading trained model...")
    model = utils.load_model()
    
    if model is None:
        error_msg = "Error: Model not found! Run train_model.py first."
        print(error_msg)
        logger.error(error_msg)
        sys.exit(1)
    
    print("ML Model loaded successfully!")
    print(f"Model can recognize: {model.classes_}")
    logger.info(f"Model loaded. Can recognize: {model.classes_}")
    
    # Initialize InsightFace ArcFace model
    try:
        print("Loading ArcFace model...")
        logger.info("Initializing ArcFace model...")
        face_app = utils.get_face_app()
        print("ArcFace model loaded!")
    except Exception as e:
        error_msg = f"Failed to initialize ArcFace model: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        sys.exit(1)
    
    # Initialize camera
    try:
        cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera. Check if camera is connected.")
        
        # Test camera
        ret, _ = cap.read()
        if not ret:
            raise RuntimeError("Camera opened but cannot read frames.")
        
        logger.info(f"Camera opened successfully (index: {config.DEFAULT_CAMERA_INDEX})")
    except Exception as e:
        error_msg = f"Failed to open camera: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        sys.exit(1)
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    logger.info("Starting face recognition loop...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                continue

            # FPS calculation
            fps_frame_count += 1
            if fps_frame_count >= 10:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Detect faces and extract embeddings using ArcFace
            try:
                faces = face_app.get(frame)
            except Exception as e:
                logger.warning(f"Error detecting faces: {e}")
                continue

            face_names = []
            face_confidences = []
            face_boxes = []
            
            for face in faces:
                try:
                    # Get 512-dimensional ArcFace embedding
                    face_encoding = face.embedding
                    
                    # Validate embedding
                    if not utils.validate_embedding(face_encoding):
                        logger.warning("Invalid embedding detected, skipping")
                        continue
                    
                    # Get bounding box
                    bbox = face.bbox.astype(int)
                    face_boxes.append(bbox)
                    
                    # Predict using ML model and calculate confidence
                    try:
                        # Prepare encoding for prediction
                        encoding_array = np.array(face_encoding).flatten().reshape(1, -1)
                        
                        # Get prediction probabilities from SVM
                        probabilities = model.predict_proba(encoding_array)[0]
                        
                        # Use the maximum probability as confidence score
                        confidence = np.max(probabilities)
                        
                        # Get the predicted class
                        predicted_class_idx = np.argmax(probabilities)
                        name = model.classes_[predicted_class_idx]
                        
                        # Only keep prediction if confidence is above threshold
                        if confidence < config.RECOGNITION_THRESHOLD:
                            name = "Unknown"
                        
                        face_names.append(name)
                        face_confidences.append(confidence)
                    except Exception as e:
                        logger.error(f"Error during recognition: {e}", exc_info=True)
                        face_names.append("Error")
                        face_confidences.append(0.0)
                        
                except Exception as e:
                    logger.error(f"Error processing face: {e}", exc_info=True)
                    continue

            # Display results
            for bbox, name, confidence in zip(face_boxes, face_names, face_confidences):
                try:
                    left, top, right, bottom = bbox

                    # Choose color based on recognition status
                    # Red for Unknown, Green for recognized
                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                    
                    # Draw slim box around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                    
                    # Add name and confidence above the box
                    label = f"{name} ({int(confidence * 100)}%)"
                    
                    # Calculate text size to create background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
                    )
                    
                    # Draw filled rectangle above the box for text background
                    cv2.rectangle(
                        frame,
                        (left, top - text_height - 10),
                        (left + text_width + 10, top),
                        color,
                        -1
                    )
                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (left + 5, top - 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                except Exception as e:
                    logger.error(f"Error drawing recognition result: {e}", exc_info=True)
                    continue

            # Update statistics (per-frame)
            current_detected = len(face_names)
            current_recognized = sum(1 for name in face_names if name != "Unknown")
            current_unknown = max(current_detected - current_recognized, 0)

            # Draw statistics panel
            try:
                panel_x, panel_y = 10, 10
                panel_w, panel_h = 260, 140
                cv2.rectangle(
                    frame,
                    (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h),
                    (20, 20, 20),
                    -1
                )

                stats_lines = [
                    f"FPS: {current_fps:.1f}",
                    f"Total faces: {current_detected}",
                    f"Recognized: {current_recognized}",
                    f"Unknown: {current_unknown}",
                ]

                for idx, line in enumerate(stats_lines):
                    y_offset = panel_y + 25 + (idx * 25)
                    cv2.putText(
                        frame,
                        line,
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
            except Exception as e:
                logger.error(f"Error drawing statistics: {e}", exc_info=True)

            cv2.imshow('ArcFace Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Recognition stopped by user")
                break
                
    except KeyboardInterrupt:
        print("\nRecognition interrupted by user.")
        logger.info("Recognition interrupted by keyboard")
    except Exception as e:
        error_msg = f"Error in recognition loop: {e}"
        print(f"Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released and windows closed")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
