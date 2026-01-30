"""
Face Management CLI Tool
-------------------------
Command-line interface for managing face database.

Provides functionality to:
- Add new faces to the database
- Remove faces from the database
- List all stored faces with sample counts
- Capture face data using webcam
"""

import cv2
import numpy as np
import logging
import sys

import config
import utils

# Configure logging
logger = logging.getLogger(__name__)


def capture_face_data(name: str, max_samples: int = None) -> bool:
    """
    Capture face data for a person using webcam.
    
    Opens camera, displays live feed, and allows user to capture
    face samples by pressing 'c'. Validates embeddings before saving.
    
    Args:
        name: Person's name to associate with captured data
        max_samples: Maximum number of samples to capture.
        If None, uses config.CAPTURE_MAX_SAMPLES
    
    Returns:
        bool: True if capture was successful, False otherwise.
        
    Example:
        >>> if capture_face_data("John Doe", 15):
        ...     print("Successfully captured face data")
    """
    if max_samples is None:
        max_samples = config.CAPTURE_MAX_SAMPLES
    
    if not name or not name.strip():
        print("Error: Name cannot be empty!")
        logger.error("Empty name provided for face capture")
        return False
    
    name = name.strip()
    
    try:
        cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
        if not cap.isOpened():
            error_msg = "Error: Could not open camera!"
            print(error_msg)
            logger.error(error_msg)
            return False
        
        # Test camera by reading one frame
        ret, _ = cap.read()
        if not ret:
            error_msg = "Error: Camera opened but cannot read frames."
            print(error_msg)
            logger.error(error_msg)
            cap.release()
            return False
        
        # Get the face analysis model
        try:
            face_app = utils.get_face_app()
        except Exception as e:
            error_msg = f"Failed to initialize ArcFace model: {e}"
            print(error_msg)
            logger.error(error_msg, exc_info=True)
            cap.release()
            return False
        
        encodings_list = []
        
        print(f"\n{'='*60}")
        print(f"Capturing face data for '{name}'")
        print(f"{'='*60}")
        print(f"* Position your face in front of the camera")
        print(f"* Press 'c' to capture face ({max_samples} samples needed)")
        print(f"* Try different angles: front, left, right for better accuracy")
        print(f"* Press 'q' to quit/cancel")
        print(f"{'='*60}\n")
        
        logger.info(f"Starting face capture for '{name}' ({max_samples} samples)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Detect faces using InsightFace
                try:
                    faces = face_app.get(frame)
                except Exception as e:
                    logger.warning(f"Error detecting faces: {e}")
                    continue
                
                # Draw green boxes around detected faces
                for face in faces:
                    try:
                        bbox = face.bbox.astype(int)
                        left, top, right, bottom = bbox
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Face Detected", (left + 6, top - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        logger.warning(f"Error drawing face box: {e}")
                        continue

                # Show progress
                progress_text = f"Samples: {len(encodings_list)}/{max_samples}"
                cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(faces) > 0:
                    cv2.putText(frame, "Face Detected - Press 'c' to capture", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face Detected - Position Yourself", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Face Capture (ArcFace)", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nCapture cancelled by user.")
                    logger.info("Face capture cancelled by user")
                    break
                elif key == ord('c'):
                    if len(faces) > 0:
                        try:
                            # Get face embedding from InsightFace (512-dimensional)
                            face_encoding = faces[0].embedding
                            
                            # Validate embedding
                            if not utils.validate_embedding(face_encoding):
                                print("[X] Invalid face encoding - Try again")
                                logger.warning("Invalid embedding detected during capture")
                                continue
                            
                            encodings_list.append(face_encoding)
                            print(f"[OK] Captured sample {len(encodings_list)}/{max_samples}")
                            logger.debug(f"Captured sample {len(encodings_list)}/{max_samples} for '{name}'")
                            
                            if len(encodings_list) >= max_samples:
                                print(f"\n[OK] Collected {max_samples} samples successfully!")
                                break
                        except Exception as e:
                            error_msg = f"Error capturing face: {e}"
                            print(f"[X] {error_msg}")
                            logger.error(error_msg, exc_info=True)
                    else:
                        print("[X] No face detected - Try again")
        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user.")
            logger.info("Face capture interrupted by keyboard")
        except Exception as e:
            error_msg = f"Unexpected error during capture: {e}"
            print(f"[X] {error_msg}")
            logger.error(error_msg, exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Save face data
        if len(encodings_list) > 0:
            try:
                encodings_array = np.array(encodings_list)
                success = utils.save_face_encodings(name, encodings_array)
                if success:
                    print(f"\n[OK] Face data for '{name}' saved successfully!")
                    logger.info(f"Successfully saved {len(encodings_list)} samples for '{name}'")
                    return True
                else:
                    print(f"\n[X] Failed to save face data for '{name}'.")
                    logger.error(f"Failed to save face data for '{name}'")
                    return False
            except Exception as e:
                error_msg = f"Error saving face data: {e}"
                print(f"\n[X] {error_msg}")
                logger.error(error_msg, exc_info=True)
                return False
        else:
            print("\n[X] No face data captured.")
            logger.warning("No face data captured")
            return False
            
    except Exception as e:
        error_msg = f"Fatal error in capture_face_data: {e}"
        print(f"[X] {error_msg}")
        logger.critical(error_msg, exc_info=True)
        return False


def add_face():
    """
    Add a new face to the database.
    
    Prompts user for person's name, checks if name exists,
    asks for number of samples, and initiates capture process.
    """
    print("\n" + "="*60)
    print("ADD NEW FACE")
    print("="*60)
    
    try:
        name = input("Enter the person's name: ").strip()
        if not name:
            print("[X] Invalid name. Operation cancelled.")
            logger.warning("Empty name provided in add_face")
            return
        
        # Check if name already exists
        stored_names = utils.get_stored_names()
        if name in stored_names:
            print(f"\n[!] Warning: '{name}' already exists in the database.")
            choice = input("Do you want to add more samples? (y/n): ").lower()
            if choice != 'y':
                print("Operation cancelled.")
                return
        
        # Ask for number of samples
        try:
            samples_input = input(f"Number of samples to capture (default: {config.CAPTURE_MAX_SAMPLES}): ").strip()
            samples = int(samples_input) if samples_input else config.CAPTURE_MAX_SAMPLES
            if samples < 1 or samples > 50:
                print(f"[!] Invalid number. Using default ({config.CAPTURE_MAX_SAMPLES}).")
                samples = config.CAPTURE_MAX_SAMPLES
        except ValueError:
            print(f"[!] Invalid input. Using default ({config.CAPTURE_MAX_SAMPLES}).")
            samples = config.CAPTURE_MAX_SAMPLES
        
        capture_face_data(name, samples)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        logger.info("add_face cancelled by keyboard")
    except Exception as e:
        error_msg = f"Error in add_face: {e}"
        print(f"[X] {error_msg}")
        logger.error(error_msg, exc_info=True)


def remove_face():
    """
    Remove a person from the database.
    
    Lists all stored faces, prompts user to select one,
    confirms deletion, and removes all data for that person.
    """
    print("\n" + "="*60)
    print("REMOVE FACE")
    print("="*60)
    
    try:
        stored_names = utils.get_stored_names()
        if not stored_names:
            print("[X] No faces stored in the database.")
            return
        
        print("\nStored faces:")
        for i, name in enumerate(stored_names, 1):
            print(f"  {i}. {name}")
        
        name = input("\nEnter the name to remove (or number): ").strip()
        
        # Check if user entered a number
        try:
            idx = int(name) - 1
            if 0 <= idx < len(stored_names):
                name = stored_names[idx]
        except ValueError:
            pass
        
        if name not in stored_names:
            print(f"[X] '{name}' not found in database.")
            return
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete '{name}'? (y/n): ").lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
        
        # Remove from database
        if utils.remove_face_from_db(name):
            print(f"[OK] '{name}' removed from database.")
            logger.info(f"Removed '{name}' from database")
            print("[!] Remember to retrain the model to apply changes.")
        else:
            raise RuntimeError("Failed to remove face from database")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        logger.info("remove_face cancelled by keyboard")
    except Exception as e:
        error_msg = f"Error removing face: {e}"
        print(f"[X] {error_msg}")
        logger.error(error_msg, exc_info=True)


def list_faces():
    """
    List all stored faces with sample counts.
    
    Displays a formatted table showing person names and
    their sample counts, plus total statistics.
    """
    print("\n" + "="*60)
    print("STORED FACES")
    print("="*60)
    
    try:
        sample_counts = utils.get_sample_counts()
        
        if not sample_counts:
            print("\n[X] No faces stored in the database.")
            return
        
        print(f"\n{'Name':<20} {'Samples':<10}")
        print("-" * 30)
        
        total_samples = 0
        for name, count in sorted(sample_counts.items()):
            print(f"{name:<20} {count:<10}")
            total_samples += count
        
        print("-" * 30)
        print(f"{'Total':<20} {total_samples:<10}")
        print(f"\nTotal persons: {len(sample_counts)}")
        logger.debug(f"Listed {len(sample_counts)} people with {total_samples} total samples")
        
    except Exception as e:
        error_msg = f"Error listing faces: {e}"
        print(f"[X] {error_msg}")
        logger.error(error_msg, exc_info=True)


def main_menu():
    """
    Display main menu and handle user input.
    
    Provides interactive menu for face management operations.
    Runs until user chooses to exit.
    """
    print("\n" + "="*60)
    print("FACE RECOGNITION - MANAGEMENT SYSTEM (SVM Model)")
    print("="*60)
    logger.info("Face management system started")
    
    try:
        while True:
            print("\n" + "="*60)
            print("FACE RECOGNITION - MANAGEMENT SYSTEM (SVM Model)")
            print("="*60)
            print("1. Add new face")
            print("2. Remove face")
            print("3. List all faces")
            print("4. Exit")
            print("-"*60)
            
            try:
                choice = input("Enter choice (1-4): ").strip()
                
                if choice == "1":
                    add_face()
                elif choice == "2":
                    remove_face()
                elif choice == "3":
                    list_faces()
                elif choice == "4":
                    print("\nGoodbye!")
                    logger.info("Face management system exited")
                    break
                else:
                    print("[X] Invalid choice. Please enter 1-4.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                logger.info("Face management system exited by keyboard")
                break
            except Exception as e:
                error_msg = f"Error processing menu choice: {e}"
                print(f"[X] {error_msg}")
                logger.error(error_msg, exc_info=True)
                
    except Exception as e:
        error_msg = f"Fatal error in main_menu: {e}"
        print(f"[X] {error_msg}")
        logger.critical(error_msg, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
