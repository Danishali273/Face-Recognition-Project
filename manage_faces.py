"""
Face Data Management System
----------------------------
Complete management interface for face recognition database.

Features:
  1. Add new face - Capture face samples for new person
  2. Remove face - Delete person from database
  3. List faces - View all stored persons with sample counts
  4. Train model - Train KNN model on collected data
"""

import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier

# File where data will be stored
DATA_FILE = "face_encodings.csv"
MODEL_FILE = "face_model.pkl"


def get_stored_names():
    """Get list of all stored person names."""
    if os.path.isfile(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, index_col=0)
            return df["name"].unique().tolist()
        except:
            return []
    return []


def get_sample_counts():
    """Get sample count for each person."""
    if os.path.isfile(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, index_col=0)
            return df["name"].value_counts().to_dict()
        except:
            return {}
    return {}


def save_encodings(name, encodings):
    """Save face encodings to CSV file."""
    if encodings is None or len(encodings) == 0:
        return False
    
    if os.path.isfile(DATA_FILE):
        df = pd.read_csv(DATA_FILE, index_col=0)
        latest = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(128)])
        latest["name"] = name
        df = pd.concat((df, latest), ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(128)])
        df["name"] = name

    df.to_csv(DATA_FILE)
    return True


def capture_face_data(name, max_samples=15):
    """Capture face data for a person using webcam."""
    if not name.strip():
        print("Error: Name cannot be empty!")
        return False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera!")
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

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        # Draw green boxes around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Face Detected", (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show progress
        progress_text = f"Samples: {len(encodings_list)}/{max_samples}"
        cv2.putText(frame, progress_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(face_locations) > 0:
            cv2.putText(frame, "Face Detected - Press 'c' to capture", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected - Position Yourself", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            print("\nCapture cancelled by user.")
            break
        elif key & 0xFF == ord('c'):
            if len(face_locations) > 0:
                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if len(face_encodings) > 0:
                    encodings_list.append(face_encodings[0])
                    print(f"[OK] Captured sample {len(encodings_list)}/{max_samples}")
                    
                    if len(encodings_list) >= max_samples:
                        print(f"\n[OK] Collected {max_samples} samples successfully!")
                        break
                else:
                    print("[X] Could not extract face encoding - Try again")
            else:
                print("[X] No face detected - Try again")

    cap.release()
    cv2.destroyAllWindows()

    # Save face data
    if len(encodings_list) > 0:
        success = save_encodings(name, np.array(encodings_list))
        if success:
            print(f"\n[OK] Face data for '{name}' saved successfully!")
            return True
        else:
            print(f"\n[X] Failed to save face data for '{name}'.")
            return False
    else:
        print("\n[X] No face data captured.")
        return False


def add_face():
    """Add a new face to the database."""
    print("\n" + "="*60)
    print("ADD NEW FACE")
    print("="*60)
    
    name = input("Enter the person's name: ").strip()
    if not name:
        print("[X] Invalid name. Operation cancelled.")
        return
    
    # Check if name already exists
    stored_names = get_stored_names()
    if name in stored_names:
        print(f"\n[!] Warning: '{name}' already exists in the database.")
        choice = input("Do you want to add more samples? (y/n): ").lower()
        if choice != 'y':
            print("Operation cancelled.")
            return
    
    # Ask for number of samples
    try:
        samples = int(input(f"Number of samples to capture (default: 15): ") or "15")
        if samples < 1 or samples > 50:
            print("[!] Invalid number. Using default (15).")
            samples = 15
    except ValueError:
        print("[!] Invalid input. Using default (15).")
        samples = 15
    
    capture_face_data(name, samples)


def remove_face():
    """Remove a person from the database."""
    print("\n" + "="*60)
    print("REMOVE FACE")
    print("="*60)
    
    stored_names = get_stored_names()
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
    
    # Remove from CSV
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
        df = df[df["name"] != name]
        df.to_csv(DATA_FILE)
        print(f"[OK] '{name}' removed from database.")
        
        # Remind to retrain model
        print("[!] Remember to train the model again (option 4) to apply changes.")
    except Exception as e:
        print(f"[X] Error removing face: {e}")


def list_faces():
    """List all stored faces with sample counts."""
    print("\n" + "="*60)
    print("STORED FACES")
    print("="*60)
    
    sample_counts = get_sample_counts()
    
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


def train_model():
    """Train KNN model on collected face data."""
    print("\n" + "="*60)
    print("TRAIN MODEL (KNN)")
    print("="*60)
    
    if not os.path.isfile(DATA_FILE):
        print("[X] No data found. Add faces first using option 1.")
        return
    
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
    except Exception as e:
        print(f"[X] Error loading data: {e}")
        return
    
    if len(df) < 1:
        print("[X] No data found. Add faces first.")
        return
    
    # Split into features and labels
    X = df.drop("name", axis=1).values
    y = df["name"].values
    
    unique_names = np.unique(y)
    print(f"\nDataset loaded: {len(X)} samples, {len(unique_names)} persons")
    print(f"Persons: {list(unique_names)}")
    
    # Determine k for KNN (use smaller of 5 or number of samples)
    n_neighbors = min(5, len(X))
    
    # Train KNN model
    print(f"\nTraining KNN model with k={n_neighbors}...")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    model.fit(X, y)
    
    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n[OK] Model trained and saved as '{MODEL_FILE}'!")
    print(f"[OK] Model can recognize: {list(model.classes_)}")


def main_menu():
    """Display main menu and handle user input."""
    while True:
        print("\n" + "="*60)
        print("FACE RECOGNITION - MANAGEMENT SYSTEM")
        print("="*60)
        print("1. Add new face")
        print("2. Remove face")
        print("3. List all faces")
        print("4. Train model (KNN)")
        print("5. Exit")
        print("-"*60)
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            add_face()
        elif choice == "2":
            remove_face()
        elif choice == "3":
            list_faces()
        elif choice == "4":
            train_model()
        elif choice == "5":
            print("\nGoodbye!")
            break
        else:
            print("[X] Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main_menu()