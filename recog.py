import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis

# Recognition threshold - faces with confidence below this are marked as Unknown
RECOGNITION_THRESHOLD = 0.5

# 1. LOAD THE TRAINED ML MODEL
if not os.path.exists("face_model.pkl"):
    print("Error: Model not found! Run train_model.py first.")
    exit()

with open("face_model.pkl", "rb") as f:
    model = pickle.load(f)

print("ML Model loaded successfully!")
print(f"Model can recognize: {model.classes_}")

# Initialize InsightFace ArcFace model
print("Loading ArcFace model...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("ArcFace model loaded!")

cap = cv2.VideoCapture(0)  # Change to 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces and extract embeddings using ArcFace
    faces = face_app.get(frame)

    face_names = []
    face_confidences = []
    face_boxes = []
    
    for face in faces:
        # Get 512-dimensional ArcFace embedding
        face_encoding = face.embedding
        
        # Get bounding box
        bbox = face.bbox.astype(int)
        face_boxes.append(bbox)
        
        # 4. PREDICT USING ML MODEL and calculate confidence
        # Get distance to nearest neighbors
        distances, _ = model.kneighbors([face_encoding])
        
        # Use the minimum distance (best match) for confidence
        min_distance = np.min(distances)
        
        # ArcFace embeddings use cosine distance, typical range 0-2
        # Distance < 0.8 = excellent match
        # Distance < 1.2 = good match  
        # Distance > 1.2 = poor match
        confidence = max(0, min(1, 1 - (min_distance / 1.2)))
        
        # Only predict name if confidence is above threshold
        if confidence >= RECOGNITION_THRESHOLD:
            name = model.predict([face_encoding])[0]
        else:
            name = "Unknown"
        
        face_names.append(name)
        face_confidences.append(confidence)

    # Display results
    for bbox, name, confidence in zip(face_boxes, face_names, face_confidences):
        left, top, right, bottom = bbox

        # Choose color based on recognition status
        # Red for Unknown, Green for recognized
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        
        # Draw slim box around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
        
        # Add name and confidence above the box
        label = f"{name} ({int(confidence * 100)}%)"
        
        # Calculate text size to create background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        
        # Draw filled rectangle above the box for text background
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width + 10, top), color, -1)
        # Draw text
        cv2.putText(frame, label, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Update statistics (per-frame)
    current_detected = len(face_names)
    current_recognized = sum(1 for name in face_names if name != "Unknown")
    current_unknown = max(current_detected - current_recognized, 0)

    # Draw statistics panel
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 260, 120
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)

    stats_lines = [
        "Current stats:",
        f"Total faces: {current_detected}",
        f"Recognized: {current_recognized}",
        f"Unknown: {current_unknown}",
    ]

    for idx, line in enumerate(stats_lines):
        y_offset = panel_y + 25 + (idx * 20)
        cv2.putText(
            frame,
            line,
            (panel_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    cv2.imshow('ArcFace Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()