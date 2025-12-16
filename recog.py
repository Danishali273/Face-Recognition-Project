import cv2
import numpy as np
import face_recognition
import pickle
import os

# Recognition threshold - faces with confidence below this are marked as Unknown
RECOGNITION_THRESHOLD = 0.4

# 1. LOAD THE TRAINED ML MODEL
if not os.path.exists("face_model.pkl"):
    print("Error: Model not found! Run train_model.py first.")
    exit()

with open("face_model.pkl", "rb") as f:
    model = pickle.load(f)

print("ML Model loaded successfully!")
print(f"Model can recognize: {model.classes_}")
print(f"Recognition threshold: {RECOGNITION_THRESHOLD}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize for speed (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 2. DETECT FACES
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # 3. EXTRACT FEATURES (The 128 numbers)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    face_confidences = []
    
    for face_encoding in face_encodings:
        # 4. PREDICT USING ML MODEL and calculate confidence
        # Get distance to nearest neighbors
        distances, _ = model.kneighbors([face_encoding])
        
        # Use the minimum distance (best match) for confidence
        min_distance = np.min(distances)
        
        # Face recognition distances typically range from 0.0 (perfect match) to 1.0+ (no match)
        # Distance < 0.4 = excellent match
        # Distance < 0.6 = good match  
        # Distance > 0.6 = poor match
        confidence = max(0, min(1, 1 - (min_distance / 0.6)))
        
        # Only predict name if confidence is above threshold
        if confidence >= RECOGNITION_THRESHOLD:
            name = model.predict([face_encoding])[0]
        else:
            name = "Unknown"
        
        face_names.append(name)
        face_confidences.append(confidence)

    # Display results
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

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
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width + 10, top), color, cv2.FILLED)
        
        # Draw text
        cv2.putText(frame, label, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('ML Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()