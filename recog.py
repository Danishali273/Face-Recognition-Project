import cv2
import numpy as np
import face_recognition
import pickle
import os

# 1. LOAD THE TRAINED ML MODEL
if not os.path.exists("face_model.pkl"):
    print("Error: Model not found! Run train_model.py first.")
    exit()

with open("face_model.pkl", "rb") as f:
    model = pickle.load(f)

print("ML Model loaded successfully!")
print(f"Model can recognize: {model.classes_}")

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
    
    for face_encoding in face_encodings:
        # Reshape the data to match what the model expects (1 sample, 128 features)
        # 4. PREDICT USING ML MODEL
        # The model returns an array, so we take the first item [0]
        name = model.predict([face_encoding])[0]
        
        # Optional: Get probability/confidence
        # proba = model.predict_proba([face_encoding])[0]
        # max_prob = np.max(proba)
        # if max_prob < 0.6: name = "Unknown" 

        face_names.append(name)

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw Green Box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('ML Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()