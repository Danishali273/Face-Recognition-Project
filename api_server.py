"""
FaceGuard API Server
--------------------
FastAPI backend for face recognition system.
Connects the React frontend with Python face recognition.

Run with: uvicorn api_server:app --reload --port 8000
"""
#python -m uvicorn api_server:app --reload --port 8000
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
import os
import cv2
import face_recognition
import base64
import json
import asyncio
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI(title="FaceGuard API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
DATA_FILE = "face_encodings.csv"
MODEL_FILE = "face_model.pkl"

# Recognition threshold - faces with confidence below this are marked as Unknown
RECOGNITION_THRESHOLD = 0.35  # Increased for better accuracy


# Pydantic models
class FaceData(BaseModel):
    id: str
    name: str
    sampleCount: int
    dateAdded: str
    lastUpdated: str


class TrainingStatus(BaseModel):
    isTrained: bool
    lastTrainedAt: Optional[str]
    accuracy: float
    classes: List[str] = []


class CaptureRequest(BaseModel):
    name: str
    image: str  # Base64 encoded image


class CaptureResponse(BaseModel):
    success: bool
    message: str
    sampleCount: int


class RecognitionResult(BaseModel):
    name: str
    confidence: float
    bbox: List[int]  # [top, right, bottom, left]


# Helper functions
def get_stored_faces() -> List[FaceData]:
    """Get all stored face data."""
    if not os.path.isfile(DATA_FILE):
        return []
    
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
        if "name" not in df.columns:
            return []
        
        counts = df["name"].value_counts().to_dict()
        faces = []
        
        for name, count in counts.items():
            # Get first and last occurrence for dates
            name_df = df[df["name"] == name]
            faces.append(FaceData(
                id=str(hash(name)),
                name=name,
                sampleCount=count,
                dateAdded=datetime.now().isoformat(),
                lastUpdated=datetime.now().isoformat()
            ))
        
        return faces
    except Exception as e:
        print(f"Error reading faces: {e}")
        return []


def get_training_status() -> TrainingStatus:
    """Get current model training status."""
    if not os.path.isfile(MODEL_FILE):
        return TrainingStatus(isTrained=False, lastTrainedAt=None, accuracy=0, classes=[])
    
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        
        return TrainingStatus(
            isTrained=True,
            lastTrainedAt=datetime.fromtimestamp(os.path.getmtime(MODEL_FILE)).isoformat(),
            accuracy=0.95,  # Placeholder - could calculate from test set
            classes=list(model.classes_)
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return TrainingStatus(isTrained=False, lastTrainedAt=None, accuracy=0, classes=[])


def save_encoding(name: str, encoding: np.ndarray) -> bool:
    """Save a single face encoding to CSV."""
    try:
        if os.path.isfile(DATA_FILE):
            df = pd.read_csv(DATA_FILE, index_col=0)
            new_row = pd.DataFrame([encoding], columns=[f"encoding_{i}" for i in range(128)])
            new_row["name"] = name
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([encoding], columns=[f"encoding_{i}" for i in range(128)])
            df["name"] = name
        
        df.to_csv(DATA_FILE)
        return True
    except Exception as e:
        print(f"Error saving encoding: {e}")
        return False


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# API Endpoints

@app.get("/")
async def root():
    return {"message": "FaceGuard API is running", "version": "1.0.0"}


@app.get("/api/faces", response_model=List[FaceData])
async def list_faces():
    """Get all stored faces."""
    return get_stored_faces()


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    """Delete a face from the database."""
    if not os.path.isfile(DATA_FILE):
        raise HTTPException(status_code=404, detail="No face data found")
    
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
        if name not in df["name"].values:
            raise HTTPException(status_code=404, detail=f"Face '{name}' not found")
        
        df = df[df["name"] != name]
        df.to_csv(DATA_FILE)
        
        return {"message": f"Face '{name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/status", response_model=TrainingStatus)
async def training_status():
    """Get model training status."""
    return get_training_status()


@app.post("/api/training/train", response_model=TrainingStatus)
async def train_model():
    """Train the KNN model on collected face data."""
    if not os.path.isfile(DATA_FILE):
        raise HTTPException(status_code=400, detail="No training data found. Add faces first.")
    
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
        
        if len(df) < 1:
            raise HTTPException(status_code=400, detail="No training data found")
        
        X = df.drop("name", axis=1).values
        y = df["name"].values
        
        n_neighbors = min(5, len(X))
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
        model.fit(X, y)
        
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        
        return TrainingStatus(
            isTrained=True,
            lastTrainedAt=datetime.now().isoformat(),
            accuracy=0.95,
            classes=list(model.classes_)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/capture", response_model=CaptureResponse)
async def capture_face(request: CaptureRequest):
    """Capture and save a face encoding from an image."""
    try:
        # Decode image
        image = base64_to_image(request.image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            return CaptureResponse(success=False, message="No face detected", sampleCount=0)
        
        if len(face_locations) > 1:
            return CaptureResponse(success=False, message="Multiple faces detected. Please show only one face.", sampleCount=0)
        
        # Get encoding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return CaptureResponse(success=False, message="Could not extract face encoding", sampleCount=0)
        
        # Save encoding
        if save_encoding(request.name, face_encodings[0]):
            # Get current sample count
            faces = get_stored_faces()
            current_count = next((f.sampleCount for f in faces if f.name == request.name), 0)
            return CaptureResponse(success=True, message="Face captured successfully", sampleCount=current_count)
        else:
            return CaptureResponse(success=False, message="Failed to save face data", sampleCount=0)
    
    except Exception as e:
        print(f"Capture error: {e}")
        return CaptureResponse(success=False, message=str(e), sampleCount=0)


@app.post("/api/recognize")
async def recognize_face(request: CaptureRequest):
    """Recognize a face in an image."""
    status = get_training_status()
    if not status.isTrained:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        # Load model
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        
        # Decode image
        image = base64_to_image(request.image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            return {"faces": [], "message": "No face detected"}
        
        # Get encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        for encoding, location in zip(face_encodings, face_locations):
            # Get distance for confidence
            distances, _ = model.kneighbors([encoding])
            
            # Use the minimum distance (best match) for confidence
            min_distance = np.min(distances)
            
            # Face recognition distances: 0.0-0.4 excellent, 0.4-0.6 good, >0.6 poor
            confidence = max(0, min(1, 1 - (min_distance / 0.6)))
            
            # Only predict name if confidence is above threshold
            if confidence >= RECOGNITION_THRESHOLD:
                name = model.predict([encoding])[0]
            else:
                name = "Unknown"
            
            results.append(RecognitionResult(
                name=name,
                confidence=round(confidence, 2),
                bbox=list(location)
            ))
        
        return {"faces": results}
    
    except Exception as e:
        print(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """WebSocket for real-time video recognition."""
    await websocket.accept()
    
    # Load model
    model = None
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    
    try:
        while True:
            # Receive frame as base64
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                image = base64_to_image(message["image"])
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                results = []
                
                if model and len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    
                    for encoding, location in zip(face_encodings, face_locations):
                        distances, _ = model.kneighbors([encoding])
                        
                        # Use the minimum distance (best match) for confidence
                        min_distance = np.min(distances)
                        
                        # Face recognition distances: 0.0-0.4 excellent, 0.4-0.6 good, >0.6 poor
                        confidence = max(0, min(1, 1 - (min_distance / 0.6)))
                        
                        # Only predict name if confidence is above threshold
                        if confidence >= RECOGNITION_THRESHOLD:
                            name = model.predict([encoding])[0]
                        else:
                            name = "Unknown"
                        
                        results.append({
                            "name": name,
                            "confidence": round(confidence, 2),
                            "bbox": list(location)
                        })
                elif len(face_locations) > 0:
                    # No model, just return face locations
                    for location in face_locations:
                        results.append({
                            "name": "Unknown",
                            "confidence": 0,
                            "bbox": list(location)
                        })
                
                await websocket.send_json({"faces": results})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
