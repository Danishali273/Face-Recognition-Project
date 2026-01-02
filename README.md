# Face Recognition Project

A modern Python-based face recognition system using ArcFace embeddings and KNN classification. Includes a desktop GUI (CustomTkinter), video/IP camera support, and flexible face management tools.

---

## Features
- Face registration and management (add, remove, list faces)
- Face recognition (live webcam, video file, or IP/CCTV camera)
- Model training (KNN classifier with ArcFace embeddings)
- Real-time statistics (faces detected, recognized, unknown)
- Modern desktop app (CustomTkinter)
- Flexible backend scripts for advanced management

---

## Setup (Python)

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Webcam or IP camera (optional)

### Installation
1. **Clone the repository** (if not already):
   ```sh
   git clone <https://github.com/Danishali273/Face-Recognition-Project>
   cd Face-Recognition-Project
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Register Faces
- Run `python manage_faces.py` to add, remove, or list faces.
- Capture 15+ samples per person for best results.

### 2. Train Model
- Run `python train_model.py` to train the KNN classifier.
- This creates `face_model.pkl` for recognition.

### 3. Recognize Faces
- Run `python recog.py` for live webcam/video/IP camera recognition.
- To use a video file or IP camera, change this line in `recog.py`:
  ```python
  cap = cv2.VideoCapture(0)  # Webcam
  # cap = cv2.VideoCapture('my_video.mp4')  # Video file
  # cap = cv2.VideoCapture('http://<IP>:<PORT>/video')  # IP camera
  ```
- Real-time stats are shown on the video feed.

### 4. Desktop App (GUI)
- Run `python app.py` for a modern GUI (CustomTkinter) with:
  - Live camera feed
  - Add/train/remove faces
  - Recognition with stats

---

## Project Structure

- `app.py`: CustomTkinter desktop app
- `manage_faces.py`, `recog.py`, `train_model.py`: Backend scripts
- `requirements.txt`: Python dependencies

---

## Notes
- For best accuracy, capture diverse samples (angles, lighting).
- Use a threshold of 0.5 for recognition; tune as needed.
- Supports webcam, video files, and IP/CCTV cameras.
- All data is stored locally in CSV/model files.

---
