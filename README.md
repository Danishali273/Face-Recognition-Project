# Face Recognition Project

A modern Python-based face recognition system using ArcFace embeddings and SVM (Support Vector Machine) classification. Includes a desktop GUI (CustomTkinter), video/IP camera support, and flexible face management tools.

---

## Features

- **Face Registration**: Add, remove, and manage faces through GUI or CLI
- **Real-time Recognition**: Live face recognition from webcam, video files, or IP/CCTV cameras
- **Automatic Model Training**: Model automatically retrains after adding new faces (GUI mode)
- **Modern Desktop App**: Beautiful CustomTkinter GUI with real-time video feed
- **Flexible Backend Scripts**: CLI tools for advanced face management
- **Robust Error Handling**: Comprehensive error handling and logging throughout
- **Configuration Management**: Centralized configuration with environment variable support

---

## Architecture

The project follows a modular architecture:

- **Face Detection & Embedding**: ArcFace model (InsightFace) - extracts 512-dimensional embeddings
- **Classification**: SVM classifier (scikit-learn) with RBF kernel
- **Data Storage**: CSV for embeddings, PKL for trained models
- **User Interface**: CustomTkinter desktop app + CLI scripts

### Project Structure

```
Face-Recognition-Project/
├── app.py                 # Main GUI application (CustomTkinter)
├── manage_faces.py        # CLI tool for face management
├── recog.py               # CLI tool for face recognition
├── train_model.py         # Model training script
├── config.py              # Configuration management
├── utils.py               # Common utilities (model loading, CSV operations)
├── visualize_embeddings.py # Embedding visualization tool (optional)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam or IP camera (optional, for real-time recognition)

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Danishali273/Face-Recognition-Project.git
   cd Face-Recognition-Project
   ```

2. **Create a virtual environment** (recommended):
   ```sh
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **First-time setup**:
   - The ArcFace model will be downloaded automatically on first run
   - This may take a few minutes and requires internet connection

---

## Usage

### Method 1: Desktop App (GUI) - Recommended

Run the modern GUI application:

```sh
python app.py
```

**Features:**
- Live camera feed with real-time recognition
- Add new faces with guided capture process
- Automatic model retraining after adding faces
- Remove faces from database
- View stored faces list
- Modern, intuitive interface

**Workflow:**
1. Click "Start Camera" to begin
2. Click "Add New Face" to register a person
3. Press SPACE to capture samples (15 samples recommended)
4. Model automatically retrains after adding faces
5. Recognition happens in real-time

### Method 2: Command Line Interface

#### 1. Register Faces

```sh
python manage_faces.py
```

Interactive menu:
- **Option 1**: Add new face (captures 15 samples by default)
- **Option 2**: Remove face from database
- **Option 3**: List all stored faces with sample counts
- **Option 4**: Exit

#### 2. Train Model

```sh
python train_model.py
```

Trains the SVM classifier using stored face embeddings. Creates `face_model.pkl` for recognition.

**Note**: In GUI mode, this happens automatically after adding faces.

#### 3. Recognize Faces

```sh
python recog.py
```

Performs real-time face recognition from webcam. Press 'q' to quit.

**For video file or IP camera**, modify `recog.py`:
```python
# Line ~28 in recog.py
cap = cv2.VideoCapture(0)  # Webcam
# cap = cv2.VideoCapture('video.mp4')  # Video file
# cap = cv2.VideoCapture('http://IP:PORT/video')  # IP camera
```

---

## Configuration

Configuration is managed through `config.py` and can be overridden with environment variables:

### Key Settings

- **RECOGNITION_THRESHOLD**: Confidence threshold (default: 0.4)
- **CAPTURE_MAX_SAMPLES**: Samples per person (default: 15)
- **ARCFACE_MODEL_NAME**: Model variant (default: 'buffalo_l')
- **SVM_KERNEL**: SVM kernel type (default: 'rbf')

### Environment Variables

```sh
# Set recognition threshold
export RECOGNITION_THRESHOLD=0.4

# Set data file location
export FACE_DATA_FILE=/path/to/face_encodings.csv

# Set log level
export LOG_LEVEL=DEBUG
```

See `config.py` for all available options.

---

## Data Files

- **face_encodings.csv**: Stores face embeddings and person names
- **face_model.pkl**: Trained SVM model for recognition

These files are created automatically and should be backed up if you have important data.

---

## Best Practices

1. **Capture Quality**: 
   - Capture diverse angles (front, left, right)
   - Ensure good lighting
   - Capture 15+ samples per person for best accuracy

2. **Recognition Threshold**:
   - Lower threshold (0.3-0.4): More permissive, may have false positives
   - Higher threshold (0.5-0.6): Stricter, fewer false positives but may miss matches

3. **Model Training**:
   - Retrain after adding/removing faces
   - Ensure at least 3 samples per person for training

4. **Performance**:
   - Uses CPU by default (configurable for GPU)
   - Frame rate depends on hardware
   - Larger detection size = more accurate but slower

---

## Troubleshooting

### Camera not working
- Check if camera is connected and not used by another application
- Try changing `DEFAULT_CAMERA_INDEX` in config.py (0, 1, 2, etc.)

### Model not loading
- Ensure `face_model.pkl` exists (run `train_model.py` first)
- Check file permissions

### Low recognition accuracy
- Capture more diverse samples (different angles, lighting)
- Increase number of samples per person (15-20 recommended)
- Adjust `RECOGNITION_THRESHOLD` in config.py

### ArcFace model download issues
- Ensure internet connection for first-time download
- Model is cached in `~/.insightface/models/` after download

---

## Technical Details

### Face Recognition Pipeline

1. **Face Detection**: ArcFace detects faces in the frame
2. **Embedding Extraction**: 512-dimensional feature vector extracted
3. **Classification**: SVM predicts identity using probability scores
4. **Threshold Check**: Confidence compared against threshold
5. **Display**: Results shown with bounding boxes and labels

### Model Information

- **Embedding Model**: ArcFace (InsightFace) - 512 dimensions
- **Classifier**: SVM with RBF kernel
- **Training**: Supervised learning on stored embeddings
- **Inference**: Real-time prediction on live video

---

## Dependencies

- `opencv-python`: Video processing and camera access
- `numpy`: Numerical operations
- `pandas`: Data management (CSV operations)
- `scikit-learn`: SVM classifier
- `insightface`: ArcFace face recognition model
- `onnxruntime`: Model inference runtime
- `customtkinter`: Modern GUI framework
- `pillow`: Image processing

See `requirements.txt` for specific versions.

---

## License

This project is open source. Feel free to use, modify, and distribute.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Acknowledgments

- **InsightFace**: For the excellent ArcFace implementation
- **CustomTkinter**: For the modern GUI framework
- **scikit-learn**: For the SVM classifier

---

**About**

This project uses deep learning (ArcFace) to extract facial features and an SVM classifier for accurate recognition. It supports real-time face recognition from images, videos, and live streams (including IP cameras), making it suitable for security, attendance, and personalization applications.
