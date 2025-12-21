# Face Recognition Project

This project is a full-stack face recognition system with a Python backend and a React (Vite + TypeScript) frontend.

## Features
- Face registration and management
- Face recognition
- Model training
- Dashboard and management UI

---

## Backend Setup (Python)

### Prerequisites
- Python 3.8+
- pip (Python package manager)

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
3. **Run the API server:**
   ```sh
   uvicorn api_server:app --reload --port 8000
   ```

---

## Frontend Setup (React + Vite)

### Prerequisites
- Node.js (v16+ recommended)
- npm or yarn

### Installation
1. **Navigate to the frontend directory:**
   ```sh
   cd frontend
   ```
2. **Install dependencies:**
   ```sh
   npm install
   # or
   yarn install
   ```
3. **Start the development server:**
   ```sh
   npm run dev
   # or
   yarn dev
   ```
4. **Open the app:**
   Visit [http://localhost:5173](http://localhost:5173) in your browser.

---

## Project Structure

- `api_server.py`, `manage_faces.py`, `recog.py`, `train_model.py`: Python backend scripts
- `frontend/`: React frontend app
- `requirements.txt`: Python dependencies

---

## Usage
- Register faces, train the model, and recognize faces via the web UI.
- Use the backend scripts for advanced management and training.

---

## Notes
- Ensure the backend server is running before using the frontend.
- Update `frontend/services/faceService.ts` if the backend API URL changes.

---
