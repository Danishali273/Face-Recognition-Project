
"""
Face Recognition Desktop App
-----------------------------
Modern GUI application using CustomTkinter for face recognition
with ArcFace embeddings and SVM (Support Vector Machine) classification.
"""


import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import pickle
import os
import threading
from tkinter import messagebox, simpledialog
from insightface.app import FaceAnalysis
from sklearn.svm import SVC

# ============== Configuration ==============
DATA_FILE = "face_encodings.csv"
MODEL_FILE = "face_model.pkl"
RECOGNITION_THRESHOLD = 0.4

# CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("🎯 Face Recognition System")
        self.geometry("1100x700")
        self.minsize(900, 600)
        
        # State variables
        self.cap = None
        self.is_running = False
        self.is_capturing = False
        self.capture_name = ""
        self.capture_count = 0
        self.capture_max = 15
        self.encodings_list = []
        self.model = None
        self.face_app = None
        
        # Create UI
        self.create_widgets()
        
        # Load model and ArcFace in background
        self.after(100, self.initialize_models)

    # (If you have any KNeighborsClassifier usage, replace with SVC)
    
    def create_widgets(self):
        """Create all UI widgets."""
        # Main container
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ============== Left Sidebar ==============
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="🎯 Face Recognition",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Status: Initializing...",
            font=ctk.CTkFont(size=12),
            text_color="yellow"
        )
        self.status_label.grid(row=1, column=0, padx=20, pady=5)
        
        # Separator
        self.sep1 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        self.sep1.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        
        # Camera Controls
        self.cam_label = ctk.CTkLabel(
            self.sidebar,
            text="📷 Camera",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.cam_label.grid(row=3, column=0, padx=20, pady=(10, 5))
        
        self.start_btn = ctk.CTkButton(
            self.sidebar,
            text="▶ Start Camera",
            command=self.start_camera,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_btn.grid(row=4, column=0, padx=20, pady=5)
        
        self.stop_btn = ctk.CTkButton(
            self.sidebar,
            text="⏹ Stop Camera",
            command=self.stop_camera,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.grid(row=5, column=0, padx=20, pady=5)
        
        # Separator
        self.sep2 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        self.sep2.grid(row=6, column=0, sticky="ew", padx=20, pady=10)
        
        # Face Management
        self.manage_label = ctk.CTkLabel(
            self.sidebar,
            text="👤 Manage Faces",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.manage_label.grid(row=7, column=0, padx=20, pady=(10, 5))
        
        self.add_face_btn = ctk.CTkButton(
            self.sidebar,
            text="➕ Add New Face",
            command=self.add_face,
            state="disabled"
        )
        self.add_face_btn.grid(row=8, column=0, padx=20, pady=5)
        
        self.train_btn = ctk.CTkButton(
            self.sidebar,
            text="🎓 Train Model",
            command=self.train_model,
            fg_color="purple",
            hover_color="darkviolet"
        )
        self.train_btn.grid(row=9, column=0, padx=20, pady=5)
        
        # Spacer
        self.spacer = ctk.CTkLabel(self.sidebar, text="")
        self.spacer.grid(row=10, column=0)
        
        # Stored Faces List
        self.faces_label = ctk.CTkLabel(
            self.sidebar,
            text="📋 Stored Faces",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.faces_label.grid(row=11, column=0, padx=20, pady=(10, 5))
        
        self.faces_listbox = ctk.CTkTextbox(
            self.sidebar,
            width=200,
            height=150,
            font=ctk.CTkFont(size=12)
        )
        self.faces_listbox.grid(row=12, column=0, padx=20, pady=5)
        
        self.refresh_btn = ctk.CTkButton(
            self.sidebar,
            text="🔄 Refresh List",
            command=self.refresh_faces_list,
            width=100
        )
        self.refresh_btn.grid(row=13, column=0, padx=20, pady=5)
        
        self.remove_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑 Remove Face",
            command=self.remove_face,
            fg_color="gray40",
            hover_color="gray30"
        )
        self.remove_btn.grid(row=14, column=0, padx=20, pady=(5, 20))
        
        # ============== Main Content ==============
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Video display
        self.video_label = ctk.CTkLabel(
            self.main_frame,
            text="📷 Camera Feed\n\nClick 'Start Camera' to begin",
            font=ctk.CTkFont(size=18),
            fg_color="gray20",
            corner_radius=10
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Info bar at bottom
        self.info_frame = ctk.CTkFrame(self.main_frame, height=50)
        self.info_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="Ready | Press 'Q' to quit recognition mode",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.pack(pady=10)
        
        # Capture progress (hidden by default)
        self.capture_progress = ctk.CTkProgressBar(self.info_frame, width=300)
        self.capture_progress.set(0)
        
        self.capture_label = ctk.CTkLabel(
            self.info_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
    
    def initialize_models(self):
        """Initialize ArcFace and load trained model."""
        self.update_status("Loading ArcFace model...", "yellow")
        
        def load():
            try:
                # Load ArcFace
                self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                
                # Load trained model if exists
                if os.path.exists(MODEL_FILE):
                    with open(MODEL_FILE, "rb") as f:
                        self.model = pickle.load(f)
                
                self.after(0, lambda: self.update_status("Ready", "green"))
                self.after(0, lambda: self.add_face_btn.configure(state="normal"))
                self.after(0, self.refresh_faces_list)
                
            except Exception as e:
                self.after(0, lambda: self.update_status(f"Error: {str(e)}", "red"))
        
        threading.Thread(target=load, daemon=True).start()
    
    def update_status(self, text, color="white"):
        """Update status label."""
        self.status_label.configure(text=f"Status: {text}", text_color=color)
    
    def start_camera(self):
        """Start the webcam feed."""
        if self.face_app is None:
            messagebox.showwarning("Wait", "Models are still loading. Please wait.")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.update_status("Camera running", "green")
        
        self.update_frame()
    
    def stop_camera(self):
        """Stop the webcam feed."""
        self.is_running = False
        self.is_capturing = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.add_face_btn.configure(state="normal", text="➕ Add New Face")
        
        self.video_label.configure(
            image=None,
            text="📷 Camera Feed\n\nClick 'Start Camera' to begin"
        )
        self.update_status("Camera stopped", "yellow")
        
        # Hide capture progress
        self.capture_progress.pack_forget()
        self.capture_label.pack_forget()
        self.info_label.pack(pady=10)
    
    def update_frame(self):
        """Update video frame."""
        if not self.is_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_frame)
            return
        
        # Detect and process faces
        faces = self.face_app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox
            
            if self.is_capturing:
                # Capture mode - just show green box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Recognition mode
                if self.model is not None:
                    face_encoding = face.embedding
                    distances, _ = self.model.kneighbors([face_encoding])
                    min_distance = np.min(distances)
                    confidence = max(0, min(1, 1 - (min_distance / 1.2)))
                    
                    if confidence >= RECOGNITION_THRESHOLD:
                        name = self.model.predict([face_encoding])[0]
                    else:
                        name = "Unknown"
                    
                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    label = f"{name} ({int(confidence * 100)}%)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                    cv2.rectangle(frame, (left, top - th - 10), (left + tw + 10, top), color, -1)
                    cv2.putText(frame, label, (left + 5, top - 5),
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                    cv2.putText(frame, "No model trained", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Convert to PIL and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit display
        display_width = self.video_label.winfo_width() - 40
        display_height = self.video_label.winfo_height() - 40
        if display_width > 100 and display_height > 100:
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk, text="")
        self.video_label.image = imgtk
        
        self.after(30, self.update_frame)
    
    def add_face(self):
        """Start capturing face data for a new person."""
        if not self.is_running:
            messagebox.showinfo("Info", "Please start the camera first.")
            return
        
        name = simpledialog.askstring("Add Face", "Enter person's name:")
        if not name or not name.strip():
            return
        
        self.capture_name = name.strip()
        self.capture_count = 0
        self.capture_max = 15
        self.encodings_list = []
        self.is_capturing = True
        
        # Update UI
        self.add_face_btn.configure(state="disabled", text="📸 Capturing...")
        self.info_label.pack_forget()
        self.capture_progress.pack(pady=5)
        self.capture_progress.set(0)
        self.capture_label.pack(pady=5)
        self.capture_label.configure(text=f"Capturing: {self.capture_name} (0/{self.capture_max})")
        
        self.update_status(f"Capturing faces for '{self.capture_name}'", "cyan")
        
        # Bind space key for capture
        self.bind("<space>", self.capture_face)
        self.bind("<Escape>", self.cancel_capture)
        
        messagebox.showinfo(
            "Capture Mode",
            f"Capturing face data for '{self.capture_name}'\n\n"
            "• Press SPACE to capture (15 samples needed)\n"
            "• Try different angles for better accuracy\n"
            "• Press ESC to cancel"
        )
    
    def capture_face(self, event=None):
        """Capture a single face sample."""
        if not self.is_capturing or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        faces = self.face_app.get(frame)
        
        if len(faces) > 0:
            encoding = faces[0].embedding
            self.encodings_list.append(encoding)
            self.capture_count += 1
            
            # Update progress
            progress = self.capture_count / self.capture_max
            self.capture_progress.set(progress)
            self.capture_label.configure(
                text=f"Capturing: {self.capture_name} ({self.capture_count}/{self.capture_max})"
            )
            
            if self.capture_count >= self.capture_max:
                self.finish_capture()
        else:
            self.update_status("No face detected - try again", "orange")
    
    def finish_capture(self):
        """Finish capturing and save face data."""
        self.is_capturing = False
        self.unbind("<space>")
        self.unbind("<Escape>")
        
        # Save encodings
        if len(self.encodings_list) > 0:
            self.save_encodings(self.capture_name, np.array(self.encodings_list))
            messagebox.showinfo(
                "Success",
                f"Saved {len(self.encodings_list)} samples for '{self.capture_name}'!\n\n"
                "Click 'Train Model' to update recognition."
            )
        
        # Reset UI
        self.add_face_btn.configure(state="normal", text="➕ Add New Face")
        self.capture_progress.pack_forget()
        self.capture_label.pack_forget()
        self.info_label.pack(pady=10)
        self.update_status("Ready", "green")
        self.refresh_faces_list()
    
    def cancel_capture(self, event=None):
        """Cancel face capture."""
        self.is_capturing = False
        self.unbind("<space>")
        self.unbind("<Escape>")
        
        self.add_face_btn.configure(state="normal", text="➕ Add New Face")
        self.capture_progress.pack_forget()
        self.capture_label.pack_forget()
        self.info_label.pack(pady=10)
        self.update_status("Capture cancelled", "yellow")
    
    def save_encodings(self, name, encodings):
        """Save face encodings to CSV file."""
        if encodings is None or len(encodings) == 0:
            return False
        
        if os.path.isfile(DATA_FILE):
            df = pd.read_csv(DATA_FILE, index_col=0)
            latest = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(512)])
            latest["name"] = name
            df = pd.concat((df, latest), ignore_index=True, sort=False)
        else:
            df = pd.DataFrame(encodings, columns=[f"encoding_{i}" for i in range(512)])
            df["name"] = name
        
        df.to_csv(DATA_FILE)
        return True
    
    def train_model(self):
        """Train the KNN model."""
        if not os.path.isfile(DATA_FILE):
            messagebox.showwarning("No Data", "No face data found. Add faces first!")
            return
        
        self.update_status("Training model...", "yellow")
        self.train_btn.configure(state="disabled")
        
        def train():
            try:
                df = pd.read_csv(DATA_FILE, index_col=0)
                X = df.drop("name", axis=1).values
                y = df["name"].values
                
                n_neighbors = min(5, len(X))
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
                model.fit(X, y)
                
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)
                
                self.model = model
                
                self.after(0, lambda: self.update_status("Model trained!", "green"))
                self.after(0, lambda: self.train_btn.configure(state="normal"))
                self.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Model trained successfully!\n\nCan recognize: {list(model.classes_)}"
                ))
                
            except Exception as e:
                self.after(0, lambda: self.update_status(f"Training failed: {str(e)}", "red"))
                self.after(0, lambda: self.train_btn.configure(state="normal"))
        
        threading.Thread(target=train, daemon=True).start()
    
    def refresh_faces_list(self):
        """Refresh the stored faces list."""
        self.faces_listbox.delete("0.0", "end")
        
        if os.path.isfile(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE, index_col=0)
                counts = df["name"].value_counts().to_dict()
                
                if counts:
                    for name, count in sorted(counts.items()):
                        self.faces_listbox.insert("end", f"• {name}: {count} samples\n")
                else:
                    self.faces_listbox.insert("end", "No faces stored")
            except:
                self.faces_listbox.insert("end", "No faces stored")
        else:
            self.faces_listbox.insert("end", "No faces stored")
    
    def remove_face(self):
        """Remove a face from the database."""
        if not os.path.isfile(DATA_FILE):
            messagebox.showinfo("Info", "No faces stored.")
            return
        
        try:
            df = pd.read_csv(DATA_FILE, index_col=0)
            names = df["name"].unique().tolist()
        except:
            messagebox.showerror("Error", "Could not read data file.")
            return
        
        if not names:
            messagebox.showinfo("Info", "No faces stored.")
            return
        
        name = simpledialog.askstring(
            "Remove Face",
            f"Enter name to remove:\n\nStored: {', '.join(names)}"
        )
        
        if not name or name not in names:
            if name:
                messagebox.showwarning("Not Found", f"'{name}' not found in database.")
            return
        
        confirm = messagebox.askyesno("Confirm", f"Remove all data for '{name}'?")
        if not confirm:
            return
        
        df = df[df["name"] != name]
        df.to_csv(DATA_FILE)
        
        messagebox.showinfo("Success", f"Removed '{name}'. Remember to retrain the model!")
        self.refresh_faces_list()
    
    def on_closing(self):
        """Handle window close."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
