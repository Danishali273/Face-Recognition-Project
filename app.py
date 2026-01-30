"""
Face Recognition Desktop App
-----------------------------
Modern GUI application using CustomTkinter for face recognition
with ArcFace embeddings and SVM (Support Vector Machine) classification.

This module provides a complete desktop interface for:
- Real-time face recognition from webcam
- Face registration and management
- Automatic model training
- Face database management
"""

import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
import threading
import logging
from tkinter import messagebox, simpledialog
from typing import Optional

import config
import utils

# Configure logging
logger = logging.getLogger(__name__)

# CustomTkinter appearance
ctk.set_appearance_mode(config.APP_APPEARANCE_MODE)
ctk.set_default_color_theme(config.APP_COLOR_THEME)


class FaceRecognitionApp(ctk.CTk):
    """
    Main application class for Face Recognition System.
    
    Provides a modern GUI interface for face recognition using ArcFace embeddings
    and SVM classification. Supports real-time recognition, face registration,
    and automatic model training.
    
    Attributes:
        cap: VideoCapture object for camera access
        is_running: Boolean indicating if camera is active
        is_capturing: Boolean indicating if face capture is in progress
        capture_name: Name of person being captured
        capture_count: Number of samples captured so far
        capture_max: Maximum number of samples to capture
        encodings_list: List of face embeddings being collected
        model: Trained SVM model for recognition
        face_app: ArcFace model instance for face detection/embedding
    """
    
    def __init__(self):
        """
        Initialize the Face Recognition application.
        
        Sets up the window, UI widgets, and initializes models in background.
        """
        super().__init__()
        
        # Window setup
        self.title("🎯 Face Recognition System")
        self.geometry(f"{config.APP_WIDTH}x{config.APP_HEIGHT}")
        self.minsize(config.APP_MIN_WIDTH, config.APP_MIN_HEIGHT)
        
        # State variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_capturing = False
        self.capture_name = ""
        self.capture_count = 0
        self.capture_max = config.CAPTURE_MAX_SAMPLES
        self.encodings_list = []
        self.model = None
        self.face_app = None
        
        # Create UI
        self.create_widgets()
        
        # Load model and ArcFace in background
        self.after(100, self.initialize_models)
    
    def create_widgets(self):
        """
        Create and configure all UI widgets.
        
        Sets up the sidebar with controls and the main content area
        for video display.
        """
        # Main container
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ============== Left Sidebar ==============
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=("#f0f0f0", "#1a1a1a"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(9, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        # Logo/Title with enhanced styling
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=(25, 15), sticky="ew")
        
        self.logo_label = ctk.CTkLabel(
            title_frame, 
            text="Face Recognition System",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="AI-Powered Recognition System",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Status label with badge style
        status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        status_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Status: Initializing...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="yellow",
            corner_radius=15,
            fg_color=("#fff3cd", "#3d3d00"),
            padx=15,
            pady=8
        )
        self.status_label.pack()
        
        # Separator
        self.sep1 = ctk.CTkFrame(self.sidebar, height=1, fg_color="gray30")
        self.sep1.grid(row=2, column=0, sticky="ew", padx=25, pady=15)
        
        # Camera Controls
        self.cam_label = ctk.CTkLabel(
            self.sidebar,
            text="📷 Camera Controls",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.cam_label.grid(row=3, column=0, padx=20, pady=(5, 10))
        
        self.start_btn = ctk.CTkButton(
            self.sidebar,
            text="▶ Start Camera",
            command=self.start_camera,
            fg_color="#4caf50",
            hover_color="#45a049",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8
        )
        self.start_btn.grid(row=4, column=0, padx=20, pady=6, sticky="ew")
        
        self.stop_btn = ctk.CTkButton(
            self.sidebar,
            text="⏹ Stop Camera",
            command=self.stop_camera,
            fg_color="#f44336",
            hover_color="#da190b",
            state="disabled",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8
        )
        self.stop_btn.grid(row=5, column=0, padx=20, pady=6, sticky="ew")
        
        # Separator
        self.sep2 = ctk.CTkFrame(self.sidebar, height=1, fg_color="gray30")
        self.sep2.grid(row=6, column=0, sticky="ew", padx=25, pady=15)
        
        # Face Management
        self.manage_label = ctk.CTkLabel(
            self.sidebar,
            text="👤 Face Management",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.manage_label.grid(row=7, column=0, padx=20, pady=(5, 10))
        
        self.add_face_btn = ctk.CTkButton(
            self.sidebar,
            text="➕ Add New Face",
            command=self.add_face,
            state="disabled",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            fg_color="#2196F3",
            hover_color="#1976D2"
        )
        self.add_face_btn.grid(row=8, column=0, padx=20, pady=8, sticky="ew")
        
        # Spacer
        self.spacer = ctk.CTkLabel(self.sidebar, text="")
        self.spacer.grid(row=9, column=0)
        
        # Stored Faces List
        self.faces_label = ctk.CTkLabel(
            self.sidebar,
            text="📋 Stored Faces",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.faces_label.grid(row=10, column=0, padx=20, pady=(10, 5))
        
        self.faces_listbox = ctk.CTkTextbox(
            self.sidebar,
            width=200,
            height=150,
            font=ctk.CTkFont(size=12),
            corner_radius=8,
            border_width=2,
            border_color="gray40"
        )
        self.faces_listbox.grid(row=11, column=0, padx=20, pady=5)
        
        self.refresh_btn = ctk.CTkButton(
            self.sidebar,
            text="🔄 Refresh List",
            command=self.refresh_faces_list,
            width=120,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.refresh_btn.grid(row=12, column=0, padx=20, pady=5)
        
        self.remove_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑 Remove Face",
            command=self.remove_face,
            fg_color="#d32f2f",
            hover_color="#b71c1c",
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.remove_btn.grid(row=13, column=0, padx=20, pady=(5, 20))
        
        # ============== Main Content ==============
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=("#ffffff", "#1e1e1e"))
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Video display container with border
        video_container = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=("#f5f5f5", "#2b2b2b"))
        video_container.grid(row=0, column=0, sticky="nsew", padx=25, pady=25)
        video_container.grid_columnconfigure(0, weight=1)
        video_container.grid_rowconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(
            video_container,
            text="📷 Camera Feed\n\nClick 'Start Camera' to begin",
            font=ctk.CTkFont(size=20),
            fg_color=("#e0e0e0", "#252525"),
            corner_radius=10,
            text_color=("#666666", "#aaaaaa")
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        # Info bar at bottom with enhanced styling
        self.info_frame = ctk.CTkFrame(self.main_frame, height=60, corner_radius=10, fg_color=("#f8f8f8", "#2a2a2a"))
        self.info_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=(0, 25))
        self.info_frame.grid_columnconfigure(0, weight=1)
        
        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="Ready | Press 'Q' to quit recognition mode",
            font=ctk.CTkFont(size=13),
            text_color=("#333333", "#cccccc")
        )
        self.info_label.pack(pady=15)
        
        # Capture progress (hidden by default)
        self.capture_progress = ctk.CTkProgressBar(
            self.info_frame, 
            width=400,
            height=20,
            corner_radius=10,
            progress_color="#4caf50"
        )
        self.capture_progress.set(0)
        
        self.capture_label = ctk.CTkLabel(
            self.info_frame,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#333333", "#cccccc")
        )
    
    def initialize_models(self):
        """
        Initialize ArcFace model and load trained SVM model.
        
        Runs in a background thread to avoid blocking the UI.
        Updates status and enables face addition button when complete.
        """
        self.update_status("Loading ArcFace model...", "yellow")
        logger.info("Initializing models...")
        
        def load():
            try:
                # Load ArcFace using utility function
                self.face_app = utils.get_face_app()
                logger.info("ArcFace model loaded successfully")
                
                # Load trained model if exists
                self.model = utils.load_model()
                if self.model is None:
                    logger.info("No trained model found. Add faces and train to enable recognition.")
                else:
                    logger.info(f"Model loaded. Can recognize: {self.model.classes_}")
                
                self.after(0, lambda: self.update_status("Ready", "green"))
                self.after(0, lambda: self.add_face_btn.configure(state="normal"))
                self.after(0, self.refresh_faces_list)
                
            except Exception as e:
                error_msg = f"Failed to initialize models: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.after(0, lambda: self.update_status(f"Error: {str(e)}", "red"))
                self.after(0, lambda: messagebox.showerror(
                    "Initialization Error",
                    f"Failed to initialize models:\n{str(e)}\n\nPlease check your installation."
                ))
        
        threading.Thread(target=load, daemon=True).start()
    
    def update_status(self, text: str, color: str = "white"):
        """
        Update the status label with enhanced styling.
        
        Args:
            text: Status message to display
            color: Color theme for the status badge
                   Options: 'green', 'yellow', 'red', 'cyan', 'orange', 'white'
        """
        color_map = {
            "green": ("#2e7d32", "#81c784"),
            "yellow": ("#f57f17", "#ffc107"),
            "red": ("#c62828", "#e57373"),
            "cyan": ("#00838f", "#4dd0e1"),
            "orange": ("#e65100", "#ffb74d"),
            "white": ("#666666", "#ffffff")
        }
        
        bg_color, text_color = color_map.get(color, ("#666666", "#ffffff"))
        self.status_label.configure(
            text=f"Status: {text}", 
            text_color=text_color,
            fg_color=bg_color
        )
        logger.debug(f"Status updated: {text} ({color})")
    
    def start_camera(self):
        """
        Start the webcam feed for face recognition.
        
        Opens the camera, enables recognition mode, and starts frame updates.
        Shows error message if camera cannot be opened or models aren't ready.
        """
        if self.face_app is None:
            messagebox.showwarning("Wait", "Models are still loading. Please wait.")
            logger.warning("Camera start attempted before models loaded")
            return
        
        try:
            self.cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera. Check if camera is connected.")
            
            # Test camera by reading one frame
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Camera opened but cannot read frames.")
            
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_status("Camera running", "green")
            logger.info("Camera started successfully")
            
            self.update_frame()
        except Exception as e:
            error_msg = f"Failed to start camera: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Camera Error", error_msg)
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def stop_camera(self):
        """
        Stop the webcam feed and clean up resources.
        
        Releases the camera, resets UI state, and hides capture progress.
        """
        logger.info("Stopping camera...")
        self.is_running = False
        self.is_capturing = False
        
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
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
        """
        Update video frame with face detection and recognition.
        
        Processes each frame, detects faces, performs recognition if model
        is available, and displays results. Handles both capture and recognition modes.
        """
        if not self.is_running or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.after(10, self.update_frame)
                return
            
            # Detect and process faces
            faces = self.face_app.get(frame)
            
            for face in faces:
                try:
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
                            try:
                                # Validate and prepare embedding
                                if not utils.validate_embedding(face.embedding):
                                    logger.warning("Invalid embedding detected, skipping")
                                    continue
                                
                                face_encoding = np.array(face.embedding).flatten().reshape(1, -1)
                                
                                # Use SVC's predict_proba to get confidence scores
                                probabilities = self.model.predict_proba(face_encoding)[0]
                                max_prob = np.max(probabilities)
                                predicted_class_idx = np.argmax(probabilities)
                                name = self.model.classes_[predicted_class_idx]
                                confidence = float(max_prob)
                                
                                if confidence < config.RECOGNITION_THRESHOLD:
                                    name = "Unknown"
                                    confidence = 1 - confidence  # Show uncertainty for unknown
                                
                                color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                
                                label = f"{name} ({int(confidence * 100)}%)"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                                cv2.rectangle(frame, (left, top - th - 10), (left + tw + 10, top), color, -1)
                                cv2.putText(frame, label, (left + 5, top - 5),
                                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                            except Exception as e:
                                logger.error(f"Error during recognition: {e}", exc_info=True)
                                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                                cv2.putText(frame, "Recognition error", (left, top - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                            cv2.putText(frame, "No model trained", (left, top - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Error processing face: {e}", exc_info=True)
                    continue
            
            # Convert to PIL and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit display
            display_width = self.video_label.winfo_width() - 40
            display_height = self.video_label.winfo_height() - 40
            if display_width > 100 and display_height > 100:
                img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Use CTkImage for better HighDPI support
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img
            
        except Exception as e:
            logger.error(f"Error in update_frame: {e}", exc_info=True)
            self.update_status(f"Frame error: {str(e)}", "red")
        
        self.after(30, self.update_frame)
    
    def add_face(self):
        """
        Start capturing face data for a new person.
        
        Prompts for person's name and enters capture mode. User presses
        SPACE to capture samples. ESC cancels the operation.
        """
        if not self.is_running:
            messagebox.showinfo("Info", "Please start the camera first.")
            return
        
        name = simpledialog.askstring("Add Face", "Enter person's name:")
        if not name or not name.strip():
            return
        
        self.capture_name = name.strip()
        self.capture_count = 0
        self.capture_max = config.CAPTURE_MAX_SAMPLES
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
        logger.info(f"Starting face capture for '{self.capture_name}'")
        
        # Bind space key for capture
        self.bind("<space>", self.capture_face)
        self.bind("<Escape>", self.cancel_capture)
        
        messagebox.showinfo(
            "Capture Mode",
            f"Capturing face data for '{self.capture_name}'\n\n"
            f"• Press SPACE to capture ({self.capture_max} samples needed)\n"
            "• Try different angles for better accuracy\n"
            "• Press ESC to cancel"
        )
    
    def capture_face(self, event=None):
        """
        Capture a single face sample.
        
        Called when user presses SPACE during capture mode.
        Validates face detection and adds embedding to collection.
        
        Args:
            event: Tkinter event (optional)
        """
        if not self.is_capturing or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame during capture")
                return
            
            faces = self.face_app.get(frame)
            
            if len(faces) > 0:
                encoding = faces[0].embedding
                
                # Validate embedding before adding
                if not utils.validate_embedding(encoding):
                    self.update_status("Invalid face - try again", "orange")
                    logger.warning("Invalid embedding detected during capture")
                    return
                
                self.encodings_list.append(encoding)
                self.capture_count += 1
                
                # Update progress
                progress = self.capture_count / self.capture_max
                self.capture_progress.set(progress)
                self.capture_label.configure(
                    text=f"Capturing: {self.capture_name} ({self.capture_count}/{self.capture_max})"
                )
                
                logger.debug(f"Captured sample {self.capture_count}/{self.capture_max} for '{self.capture_name}'")
                
                if self.capture_count >= self.capture_max:
                    self.finish_capture()
            else:
                self.update_status("No face detected - try again", "orange")
        except Exception as e:
            logger.error(f"Error during face capture: {e}", exc_info=True)
            self.update_status(f"Capture error: {str(e)}", "red")
    
    def finish_capture(self):
        """
        Finish capturing and save face data.
        
        Saves collected encodings to database, triggers automatic model
        retraining if sufficient samples exist, and resets UI state.
        """
        self.is_capturing = False
        self.unbind("<space>")
        self.unbind("<Escape>")
        
        # Save encodings
        if len(self.encodings_list) > 0:
            try:
                encodings_array = np.array(self.encodings_list)
                success = utils.save_face_encodings(self.capture_name, encodings_array)
                
                if success:
                    logger.info(f"Successfully saved {len(self.encodings_list)} samples for '{self.capture_name}'")
                    messagebox.showinfo(
                        "Success",
                        f"Saved {len(self.encodings_list)} samples for '{self.capture_name}'!\n\n"
                        "Face data has been saved successfully."
                    )
                    
                    # Automatic model retraining
                    self.auto_retrain_model()
                else:
                    raise RuntimeError("Failed to save encodings")
            except Exception as e:
                error_msg = f"Failed to save face data: {str(e)}"
                logger.error(error_msg, exc_info=True)
                messagebox.showerror("Error", error_msg)
        
        # Reset UI
        self.add_face_btn.configure(state="normal", text="➕ Add New Face")
        self.capture_progress.pack_forget()
        self.capture_label.pack_forget()
        self.info_label.pack(pady=10)
        self.update_status("Ready", "green")
        self.refresh_faces_list()
    
    def auto_retrain_model(self):
        """
        Automatically retrain the model after adding new faces.
        
        Checks if sufficient samples exist for training, then trains
        the model in a background thread. Updates UI when complete.
        """
        try:
            # Check if we have enough samples
            sample_counts = utils.get_sample_counts()
            total_samples = sum(sample_counts.values())
            unique_people = len(sample_counts)
            
            if total_samples < config.MIN_SAMPLES_FOR_TRAINING:
                logger.info(f"Insufficient samples for training ({total_samples} < {config.MIN_SAMPLES_FOR_TRAINING})")
                return
            
            if unique_people < 1:
                logger.warning("No people in database, cannot train model")
                return
            
            logger.info(f"Auto-retraining model with {total_samples} samples from {unique_people} people...")
            self.update_status("Auto-training model...", "yellow")
            
            def train():
                try:
                    X, y = utils.load_face_data()
                    
                    if len(X) == 0 or len(y) == 0:
                        logger.warning("No data available for training")
                        self.after(0, lambda: self.update_status("No data to train", "orange"))
                        return
                    
                    # Train SVM model
                    from sklearn.svm import SVC
                    model = SVC(
                        kernel=config.SVM_KERNEL,
                        C=config.SVM_C,
                        gamma=config.SVM_GAMMA if isinstance(config.SVM_GAMMA, str) else float(config.SVM_GAMMA),
                        probability=config.SVM_PROBABILITY,
                        random_state=config.SVM_RANDOM_STATE
                    )
                    
                    logger.info("Training SVM model...")
                    model.fit(X, y)
                    
                    # Save model
                    if utils.save_model(model):
                        self.model = model
                        logger.info(f"Model auto-trained successfully. Classes: {model.classes_}")
                        self.after(0, lambda: self.update_status("Model auto-trained!", "green"))
                    else:
                        raise RuntimeError("Failed to save trained model")
                        
                except Exception as e:
                    error_msg = f"Auto-training failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    self.after(0, lambda: self.update_status("Auto-training failed", "red"))
            
            threading.Thread(target=train, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error in auto_retrain_model: {e}", exc_info=True)
            self.update_status("Auto-train error", "orange")
    
    def cancel_capture(self, event=None):
        """
        Cancel face capture operation.
        
        Resets capture state and UI without saving any data.
        
        Args:
            event: Tkinter event (optional)
        """
        logger.info("Face capture cancelled by user")
        self.is_capturing = False
        self.unbind("<space>")
        self.unbind("<Escape>")
        
        self.add_face_btn.configure(state="normal", text="➕ Add New Face")
        self.capture_progress.pack_forget()
        self.capture_label.pack_forget()
        self.info_label.pack(pady=10)
        self.update_status("Capture cancelled", "yellow")
    
    def refresh_faces_list(self):
        """
        Refresh the stored faces list in the sidebar.
        
        Loads current face data from database and displays
        person names with sample counts.
        """
        self.faces_listbox.delete("0.0", "end")
        
        try:
            sample_counts = utils.get_sample_counts()
            
            if sample_counts:
                for name, count in sorted(sample_counts.items()):
                    self.faces_listbox.insert("end", f"• {name}: {count} samples\n")
            else:
                self.faces_listbox.insert("end", "No faces stored")
        except Exception as e:
            logger.error(f"Error refreshing faces list: {e}", exc_info=True)
            self.faces_listbox.insert("end", "Error loading faces")
    
    def remove_face(self):
        """
        Remove a face from the database.
        
        Prompts user to select a person to remove, confirms deletion,
        and updates the database. Does not automatically retrain model.
        """
        try:
            stored_names = utils.get_stored_names()
            
            if not stored_names:
                messagebox.showinfo("Info", "No faces stored.")
                return
            
            name = simpledialog.askstring(
                "Remove Face",
                f"Enter name to remove:\n\nStored: {', '.join(stored_names)}"
            )
            
            if not name or name not in stored_names:
                if name:
                    messagebox.showwarning("Not Found", f"'{name}' not found in database.")
                return
            
            confirm = messagebox.askyesno("Confirm", f"Remove all data for '{name}'?")
            if not confirm:
                return
            
            if utils.remove_face_from_db(name):
                logger.info(f"Removed '{name}' from database")
                messagebox.showinfo("Success", f"Removed '{name}' from the database.")
                self.refresh_faces_list()
            else:
                raise RuntimeError("Failed to remove face from database")
                
        except Exception as e:
            error_msg = f"Error removing face: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Error", error_msg)
    
    def on_closing(self):
        """
        Handle window close event.
        
        Stops camera, releases resources, and closes the application.
        """
        logger.info("Application closing...")
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera on close: {e}")
        self.destroy()


if __name__ == "__main__":
    try:
        app = FaceRecognitionApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        logger.critical(f"Fatal error starting application: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Failed to start application:\n{str(e)}")
