import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC

DATA_FILE = "face_encodings.csv"

def train():
    # 1. Load the dataset
    try:
        df = pd.read_csv(DATA_FILE, index_col=0)
    except FileNotFoundError:
        print("Error: No data found. Run manage_faces.py first to collect data.")
        return

    print(f"Dataset loaded. Shape: {df.shape}")

    # 2. Split into Features (X) and Labels (Y)
    # X = All columns except 'name' (the 512 ArcFace embedding numbers)
    X = df.drop("name", axis=1).values
    # Y = The 'name' column
    y = df["name"].values

    # 3. Train the Model using SVM (Support Vector Machine)
    # SVM works well for face recognition with high-dimensional embeddings
    # Using RBF kernel which is effective for non-linear classification
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    print("Training SVM model with RBF kernel...")
    model.fit(X, y)
    
    # 4. Save the trained model
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Model trained and saved as 'face_model.pkl'!")
    print(f"Classes (People) learned: {model.classes_}")

if __name__ == "__main__":
    train()