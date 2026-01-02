import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

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

    # 3. Train the Model using KNN (K-Nearest Neighbors)
    # KNN works well for face recognition as it finds similar faces
    # Using cosine metric which works better with ArcFace embeddings
    n_neighbors = min(5, len(X))  # Use k=5 or less if we have fewer samples
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
    
    print(f"Training KNN model with k={n_neighbors} (cosine metric for ArcFace)...")
    model.fit(X, y)
    
    # 4. Save the trained model
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Model trained and saved as 'face_model.pkl'!")
    print(f"Classes (People) learned: {model.classes_}")

if __name__ == "__main__":
    train()