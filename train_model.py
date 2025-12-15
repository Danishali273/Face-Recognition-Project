import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from npwriter import f_name  # Gets the filename 'face_encodings.csv'

def train():
    # 1. Load the dataset
    try:
        df = pd.read_csv(f_name, index_col=0)
    except FileNotFoundError:
        print("Error: No data found. Run manage_faces.py first to collect data.")
        return

    print(f"Dataset loaded. Shape: {df.shape}")

    # 2. Split into Features (X) and Labels (Y)
    # X = All columns except 'name' (the 128 numbers)
    X = df.drop("name", axis=1).values
    # Y = The 'name' column
    y = df["name"].values

    # 3. Train the Model using KNN (K-Nearest Neighbors)
    # KNN works well for face recognition as it finds similar faces
    n_neighbors = min(5, len(X))  # Use k=5 or less if we have fewer samples
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    
    print(f"Training KNN model with k={n_neighbors}...")
    model.fit(X, y)
    
    # 4. Save the trained model
    with open("face_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Model trained and saved as 'face_model.pkl'!")
    print(f"Classes (People) learned: {model.classes_}")

if __name__ == "__main__":
    train()