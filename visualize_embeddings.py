"""
Embedding Space Visualization
Visualize the 512-dimensional ArcFace embeddings in 3D space using dimensionality reduction.
Shows how Deep Metric Learning separates different identities into distinct clusters.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def load_embeddings_from_model():
    """Load embeddings and labels from the trained model"""
    if not os.path.exists("face_model.pkl"):
        print("Error: Model not found! Run train_model.py first.")
        exit()
    
    with open("face_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Get training data (embeddings) and labels
    X = model.support_vectors_ if hasattr(model, 'support_vectors_') else None
    
    # If we can't get from model, load from CSV
    if X is None:
        import pandas as pd
        df = pd.read_csv("face_encodings.csv")
        labels = df['name'].values
        X = df.drop('name', axis=1).values
    else:
        # This might not give us all the data, so let's use CSV approach
        import pandas as pd
        df = pd.read_csv("face_encodings.csv")
        labels = df['name'].values
        X = df.drop('name', axis=1).values
    
    return X, labels, model.classes_

def reduce_dimensions_pca(embeddings, n_components=3):
    """Reduce 512D embeddings to 3D using PCA"""
    print(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using PCA...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: {explained_var:.2f}% variance explained")
    return reduced

def reduce_dimensions_tsne(embeddings, n_components=3):
    """Reduce 512D embeddings to 3D using t-SNE (better clustering visualization)"""
    print(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using t-SNE...")
    print("This may take a moment...")
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    print("t-SNE reduction complete!")
    return reduced

def create_3d_visualization(embeddings_3d, labels, class_names, method="PCA"):
    """Create interactive 3D scatter plot of embeddings"""
    
    # Create color mapping for each person
    unique_labels = np.unique(labels)
    colors = []
    
    # Generate distinct colors for each person
    import colorsys
    n_people = len(unique_labels)
    for i in range(n_people):
        hue = i / n_people
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    
    label_to_color = dict(zip(unique_labels, colors))
    
    # Create traces for each person
    traces = []
    for person in unique_labels:
        mask = labels == person
        person_embeddings = embeddings_3d[mask]
        
        trace = go.Scatter3d(
            x=person_embeddings[:, 0],
            y=person_embeddings[:, 1],
            z=person_embeddings[:, 2],
            mode='markers',
            name=person,
            marker=dict(
                size=8,
                color=label_to_color[person],
                opacity=0.8,
                line=dict(color='white', width=0.5)
            ),
            text=[f"{person}<br>Sample {i+1}" for i in range(len(person_embeddings))],
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        )
        traces.append(trace)
    
    # Create layout
    layout = go.Layout(
        title=dict(
            text=f'<b>ArcFace Embedding Space Visualization ({method})</b><br>'
                f'<sub>512D → 3D | Angular Margin Separation | Deep Metric Learning</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title=f'{method} Component 1', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title=f'{method} Component 2', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title=f'{method} Component 3', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            bgcolor="rgb(240, 240, 240)"
        ),
        paper_bgcolor='rgb(250, 250, 250)',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=1400,
        height=900
    )
    
    fig = go.Figure(data=traces, layout=layout)
    
    return fig

def main():
    print("="*60)
    print("ARCFACE EMBEDDING SPACE VISUALIZATION")
    print("="*60)
    
    # Load embeddings
    print("\n1. Loading embeddings from trained model...")
    embeddings, labels, class_names = load_embeddings_from_model()
    print(f"   Loaded {len(embeddings)} embeddings from {len(class_names)} people")
    print(f"   Embedding dimension: {embeddings.shape[1]}D")
    print(f"   People: {', '.join(class_names)}")
    
    # Choose method
    print("\n2. Dimensionality Reduction Method:")
    print("   [1] PCA - Fast, linear (shows global structure)")
    print("   [2] t-SNE - Slower, non-linear (better clustering visualization)")
    
    choice = input("\nChoose method (1 or 2, default=2): ").strip() or "2"
    
    if choice == "1":
        embeddings_3d = reduce_dimensions_pca(embeddings)
        method = "PCA"
    else:
        embeddings_3d = reduce_dimensions_tsne(embeddings)
        method = "t-SNE"
    
    # Create visualization
    print("\n3. Creating 3D visualization...")
    fig = create_3d_visualization(embeddings_3d, labels, class_names, method)
    
    # Save and display
    output_file = f"embedding_space_{method.lower()}.html"
    print(f"\n4. Saving interactive plot to: {output_file}")
    fig.write_html(output_file)
    
    print("\n5. Opening visualization in browser...")
    fig.show()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nInteractive 3D plot saved to: {output_file}")
    print("\nWhat you're seeing:")
    print("  • Each dot = one face embedding (512D vector)")
    print("  • Each color = different person")
    print("  • Clusters = Angular Margin separation by ArcFace")
    print("  • Distance = Cosine similarity in embedding space")
    print("\nYou can rotate, zoom, and hover over points!")
    print("="*60)

if __name__ == "__main__":
    main()
