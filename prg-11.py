from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Generate curved 2D data
def generate_data(samples=300, noise=0.1):
    X, _ = make_moons(n_samples=samples, noise=noise, random_state=42)
    return X

# Train DBSCAN
def train_model(X, eps=0.3, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

# Visualize clusters and outliers
def visualize(X, model):
    labels = model.labels_
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        mask = labels == k
        if k == -1:
            col = 'black'  # noise
        plt.scatter(X[mask, 0], X[mask, 1], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise')
    
    plt.title("DBSCAN Clustering")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main logic
X = generate_data()
model = train_model(X, eps=0.25, min_samples=5)
print("Cluster labels:", set(model.labels_))
visualize(X, model)
