from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create synthetic data
def generate_data(samples=300, centers=3):
    X, _ = make_blobs(n_samples=samples, centers=centers, cluster_std=1.0, random_state=42)
    return X

# Train KMeans
def train_model(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model

# Plot clusters
def visualize_clusters(X, model):
    labels = model.labels_
    centers = model.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, label='Points')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main logic
X = generate_data(samples=300, centers=3)
model = train_model(X, k=3)
print("Cluster Centers:\n", model.cluster_centers_)
visualize_clusters(X, model)
