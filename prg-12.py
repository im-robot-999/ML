from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate simple 2D cluster data
def generate_data(samples=200, centers=3):
    X, _ = make_blobs(n_samples=samples, centers=centers, random_state=42)
    return X

# Apply Agglomerative Clustering
def train_model(X, k=3):
    return AgglomerativeClustering(n_clusters=k).fit(X)

# Plot dendrogram using linkage
def plot_dendrogram(X):
    linked = linkage(X, method='ward')
    plt.figure(figsize=(8, 5))
    dendrogram(linked)
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

# Visualize final cluster labels
def plot_clusters(X, model):
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis')
    plt.title("Agglomerative Clustering")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

# Main logic
X = generate_data()
model = train_model(X, k=3)
plot_dendrogram(X)
plot_clusters(X, model)
