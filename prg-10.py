from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load and scale data
def load_scaled_data():
    iris = load_iris()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(iris.data)
    return X_scaled, iris.target, iris.target_names

# Apply PCA to reduce dimensions
def reduce_dimensions(X, n=2):
    return PCA(n_components=n).fit_transform(X)

# Visualize 2D data
def plot_2D(X_pca, y, labels):
    colors = ['red', 'green', 'blue']
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                    label=labels[label], color=colors[i])
    plt.title("PCA – Iris Dataset (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main logic
X, y, label_names = load_scaled_data()
X_pca = reduce_dimensions(X, n=2)
plot_2D(X_pca, y, label_names)
