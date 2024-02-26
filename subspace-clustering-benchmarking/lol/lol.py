import numpy as np
from scipy.linalg import eigh

def normalized_cut(A):
    # A is the affinity matrix (pairwise weight matrix)

    # Step 1: Create the degree matrix D
    D = np.diag(np.sum(A, axis=1))

    # Step 2: Compute the unnormalized Laplacian matrix L
    L = D - A

    # Step 3: Compute the first k eigenvectors of L
    # where k is the number of clusters you want
    _, eigenvectors = eigh(L, eigvals=(0, 1))  # For simplicity, using only the first eigenvector

    # Step 4: Perform k-means clustering on the eigenvectors
    # In practice, you might use a more sophisticated clustering algorithm
    k = 2  # Number of clusters
    _, labels = kmeans(eigenvectors[:, :k], k)

    return labels

def kmeans(X, k, max_iters=100):
    n, m = X.shape
    centers = X[np.random.choice(n, k, replace=False)]  # Randomly initialize cluster centers
    for _ in range(max_iters):
        # Assign each point to the nearest cluster center
        labels = np.argmin(np.linalg.norm(X[:, None] - centers, axis=2), axis=1)
        # Update cluster centers
        centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    return centers, labels

# Example usage
n = 100
affinity_matrix = np.random.rand(n, n)  # Replace this with your actual affinity matrix
segmentation_labels = normalized_cut(affinity_matrix)
print(segmentation_labels)
