'''
# GOOD Code for getting C matrix

import numpy as np
from githubclusteringcode import ElasticNetSubspaceClustering

# generate 7 data points from 3 independent subspaces as columns of data matrix X
X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
              [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

model = ElasticNetSubspaceClustering(n_clusters=3,algorithm='lasso_lars',gamma=50).fit(X.T)
print(model.labels_)
print(model.affinity_matrix_)
print(model.affinity_matrix_.shape)
# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labels
'''


'''
GOOD Code for Spectra Clustering

import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment

def spectral_clustering(A, num_clusters):
    # Apply Spectral Clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=0).fit(A)
    labels = clustering.labels_
    return labels

def hungarian_mapping(true_labels, predicted_labels):
    # Create a cost matrix based on label similarities
    num_classes = len(np.unique(true_labels))
    num_clusters = len(np.unique(predicted_labels))
    cost_matrix = np.zeros((num_classes, num_clusters))
    for i in range(num_classes):
        for j in range(num_clusters):
            # Count how many points from true class i are assigned to predicted cluster j
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    
    # Apply Hungarian algorithm to find the best mapping
    true_class_indices, cluster_indices = linear_sum_assignment(-cost_matrix)
    mapping = {true_class_indices[i]: cluster_indices[i] for i in range(len(true_class_indices))}
    mapped_labels = [mapping[label] for label in predicted_labels]
    
    return mapped_labels

def create_similarity_matrix(N):
    # Generate random similarity scores
    A = np.random.rand(N, N)
    # Make the matrix symmetric (similarity matrix property)
    A = (A + A.T) / 2
    # Ensure diagonal elements are 1 (each sample is perfectly similar to itself)
    np.fill_diagonal(A, 1)
    return A

# Example usage:
N = 1000  # Change N to desired size
A = create_similarity_matrix(N)
true_labels = np.random.randint(0, 51, size=N)

# Spectral Clustering
num_clusters = 51  # Assuming you want 2 clusters
predicted_labels = spectral_clustering(A, num_clusters)

# Hungarian Mapping
mapped_labels = hungarian_mapping(true_labels, predicted_labels)

print("Predicted Labels:", predicted_labels)
print("Mapped Labels:", mapped_labels)
'''



'''
Given a matrix A of shape NxN where N represents the number of samples, and A[i][j] represents the score value for sample i and sample j,
provide python code for the following:

1) Spectral clustering is applied to A to obtain the clustering labels, by assigning each of the N datapoint xj into its corresponding subspace
2) Apply Hungarian algorithm to compare and map subspace labels into actual class labels
'''