import numpy as np

import config

def generate_symmetric_binary_matrix(n):
    # Generate a random upper triangular matrix
    upper_triangular = np.random.randint(2, size=(n,n))
    
    # Make it symmetric by adding it to its transpose
    symmetric_matrix = np.triu(upper_triangular) + np.triu(upper_triangular, 1).T
    
    return symmetric_matrix

def get_linear_temporal_graph():
    n = config.Nt
    adj = np.zeros((n, n), dtype=int)
    for i in range(n-1):
        adj[i][i+1] = adj[i+1][i] = 1
    return adj

def cartesian_product(adj_matrix1, adj_matrix2):
    n1 = len(adj_matrix1)
    n2 = len(adj_matrix2)
    # Initialize the adjacency matrix for the Cartesian product graph
    cartesian_adj_matrix = np.zeros((n1 * n2, n1 * n2), dtype=int)

    for i1 in range(n1):
        for i2 in range(n1):
            for j1 in range(n2):
                for j2 in range(n2):
                    r = n2 * i1 + j1
                    c = n2 * i2 + j2
                    if (i1 == i2 and adj_matrix2[j1][j2]) or (
                        j1 == j2 and adj_matrix1[i1][i2]
                    ):
                        cartesian_adj_matrix[r][c] = 1

    return cartesian_adj_matrix