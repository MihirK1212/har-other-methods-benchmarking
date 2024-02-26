import numpy as np
from scipy.linalg import block_diag, solve
from sklearn.metrics.pairwise import cosine_similarity


def initialize_W(n, s):
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= s // 2:
                W[i, j] = 1
    return W

def update_V(X, U, Z, alpha, lambda1, lambda2, L, pi):
    d, n = X.shape
    r = U.shape[1]
    I = np.identity(n)
    vec_UX = np.reshape(U.T @ X - pi + alpha * Z, (-1, 1))  # Vectorize U^T X - Pi + alpha Z
    M = np.kron(I, U.T @ U + (lambda1 + alpha) * np.identity(r)) + \
        lambda2 * np.kron(L, np.identity(r))
    vec_V = solve(M, vec_UX)
    V = np.reshape(vec_V, (r, n))
    return V

def update_U(X, V, D, alpha, dellamb):
    tr_V = np.transpose(V)
    A = X @ tr_V - dellamb + alpha * D
    B = V @ tr_V + alpha * np.identity(V.shape[0])
    U = A @ np.linalg.inv(B)
    return U

def update_Z_D(V, U, alpha, pi, dellamb):
    Z = np.maximum(V + pi / alpha, 0)
    D = np.maximum(U + dellamb / alpha, 0)
    return Z, D

def laplacian_matrix(W):
    D = np.sum(W, axis=1)
    D = np.diag(D)
    L = D - W
    return L

def graph_clustering(X, r, s, lambda1, lambda2, alpha, lr = 0.1, max_iters=100, tolerance=1e-5):
    d, num_frames = X.shape

    W = initialize_W(num_frames, s)
    assert W.shape == (num_frames, num_frames)
    
    D_prime = np.diag(np.sum(W, axis=1))
    L = laplacian_matrix(W)
    I = np.identity(num_frames)
    
    # Initialize Z, D, U, V
    Z = np.random.rand(r, num_frames)
    D = np.random.rand(d, r)
    U = D
    V = Z

    pi = np.zeros((r, num_frames))
    dellamb = np.zeros((d, r))
    
    for _ in range(max_iters):

        old_Z, old_D, old_U, old_V = Z.copy(), D.copy(), U.copy(), V.copy()
        
        # Update V
        V = update_V(X, old_U, old_Z, alpha, lambda1, lambda2, L, pi=np.zeros((r, num_frames)))
        
        # Update U
        U = update_U(X, old_V, old_D, alpha, dellamb = dellamb)
        
        # Update Z and D
        Z, D = update_Z_D(old_V, old_U, alpha, pi = pi, dellamb = dellamb)

        # Update pi and dellamb
        pi = pi + lr*alpha*(V-Z)
        dellamb = dellamb + lr*alpha*(U-D)
        
        assert Z.shape == (r, num_frames)
        assert D.shape == (d, r)
        assert U.shape == (d, r)
        assert V.shape == (r, num_frames)

        # # Check for convergence
        # if np.linalg.norm(Z - old_Z) < tolerance and np.linalg.norm(D - old_D) < tolerance:
        #     break
    
    return Z, D, U, V