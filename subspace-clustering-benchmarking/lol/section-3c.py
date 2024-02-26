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
    d, num_frames = X.shape
    r = U.shape[1]
    I = np.identity(num_frames)
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

# Example usage
n = 100
d = 50
X = np.random.rand(n, d)  # Example time series data
X = X.transpose()

r = 10
s = 3
lambda1 = 0.1
lambda2 = 0.1
alpha = 0.1

Z, D, U, V = graph_clustering(X, r, s, lambda1, lambda2, alpha)
# print("Z:", Z)
# print("D:", D)
# print("U:", U)
# print("V:", V)

print("X:", X.shape)
print("D:", D.shape)
print("Z:", Z.shape)

W = cosine_similarity(Z)
print("W:", W.shape)

'''
Provide python code to solve for V:

[I ⊗ (U^T U + (λ1 + alpha)I) + λ2*L ⊗ I]vec(V)= vec(U^T X - Π + alpha Z),

where ⊗ is the tensor product.

'''

'''
Provide python code for the following:

Input: Time series data X of shape (num_frames, d), k = 0, step size η, number
of clusters k, parameters s, λ1, λ2, alpha

Output: Z, D, U, V

1) Initialize W of shape nxn
wij = 1, if |i - j| ≤ s/2, 0, otherwise.

2) Initialize D' which is the degree matrix for graph W
3) Initialize L which is the laplacian matrix for graph W

Objective function is w.r.t Z, D, U, V
U = D, V = Z, Z ≥ 0, D ≥ 0

while not converged do 
    - Update V (fix others)
      (tr(U)*U + (λ1 + alpha)I)*V + lambda2*V*L = tr(U)*X - Π + alpha*Z
      where Π is the lagrange multiplier. solve this equation to get updated V

    - Update U (fix others)
        U = (X*tr(V) - Λ + alpha*D)(V*tr(V) + alpha*I)^-1
        where Λ is the lagrange multiplier

    - Update Z and D fixing others
        Z = F+(V + Π/alpha) 
        D = F+(U + Λ/alpha)
        where (F+(A))ij = max{Aij , 0}

     
(Here * denotes matrix multiplication or scalar to matrix multiplication)
(tr(A) denotes transpose of A)

'''


'''
Python Code for temporal subspace clustering. 

Precisely, given a dictionary D belonging to R(dxr) and a coding matrix Z belonging to R(rxn), a collection of data points X belonging to R(dxn) can be approximately represented as X = DZ;
where each data point is encoded using a Least Squares regression, and a temporal Laplacian regularization L(Z) function encourages the encoding of the sequential relationships in time-series data. This can done by minimising
min (Z,D) norm(X - DZk)^2 + lambda1*norm(Z)^2 + lambda2*L(Z) subject to Z >= 0; D >= 0; using the ADMM algorithm to encourage convergence by
solving a stack of easier sub-problems

Subspace Clustering for Action Recognition with
Covariance Representations and Temporal Pruning
Giancarlo Paoletti, Jacopo Cavazza, Cigdem Beyan and Alessio Del Bue
Pattern Analysis and Computer Vision, Istituto Italiano di Tecnologia, via Enrico Melen 83, 16152 Genova, Italy

do you have this paper in your dataset?

'''
