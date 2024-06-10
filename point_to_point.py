import numpy as np

def minimization(P, Q):
    errors_before = Q - P # Errors at the beginning
    mu_p = np.mean(P[0:3, :], axis=1) #Centroide of each pointcloud
    mu_q = np.mean(Q[0:3, :], axis=1)
    P_mu = np.ones((3, P.shape[1])) # Centered each pointclouds
    Q_mu = np.ones((3, Q.shape[1]))
    for i in range(0, P_mu.shape[1]):
        P_mu[0:3, i] = P[0:3, i] - mu_p
    for i in range(0, Q_mu.shape[1]):
        Q_mu[0:3, i] = Q[0:3, i] - mu_q
    H = P_mu @ Q_mu.T # Cross covariance matrix
    U, s, V = np.linalg.svd(H) # Use SVD decomposition
    M = np.eye(3) # Compute rotation
    M[2, 2] = np.linalg.det(V.T @ U.T)
    R = V.T @ M @ U.T
    t = mu_q - R @ mu_p # Compute translation
    T = np.eye(4) # Transformation matrix
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

