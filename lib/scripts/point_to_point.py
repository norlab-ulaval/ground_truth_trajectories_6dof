import numpy as np

def minimization(P, Q):
    # Errors at the beginning
    errors_before = Q - P
    # Centroide of each pointcloud
    mu_p = np.mean(P[0:3, :], axis=1)
    mu_q = np.mean(Q[0:3, :], axis=1)
    # Center each pointcloud
    P_mu = np.ones((3, P.shape[1]))
    Q_mu = np.ones((3, Q.shape[1]))
    for i in range(0, P_mu.shape[1]):
        P_mu[0:3, i] = P[0:3, i] - mu_p
    for i in range(0, Q_mu.shape[1]):
        Q_mu[0:3, i] = Q[0:3, i] - mu_q
    # Compute cross covariance matrix
    H = P_mu @ Q_mu.T
    # Use SVD decomposition
    U, s, V = np.linalg.svd(H)
    # Compute rotation
    M = np.eye(3)
    M[2, 2] = np.linalg.det(V.T @ U.T)
    R = V.T @ M @ U.T
    # if np.linalg.det(R) < 0:
    #     # print(V.T)
    #     V_t = V.T
    #     V_t[:, 2] = -V_t[:, 2]
    #     R = V_t @ U.T
    # Compute translation
    t = mu_q - R @ mu_p
    # Compute rigid transformation obtained
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T