import numpy as np
from scipy.linalg import pinv
from tqdm import trange
"""
"""
def forward(X, params):
    T, N = X.shape
    A = params["matA"]
    C = params["matC"]
    Q = params["Q"]
    R = params["R"]
    L = A.shape[1]  # dimension of latent variable
    Ih = np.eye(L)

    mu = np.zeros((T, L))
    V = np.zeros((T, L, L))
    P = np.zeros((T, L, L))

    for t in trange(T, desc="forward"):
        if t == 0:
            KP = params["Q0"]
            V[0] = params["Q0"]
            mu[0] = params["mu0"]
        else:
            P[t-1] = A @ V[t-1] @ A.T + Q
            KP = P[t-1]
            mu[t] = A @ mu[t-1]

        sigma_c = C @ KP @ C.T + R
        inv_sgm = pinv(sigma_c)

        K = KP @ C.T @ inv_sgm
        u_c = C @ mu[t]
        delta = X[t, :] - u_c
        mu[t] = mu[t] + K @ delta
        V[t] = (Ih - K @ C) @ KP

    return mu, V, P
