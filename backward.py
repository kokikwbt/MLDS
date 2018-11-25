import numpy as np
from scipy.linalg import pinv
from tqdm import tqdm
"""
"""
def backward(mu, V, P, params):
    T, N = mu.shape
    A = params["matA"]
    C = params["matC"]
    Q = params["Q"]
    R = params["R"]
    L = A.shape[1]  # dimension of latent variable
    Ih = np.eye(L)

    Ez = np.zeros((T, L))
    Ezz = np.zeros((T, L, L))
    Ez1z = np.zeros((T, L, L))
    Vhat = V[-1]
    Ez[-1] = mu[-1]
    Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])

    for t in tqdm(list(reversed(range(T - 1))), desc="backward"):
        J = V[t] @ A.T @ pinv(P[t])
        Ez[t] = mu[t] + J @ (Ez[t+1] - A @ mu[t])
        Ez1z[t] = Vhat @ J.T + np.outer(Ez[t+1], Ez[t])
        Vhat = V[t] + J @ (Vhat - P[t]) @ J.T
        Ezz[t] = Vhat + np.outer(Ez[t], Ez[t])

    return Ez, Ezz, Ez1z
