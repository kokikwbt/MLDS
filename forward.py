#!/bin/env python3
import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import pinv
from tqdm import trange


def forward(X, params, loglikelihood=True):
    # inputs
    T, N = X.shape
    M = len(params["A"])
    A = params["matA"]
    C = params["matC"]
    b = params["b"]
    d = params["d"]
    Q = params["Q"]
    R = params["R"]
    L = A.shape[1]  # dimension of latent variable
    Ih = np.eye(L)
    # outputs
    mu = np.zeros((T, L))
    V = np.zeros((T, L, L))
    P = np.zeros((T, L, L))
    llh = 0

    for t in trange(T, desc="forward"):
        if t == 0:
            KP = params["Q0"]
            V[0] = params["Q0"]
            mu[0] = params["mu0"]
        else:
            P[t-1] = A @ V[t-1] @ A.T + Q
            KP = P[t-1]
            mu[t] = A @ mu[t-1] + b

        sigma_c = C @ KP @ C.T + R
        inv_sgm = pinv(sigma_c)

        K = KP @ C.T @ inv_sgm
        u_c = C @ mu[t] + d
        delta = X[t, :] - u_c
        mu[t] = mu[t] + K @ delta
        V[t] = (Ih - K @ C) @ KP

        if loglikelihood:
            df = delta @ inv_sgm @ delta / 2
            if df < 0:
                print("det of not positive definite < 0")
            sign, logdet = slogdet(inv_sgm)
            llh -= M / 2 * np.log(2 * np.pi)
            llh += sign * logdet / 2 - df

    return mu, V, P, llh
