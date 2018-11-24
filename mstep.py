import numpy as np
from scipy.linalg import pinv

"""
Maximum likelihood estimation of
multi-linear/linear dynamical system (M-step)
"""

def update_lds_params(X, Ez, Ezz, Ez1z, covariance_types):
    T, N = X.shape
    L = Ez.shape[1]
    type_Q0, type_Q, type_R = covariance_types

    Sz1z = sum(Ez1z[:-1])
    Szz = sum(Ezz)
    Sxz = sum(np.outer(X[t, :], Ez[t]) for t in range(T))
    SzzN = Szz - Ezz[-1]

    mu0 = Ez[0]

    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
    if type_Q0 is "diag":
        Q0 = np.diag(np.diag(Q0))
    elif type_Q0 is "full":
        pass
    elif type_Q0 is "isotropic":
        Q0 = np.diag(np.tile(np.trace(Q0) / L, (L, 1)))

    A = Sz1z @ pinv(SzzN)

    if type_Q is "diag":
        Q = np.diag(Szz) - np.diag(Ezz[0])
        Q -= 2 * np.diag(A @ Sz1z.T)
        Q += np.diag(A @ SzzN @ A.T)
        Q = np.diag(Q / (T - 1))
    elif type_Q is "full":
        val = A @ Sz1z.T
        Q = Szz - Ezz[0] - val - val.T + A @ SzzN @ A.T
        Q /= T - 1
    elif type_Q is "isotropic":
        pass

    C = Sxz @ pinv(Szz)

    if type_R is "diag":
        R = np.diag(X.T @ X) - 2 * np.diag(C @ Sxz.T)
        R += np.diag(C @ Szz @ C.T)
        R = np.diag(R / T)
    elif type_R is "full":
        val = C @ Sxz.T
        R = X.T @ X - val - val.T + C @ Szz @ C.T / T
    elif type_R is "isotropic":
        pass

    return {"mu0": mu0, "Q0": Q0, "Q": Q, "R": R, "A": A, "C": C}

def update_mlds_params():
    pass
