import numpy as np
from tensorly import (
    to_numpy, vec_to_tensor, tensor_to_vec,
    fold, unfold
)
from tensorly.tenalg import kronecker
from forward import forward
from backward import backward
from mstep import update_lds_params
from mstep import update_mlds_params

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class MLDS(object):
    def __init__(self, X, ranks, init='random'):
        self.M = X.ndim
        self.T = T = X.shape[0]
        self.N = N = X.shape[1:]
        self.L = L = ranks
        self.initialize_parameters(N, L)
        if init is 'random':
            Zs, X = self.random_sample(T)
            self.Zs = Zs
            self.X = X
        else:
            self.X = X
        self.W = ~np.isnan(X)
        self.vecX = to_numpy([tensor_to_vec(X[t]) for t in range(T)])
        self.vecW = ~np.isnan(self.vecX)

    def fit(self, max_iter=10, tol=1.e-5,
            covariance_types=('diag', 'diag', 'diag')):
        """
        max_iter: the maximum # of EM iterations
        tol: the convergence bound
        covariance_types: Isotropic, diag, or full
        """
        print("Fitting LDS...")
        vecX = self.vecX
        params = pack_params(self)
        for iter in range(max_iter):
            # E-step
            mu, V, P = forward(vecX, params)
            Ez, Ezz, Ez1z = backward(mu, V, P, params)
            # M-step
            params = update_mlds_params(
                vecX, Ez, Ezz, Ez1z, covariance_types
            )

        print("Fitting MLDS with matching # of parameters")


    def initialize_parameters(self, I=None, J=None):
        """
        Inputs
        I: observation dimensionality
        J: latent dimensionality
        """
        M = len(I)
        I = self.I = to_numpy(I) if I is not None else self.I
        J = self.J = to_numpy(J) if J is not None else self.J
        Ip, Jp = np.prod(I), np.prod(J)
        IJ = I * J
        self.mu0 = np.random.normal(0, 1, size=Jp)
        self.Q0 = np.eye(Jp)
        self.Q = np.eye(Jp)
        self.R = np.eye(Ip)
        self.A = initialize_multilinear_operator(J, J)
        self.C = initialize_multilinear_operator(I, J)

    def random_sample(self, T, noise=True):
        Z = np.zeros((T, np.prod(self.J)))
        X = np.zeros((T, np.prod(self.I)))
        A = kronecker(self.A, reverse=True)
        C = kronecker(self.C, reverse=True)
        for t in trange(T, desc="random sampling"):
            if t == 0:
                Z[0, :] = np.random.multivariate_normal(self.mu0, self.Q0)
            else:
                Z[t, :] = np.random.multivariate_normal(A @ Z[t-1, :], self.Q)
            X[t, :] = np.random.multivariate_normal(C @ Z[t, :], self.R)
        return Z, X

    def unpack_params(self, params):
        self.mu0 = params["mu0"]
        self.Q0 = params["Q0"]
        self.Q = params["Q"]
        self.R = params["R"]
        self.A = params["A"]
        self.C = params["C"]

def pack_params(mlds):
    params = {
        "mu0": mlds.mu0,
        "Q0": mlds.Q0,
        "Q": mlds.Q,
        "R": mlds.R,
        "A": kronecker(mlds.A, reverse=True),
        "C": kronecker(mlds.C, reverse=True)
    }
    return params

def initialize_multilinear_operator(I, J):
    Ip = np.prod(I)
    Jp = np.prod(J)
    M = len(I)
    C = [None] * M
    for mode in range(M):
        size = (I[mode], I[mode])
        row = np.random.normal(size=size)
        while np.linalg.matrix_rank(row) < I[mode]:
            row = np.random.normal(size=size)
        U, S, V = np.linalg.svd(row)
        C[mode] = U[:, :J[mode]]
    return C


if __name__ == '__main__':

    X = np.zeros((100, 5, 5, 7))
    ranks = [2, 4, 7]

    model = MLDS(X, ranks)
    # model.initialize_parameters(X.shape[1:], ranks)
    # Z, X = model.random_sample(100)
    print(model.X.shape)
    model.fit()

    # plt.plot(model.X[:, 10])
    # plt.show()
