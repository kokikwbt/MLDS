import os
import shutil
import warnings
import numpy as np
import pandas as pd
from tensorly import (
    to_numpy, vec_to_tensor, tensor_to_vec,
    fold, unfold
)
from tensorly.tenalg import kronecker
from forward import forward
from backward import backward
from mstep import update_lds_params
from mstep import update_mlds_params

from sklearn.preprocessing import scale
from tqdm import tqdm, trange
from myplot import *

warnings.filterwarnings("ignore")

class MLDS(object):
    def __init__(self, X, ranks):
        self.M = X.ndim - 1
        self.T = T = X.shape[0]
        self.I = I = X.shape[1:]
        self.J = J = ranks
        self.X = X
        self.W = ~np.isnan(X)
        self.vecX = to_numpy([X[t].flatten('F') for t in range(T)])
        self.vecW = ~np.isnan(self.vecX)
        self.initialize_parameters(I, J)
        # for training log
        self.llh = []

    def em(self, max_iter=10, tol=1.e-5,
            covariance_types=('diag', 'diag', 'diag'), log=True):
        """
        max_iter: the maximum # of EM iterations
        tol: the convergence bound
        covariance_types: Isotropic, diag, or full
        """
        print("\nFitting MLDS...")
        vecX = self.vecX

        for iter in range(max_iter):
            print("===> iter", iter + 1)
            # E-step
            params = pack_params(self, multilinear=False)
            mu, V, P, llh = forward(vecX, params, loglh=True)
            Ez, Ezz, Ez1z = backward(mu, V, P, params)
            # M-step
            # params = pack_params(self, multilinear=True)
            params = update_mlds_params(
                vecX, params, Ez, Ezz, Ez1z, covariance_types
            )
            self.unpack_params(params)
            print("log-likelihood=", llh)
            # save training log
            self.llh.append(llh)
            if iter > 0:
                if abs(self.llh[-1] - self.llh[-2]) < tol:
                    print("converged!!")
                    break
        self.Ez = Ez

    def compute_log_likelihood(self):
        pass

    def initialize_parameters(self, I=None, J=None):
        """
        Inputs
        I: observation dimensionality
        J: latent dimensionality
        """
        M = len(self.I) if I is None else len(I)
        I = self.I = to_numpy(I) if I is not None else self.I
        J = self.J = to_numpy(J) if J is not None else self.J
        Ip, Jp = np.prod(I), np.prod(J)
        IJ = I * J
        self.mu0 = np.random.normal(0, 1, size=Jp)
        self.Q0 = np.eye(Jp) / 2
        self.Q = np.eye(Jp) / 2
        self.R = np.eye(Ip) / 2
        self.A = A = initialize_multilinear_operator(J, J)
        self.C = C = initialize_multilinear_operator(I, J)
        self.matA = kronecker(A, reverse=True)
        self.matC = kronecker(C, reverse=True)

    def random_sample(self, T):
        self.rndZ = Z = np.zeros((T, np.prod(self.J)))
        self.rndX = X = np.zeros((T, np.prod(self.I)))
        for t in trange(T, desc="random sampling"):
            if t == 0:
                Z[0, :] = np.random.multivariate_normal(self.mu0, self.Q0)
            else:
                Z[t, :] = np.random.multivariate_normal(self.matA @ Z[t-1, :], self.Q)
            X[t, :] = np.random.multivariate_normal(self.matC @ Z[t, :], self.R)
        return X, Z

    def unpack_params(self, params):
        self.mu0 = params["mu0"]
        self.Q0 = params["Q0"]
        self.Q = params["Q"]
        self.R = params["R"]
        self.A = params["A"]
        self.C = params["C"]
        self.matA = params["matA"]
        self.matC = params["matC"]

    def save_params(self, outdir="./out/"):
        outdir += "results/"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        np.save(outdir+"X", self.X)
        np.save(outdir+"mu0", self.mu0)
        np.save(outdir+"Q0", self.Q0)
        np.save(outdir+"Q", self.Q)
        np.save(outdir+"R", self.R)
        np.savetxt(outdir+"vecX.txt", self.vecX)
        np.savetxt(outdir+"vecz.txt", self.Ez)
        np.savetxt(outdir+"matA.txt", self.matA)
        np.savetxt(outdir+"matC.txt", self.matC)
        for m in range(self.M):
            np.savetxt(outdir+f"A_{m}.txt", self.A[m])
            np.savetxt(outdir+f"C_{m}.txt", self.C[m])
        plot(self.llh, xlabel="# of EM iterations",
             ylabel="Log-likelihood",
             outfn=outdir+"loss.png")
        heatmap(self.matA, title="mat(A)",
                outfn=outdir+"matA.png")
        heatmap(self.matC, title="mat(C)",
                outfn=outdir+"matC.png")
        heatmap(self.Q0, title="initial state covariance",
                outfn=outdir+"Q0.png")
        heatmap(self.Q, title="Transitioin covariance",
                outfn=outdir+"Q.png")
        heatmap(self.R, title="Observation covariance",
                outfn=outdir+"R.png")
        for m in range(self.M):
            heatmap(self.A[m], title=f"A_{m}",
                    outfn=outdir+f"A_{m}.png")
            heatmap(self.C[m], title=f"C_{m}",
                    outfn=outdir+f"C_{m}.png")


def pack_params(mlds, multilinear=True):
    params = {
        "mu0": mlds.mu0,
        "Q0": mlds.Q0,
        "Q": mlds.Q,
        "R": mlds.R,
        "A": mlds.A,
        "C": mlds.C,
        "matA": mlds.matA,
        "matC": mlds.matC
    }
    return params

def initialize_multilinear_operator(I, J):
    Ip = np.prod(I)
    Jp = np.prod(J)
    M = len(I)
    B = [None] * M
    for mode in range(M):
        size = (I[mode], I[mode])
        row = np.random.normal(size=size)
        while np.linalg.matrix_rank(row) < I[mode]:
            row = np.random.normal(size=size)
        U, S, V = np.linalg.svd(row)
        B[mode] = U[:, :J[mode]]
    return B


if __name__ == '__main__':

    # generate synthetic tensor series
    X = np.zeros((100, 5, 5, 7))
    ranks = [2, 4, 2]
    model = MLDS(X, ranks)
    vecX, _ = model.random_sample(X.shape[0])
    X = to_numpy([vec_to_tensor(vecX[t], X.shape[1:]) for t in range(len(X))])

    # fit MLDS
    model = MLDS(X, ranks)
    # model.em(max_iter=10, covariance_types=('full', 'full', 'full'))
    # model.em(max_iter=20, covariance_types=('diag', 'diag', 'diag'))
    model.em(max_iter=10, covariance_types=('diag', 'full', 'diag'))
    model.save_params()
