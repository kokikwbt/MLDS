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
        self.init_mlds_params(I, J)
        # for training log
        self.llh = []

    def em(self, vecX=None, max_iter=10, tol=1.e-5,
            covariance_types=('diag', 'diag', 'diag'), log=True):
        """
        max_iter: the maximum # of EM iterations
        tol: the convergence bound
        covariance_types: Isotropic, diag, or full
        """
        print("\nFitting MLDS...")
        params = self.pack_params()
        if vecX is None:
            vecX = self.vecX

        for iter in range(max_iter):
            print("===> iter", iter + 1)

            # E-step
            mu, V, P, llh = forward(vecX, params, loglh=True)
            Ez, Ezz, Ez1z = backward(mu, V, P, params)

            # M-step
            params = update_mlds_params(
                vecX, params, Ez, Ezz, Ez1z, covariance_types
            )

            print("log-likelihood=", llh)
            # save training log
            self.llh.append(llh)
            if iter > 0:
                if abs(self.llh[-1] - self.llh[-2]) < tol:
                    print("converged!!")
                    break

        self.unpack_params(params)
        self.Ez = Ez

    def compute_log_likelihood(self):
        pass

    def init_mlds_params(self, I=None, J=None):
        """
        Inputs
        I: observation dimensionality
        J: latent dimensionality
        """
        M = len(self.I) if I is None else len(I)
        I = self.I = to_numpy(I) if I is not None else self.I
        J = self.J = to_numpy(J) if J is not None else self.J
        Ip = np.prod(I)
        Jp = np.prod(J)
        IJ = I * J
        self.mu0 = np.random.normal(0, 1, size=Jp)
        self.Q0 = np.eye(Jp) / 2
        self.Q = np.eye(Jp) / 2
        self.R = np.eye(Ip) / 2
        self.A = A = init_multilinear_operator(J, J)
        self.C = C = init_multilinear_operator(I, J)
        self.b = np.random.rand(Jp)
        self.d = np.random.rand(Ip)
        self.matA = kronecker(A, reverse=True)
        self.matC = kronecker(C, reverse=True)

    def random_sample(self, T):
        self.rndZ = vecZ = np.zeros((T, np.prod(self.J)))
        self.rndX = vecX = np.zeros((T, np.prod(self.I)))
        for t in trange(T, desc="random sampling"):
            if t == 0:
                vecZ[0, :] = np.random.multivariate_normal(self.mu0, self.Q0)
            else:
                vecZ[t, :] = np.random.multivariate_normal(self.matA @ vecZ[t-1, :], self.Q)
            vecX[t, :] = np.random.multivariate_normal(self.matC @ vecZ[t, :], self.R)
        # convert vector to tensor
        Z = to_numpy([vecZ[t].reshape(self.J[::-1]).T for t in range(T)])
        X = to_numpy([vecX[t].reshape(self.I[::-1]).T for t in range(T)])
        return (X, vecX), (Z, vecZ)

    def pack_params(self):
        params = {
            "mu0": self.mu0,
            "Q0": self.Q0,
            "Q": self.Q,
            "R": self.R,
            "A": self.A,
            "b": self.b,
            "C": self.C,
            "d": self.d,
            "matA": self.matA,
            "matC": self.matC
        }
        return params

    def unpack_params(self, params):
        self.mu0 = params["mu0"]
        self.Q0 = params["Q0"]
        self.Q = params["Q"]
        self.R = params["R"]
        self.A = params["A"]
        self.b = params["b"]
        self.C = params["C"]
        self.d = params["d"]
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
        np.savetxt(outdir+"vecb.txt", self.b)
        np.savetxt(outdir+"vecd.txt", self.d)
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
        bar(self.b, title="b", outfn=outdir+"vecb.png")
        bar(self.d, title="d", outfn=outdir+"vecd.png")
        for m in range(self.M):
            heatmap(self.A[m], title=f"A_{m}",
                    outfn=outdir+f"A_{m}.png")
            heatmap(self.C[m], title=f"C_{m}",
                    outfn=outdir+f"C_{m}.png")


def init_multilinear_operator(I, J):
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


def main():
    # generate synthetic tensor series
    X = np.zeros((100, 5, 5, 7))
    ranks = [2, 4, 2]
    model = MLDS(X, ranks)
    (X, vecX), _ = model.random_sample(X.shape[0])
    # X = to_numpy([vec_to_tensor(vecX[t], X.shape[1:]) for t in range(len(X))])

    # fit MLDS
    model = MLDS(X, ranks)
    # model.em(max_iter=10, covariance_types=('full', 'full', 'full'))
    # model.em(max_iter=20, covariance_types=('diag', 'diag', 'diag'))
    model.em(max_iter=10, covariance_types=('diag', 'full', 'diag'))
    model.save_params()

if __name__ == '__main__':
    main()
