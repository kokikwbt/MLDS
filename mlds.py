#!/bin/env python3
import os
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, normalize
from tensorly import fold, unfold, to_numpy
from tensorly.tenalg import kronecker
from tqdm import tqdm, trange

from forward import forward
from backward import backward
from mlds_solver import update_mlds_params
from reshape import vec_to_tensor, tensor_to_vec
from myplot import *


class MLDS(object):
    """
    """
    def __init__(self, X, ranks):
        self.M = X.ndim - 1
        self.T = T = X.shape[0]  # sequence length
        self.I = I = to_numpy(X.shape[1:])  # shape of observation tensor
        self.J = J = to_numpy(ranks)  # shape of latent state tensor
        self.X = X
        self.W = ~np.isnan(X)
        self.vecX = tensor_to_vec(X, sequential=True)
        self.vecW = ~np.isnan(self.vecX)
        self.hist = []  # for training log
        self.init_mlds_params(random=True)

    def init_mlds_params(self, random=True):
        """
        """
        M, I, J = self.M, self.I, self.J
        Ip, Jp, IJ = np.prod(I), np.prod(J), I * J
        self.Q0 = np.eye(Jp)  # initial state covariances
        self.Q = np.eye(Jp)  # transition covariances
        self.R = np.eye(Ip)  # observation covariances

        if random == True:
            rnd = np.random.RandomState(None)
            self.mu0 = rnd.normal(0, 1, size=Jp)
            self.A = init_multilinear_operator(J, J)
            self.C = init_multilinear_operator(I, J)
            self.b = rnd.normal(0, 1, size=Jp)
            self.d = rnd.normal(0, 1, size=Ip)

        else:
            self.mu0 = np.zeros(Jp)
            self.A = [np.eye(J[m], J[m]) for m in range(M)]
            self.C = [np.eye(I[m], J[m]) for m in range(M)]
            self.b = np.zeros(Jp)
            self.d = np.zeros(Ip)

        self.matA = kronecker(self.A, reverse=True)
        self.matC = kronecker(self.C, reverse=True)

    def em(self, covariance_types, X=None,
           max_iter=10, tol=1.e-5, verbose=False):
        """EM algorithm for MLDS

        max_iter: the maximum # of EM iterations
        tol: the convergence bound
        covariance_types: [isotropic/diag/full]

        """
        X = self.X if X is None else X
        vecX = tensor_to_vec(X, sequential=True)
        params = self.pack_params()

        for iteration in trange(max_iter, desc='EM'):
            # E-step
            mu, V, P, llh = forward(vecX, params, loglikelihood=True)
            Ez, Ezz, Ez1z = backward(mu, V, P, params)
            # M-step
            params = update_mlds_params(
                vecX, params, Ez, Ezz, Ez1z, covariance_types
            )

            print(f'log-likelihood= {llh:.2f}')
            self.hist.append(llh)
            if iteration > 0:
                if abs(self.hist[-1] - self.hist[-2]) < tol:
                    break  # converged

        self.unpack_params(params)
        self.vecZ = Ez

    def random_sample(self, n_sample, return_vec=False):
        """"""
        vecZ = np.zeros((n_sample, np.prod(self.J)))
        vecX = np.zeros((n_sample, np.prod(self.I)))

        for t in trange(n_sample, desc="random sampling"):
            if t == 0:
                vecZ[0, :] = np.random.multivariate_normal(self.mu0, self.Q0)
            else:
                vecZ[t, :] = np.random.multivariate_normal(self.matA @ vecZ[t-1, :], self.Q)
            vecX[t, :] = np.random.multivariate_normal(self.matC @ vecZ[t, :], self.R)

        # convert vector to tensor
        Z = vec_to_tensor(vecZ, self.J, sequential=True)
        X = vec_to_tensor(vecX, self.I, sequential=True)

        if return_vec == True:
            return Z, X, vecZ, vecX
        else:
            return Z, X

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
        np.savetxt(outdir+"vecz.txt", self.vecZ)
        np.savetxt(outdir+"matA.txt", self.matA)
        np.savetxt(outdir+"matC.txt", self.matC)
        np.savetxt(outdir+"vecb.txt", self.b)
        np.savetxt(outdir+"vecd.txt", self.d)
        for m in range(self.M):
            np.savetxt(outdir+f"A_{m}.txt", self.A[m])
            np.savetxt(outdir+f"C_{m}.txt", self.C[m])
        plot(self.hist, xlabel="# of EM iterations",
             ylabel="Log-likelihood",
             outfn=outdir+"loss.png")
        heatmap(self.matA, title="mat(A)",
                outfn=outdir+"matA.png")
        heatmap(self.matC, title="mat(C)",
                outfn=outdir+"matC.png")
        heatmap(self.Q0, title="initial state covariance",
                outfn=outdir+"Q0.png")
        heatmap(self.Q, title="Transition covariance",
                outfn=outdir+"Q.png")
        heatmap(self.R, title="Observation covariance",
                outfn=outdir+"R.png")
        bar(self.b, title="b", outfn=outdir+"vecb.png")
        bar(self.d, title="d", outfn=outdir+"vecd.png")
        for m in range(self.M):
            heatmap(self.A[m], center=0,
                    title=f"A_{m}", outfn=outdir+f"A_{m}.png")
            heatmap(self.C[m], center=0,
                    title=f"C_{m}", outfn=outdir+f"C_{m}.png")


def init_multilinear_operator(I, J):
    Ip = np.prod(I)
    Jp = np.prod(J)
    M = len(I)
    B = [None] * M
    for mode in range(M):
        size = (I[mode], I[mode])
        row = np.random.normal(0, 1, size=size)
        while np.linalg.matrix_rank(row) < I[mode]:
            row = np.random.normal(0, 1, size=size)
        U, S, V = np.linalg.svd(row)
        B[mode] = U[:, :J[mode]]
    return B


def main():
    # generate synthetic tensor series
    X = np.zeros((100, 5, 5, 7))
    ranks = [2, 4, 2]
    model = MLDS(X, ranks)
    _, X = model.random_sample(X.shape[0])

    # fit MLDS
    model = MLDS(X, ranks)
    model.em(max_iter=20, covariance_types=('diag', 'diag', 'diag'))

    model.save_params()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
