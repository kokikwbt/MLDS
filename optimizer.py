#!/bin/env python3
import numpy as np
from numpy import diag, prod, trace
from tensorly import to_numpy
from tensorly.tenalg import kronecker
from itertools import product


def vec_to_factors(vector, I, J):
    """
    Given:
    - vector: np.array
    - I: list object
    - J: list object

    Convert vector to the list of matrices,
    each of which shape is i x j in [I, J].
    """
    sizes = I * J
    index = []

    for m in range(len(I)):
        index.append(np.arange(sizes[m]))
        if m > 0: index[-1] += sum(sizes[:m])

    return [vector[index[m]].reshape((I[m], J[m]), order='F') for m in range(len(I))]


def factors_to_vec(factors):
    return np.concatenate([factor.flatten('F') for factor in factors])


def mat_to_tensor(matrix, shape):
    return matrix.flatten('F').reshape(shape, order='F')


def tensor_to_mat(tensor):
    """
    """
    M = tensor.ndim
    I = list(tensor.shape)

    if M % 2 == 1:
        M += 1
        I.append(1)

    I = to_numpy(I)
    r = np.prod(I[:int(M/2)])
    c = np.prod(I[(np.arange(M/2) + M/2).astype(int)])

    return tensor.flatten('F').reshape((r, c), order='F')


def update_multilinear_operator(B, omega, psi, phi, cov_type):
    """
    B: factorized tensor
    """
    M = len(B)
    I = to_numpy([B[m].shape[0] for m in range(M)])
    J = to_numpy([B[m].shape[1] for m in range(M)])

    if not cov_type == "diag":
        omega = (omega + omega.T) / 2

    psi = (psi + psi.T) / 2
    vecB = factors_to_vec(B)
    newB = descend(vecB, omega, psi, phi, cov_type, I, J)
    newB = vec_to_factors(newB, I, J)

    return newB


def descend(x0, omega, psi, phi, cov_type, I, J,
            epsilon=1.e-18, maxiter=1.e+12, learning_rate=.8):
    """
    ---Inputs---
    f:  function from R^n to R, where n is the dimensionality of x0
    df:  gradient function of f
    ddf:  Hessian of f
    x0:  vector in R^n. Initial point
    ---Outputs---
    x:  vector in R^n.  Location of minimum closest to x_0 with respect to f.
    """
    f = lambda z: _f(z, omega, psi, phi, cov_type, I, J)
    df = lambda z: _df(z, omega, psi, phi, cov_type, I, J)
    step_size = learning_rate
    x = x0
    fx = np.inf
    fx0 = -np.inf

    while abs(fx - fx0) / step_size > epsilon:

        x0 = x
        fx0 = f(x0)
        x = x0 - step_size * df(x0)
        fx = f(x)

        while fx > fx0:
            step_size *= learning_rate
            x = x0 - step_size * df(x0)
            fx = f(x)

    return x


def _f(z, omega, psi, phi, cov_type, I, J):

    B = kronecker(vec_to_factors(z, I, J), reverse=True)
    tmp = phi @ B.T
    tmp = B @ psi @ B.T - tmp - tmp.T

    if cov_type == "full":
        return trace(omega @ tmp)

    elif cov_type == 'diag':
        return omega.T @ diag(tmp)

    elif cov_type == 'isotropic':
        return omega @ trace(tmp)


def _df(z, omega, psi, phi, cov_type, I, J):

    IJ = I * J
    M = len(I)
    B = vec_to_factors(z, I, J)
    H = psi @ kronecker(B, reverse=True).T - phi.T
    H = H @ diag(omega) if cov_type == "diag" else H @ omega
    H = H.T
    g = np.zeros(sum(IJ))

    for m in range(M):
        notm = to_numpy([i for i in range(M) if not i == m])
        notmM = notm + M

        F = kronecker(B, skip_matrix=m, reverse=True)
        G = np.zeros((I[m], J[m]))

        shape = to_numpy((*I, *J)).astype(int)

        Hm = mat_to_tensor(H, shape)
        Hm = np.transpose(Hm, [*notm, m, *notmM, m+M])
        Hm = tensor_to_mat(Hm)

        Im = range(I[m])
        Jm = range(J[m])

        for i, j in product(Im, Jm):
            r = np.arange(prod(I[notm])) + prod(I[notm]) * i
            c = np.arange(prod(J[notm])) + prod(J[notm]) * j
            G[i, j] = (F * Hm[r, :][:, c]).sum()

        index = np.arange(IJ[m])
        if m > 0: index += sum(IJ[:m])
        g[index] = 2 * G.flatten('F')

    return g
