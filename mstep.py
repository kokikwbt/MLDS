import numpy as np
from numpy import diag, prod, trace
from scipy.linalg import pinv
from tensorly import to_numpy
from tensorly.tenalg import kronecker
from itertools import product
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
    if type_Q0 == "diag":
        Q0 = diag(diag(Q0))
    elif type_Q0 == "full":
        pass
    elif type_Q0 == "isotropic":
        Q0 = diag(np.tile(trace(Q0) / L, (L, 1)))

    A = Sz1z @ pinv(SzzN)

    if type_Q == "diag":
        Q = diag(Szz) - diag(Ezz[0])
        Q -= 2 * diag(A @ Sz1z.T)
        Q += diag(A @ SzzN @ A.T)
        Q = diag(Q / (T - 1))
    elif type_Q == "full":
        val = A @ Sz1z.T
        Q = Szz - Ezz[0] - val - val.T + A @ SzzN @ A.T
        Q /= T - 1
    elif type_Q == "isotropic":
        pass

    C = Sxz @ pinv(Szz)

    if type_R == "diag":
        R = diag(X.T @ X) - 2 * diag(C @ Sxz.T)
        R += diag(C @ Szz @ C.T)
        R = diag(R / T)
    elif type_R == "full":
        val = C @ Sxz.T
        R = X.T @ X - val - val.T + C @ Szz @ C.T / T
    elif type_R == "isotropic":
        pass

    return {"mu0": mu0, "Q0": Q0, "Q": Q, "R": R, "A": A, "C": C}

def update_mlds_params(X, params, Ez, Ezz, Ez1z, covariance_types):
    type_Q0, type_Q, type_R = covariance_types
    mu0 = params["mu0"]
    Q0 = params["Q0"]
    Q = params["Q"]
    R = params["R"]
    A = params["A"]
    b = params["b"]
    C = params["C"]
    d = params["d"]
    M = len(C)
    I = [C[m].shape[0] for m in range(M)]
    J = [A[m].shape[0] for m in range(M)]
    T = len(X)

    Sz1z = sum(Ez1z[:-1])
    Szz = sum(Ezz)
    Sxz = sum(np.outer(X[t, :], Ez[t]) for t in range(T))
    SzzT = Szz - Ezz[-1]

    """
    update mu0 & Q0
    """
    mu0 = Ez[0]
    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])
    if type_Q0 == "full":
        pass
    elif type_Q0 == "diag":
        Q0 = diag(diag(Q0))
    elif type_Q0 == "isotropic":
        # Q0 = diag(np.tile(trace(Q0) / J, (J, 1)))
        pass

    """
    update A
    """
    A_unfactorized = Sz1z @ pinv(SzzT)
    if M == 1:
        A = matA = A_unfactorized
    else:
        if type_Q == "full":
            omega = np.eye(prod(J)) @ pinv(Q)
        elif type_Q == "diag":
            omega = 1 / diag(Q)
        elif type_Q == "isotropic":
            omega = 1 / Q
        psi = sum(Ezz[:-1])
        phi = sum(Ez1z[:-1])
        A = update_multilinear_operator(A, omega, psi, phi, type_Q)
    matA = kronecker(A, reverse=True)

    """
    update b
    """
    b = sum([Ez[t, :] - matA @ Ez[t-1, :] for t in range(1, T)])
    b = np.asarray(b) / (T - 1)

    """
    update Q
    """
    if type_Q == "diag":
        Q = diag(Szz) - diag(Ezz[0])
        Q -= 2 * diag(matA @ Sz1z.T)
        Q += diag(matA @ SzzT @ matA.T)
        Q = diag(Q / (T - 1))
    elif type_Q == "full":
        val = matA @ Sz1z.T
        Q = Szz - Ezz[0] - val - val.T + matA @ SzzT @ matA.T
        Q /= T - 1
    elif type_Q == "isotropic":
        delta = trace(Szz) - trace(Ezz[0])
        delta -= 2 * trace(matA @ Sz1z.T)
        delta += trace(matA @ SzzT @ matA.T)
        delta = delta / (T - 1) / prod(J)
        Q = diag(np.matlib.repmat(delta, prod(J), 1))

    """
    update C (i.e., observation/projection tensor)
    """
    C_unfactorized = Sxz @ pinv(Szz)
    if M == 1:
        C = matC = C_unfactorized
    else:
        if type_R == "full":
            omega = np.eye(prod(I)) @ pinv(R)
        elif type_R == "diag":
            omega = 1 / diag(R)
        elif type_R == "isotropic":
            omega = 1 / R
        psi = sum(Ezz)
        phi = Sxz
        C = update_multilinear_operator(C, omega, psi, phi, type_R)
    matC = kronecker(C, reverse=True)

    """
    update d
    """
    d = sum([X[t] - matC @ Ez[t] for t in range(T)])
    d = np.asarray(d) / T

    """
    update R
    """
    if type_R == "full":
        val = matC @ Sxz.T
        R = (X.T @ X - val - val.T + matC @ Szz @ matC.T) / T
    elif type_R == "diag":
        R = diag(
            (diag(X.T @ X) - 2 * diag(matC @ Sxz.T)
            + diag(matC @ Szz @ matC.T)) / T
        )
    elif type_R == "isotropic":
        delta = (trace(X.T @ X) - 2 * trace(matC @ Sxz.T) + trace(matC @ Szz @ matC.T)) / T / N
        R = diag(np.matlib.repmat(delta, N, 1))

    params = {"mu0": mu0, "Q0": Q0, "Q": Q, "R": R, "A": A,
                "b": b, "d": d, "matA": matA, "C": C, "matC": matC}
    return params

def update_multilinear_operator(B, omega, psi, phi, cov_type):
    """
    B: factorized tensor
    """
    M = len(B)
    I = to_numpy([B[m].shape[0] for m in range(M)])
    J = to_numpy([B[m].shape[1] for m in range(M)])

    if cov_type == "diag":
        omega = (omega + omega.T) / 2
    psi = (psi + psi.T) / 2

    vecB = factors_to_vec(B)
    newB = descend(vecB, omega, psi, phi, cov_type, I, J)
    newB = vec_to_factors(newB, I, J)
    return newB

def descend(x0, omega, psi, phi, cov_type, I, J):
    f = lambda z: _f(z, omega, psi, phi, cov_type, I, J)
    df = lambda z: _df(z, omega, psi, phi, cov_type, I, J)
    """
    ---Inputs---
    f:  function from R^n to R, where n is the dimensionality of x0
    df:  gradient function of f
    ddf:  Hessian of f
    x0:  vector in R^n. Initial point
    ---Outputs---
    x:  vector in R^n.  Location of minimum closest to x_0 with respect to f.
    """
    epsilon = 1.e-18
    maxiter = 1.e+12
    learning_rate = .8
    step_size = .8
    # initialize
    numiter = 0
    x = x0
    fx = np.inf
    fx0 = -np.inf
    while abs(fx - fx0) / step_size > epsilon:
        numiter += 1
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
    B = vec_to_factors(z, I, J)
    B = kronecker(B, reverse=True)
    res = phi @ B.T
    res = B @ psi @ B.T - res - res.T
    if cov_type == "full":
        y = trace(omega @ res)
    elif cov_type == 'diag':
        y = omega.T @ diag(res)
    elif cov_type == 'isotropic':
        y = omega @ trace(res)
    return y

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
        Hm = H.reshape((*I, *J))
        Hm = np.transpose(Hm, [*notm, m, *notmM, m+M])
        Hm = tensor_to_mat(Hm)
        Im = range(I[m])
        Jm = range(J[m])
        for i, j in product(Im, Jm):
            r = np.arange(prod(I[notm])) + prod(I[notm]) * i
            c = np.arange(prod(J[notm])) + prod(J[notm]) * j
            # G[i, j] = sum((F * Hm[r, c]).T.flatten())
            G[i, j] = (F * Hm[r, :][:, c]).sum()
        index = np.arange(IJ[m]) + sum(IJ[:m-1])
        # print(index)
        # print(G.flatten('F').shape)
        # print(g[index])
        g[index] = 2 * G.flatten('F')
    return g

# https://jp.mathworks.com/help/matlab/math/matrix-indexing.html
# matlab: A(:) -> python: A.T.flatten() = A.flatten('F')
def factors_to_vec(factors):
    M = len(factors)
    v = [factors[m].T.flatten() for m in range(M)]  # concatenate column vectors
    return np.hstack(v)

def vec_to_factors(vector, I, J):
    M = len(I)
    sizes = I * J
    index = [np.arange(sizes[m]) + sum(sizes[:m-1]) for m in range(M)]
    T = [np.reshape(vector[index[m]], (I[m], J[m])) for m in range(M)]
    return T

def tensor_to_mat(tensor):
    I = tensor.shape
    M = tensor.ndim
    if M % 2 == 1:
        M += 1
        list(I).append(1)
    I = to_numpy(I)
    r = prod(I[:int(M/2)])
    c = prod(I[(np.arange(M/2) + M/2).astype('int8')])
    return tensor.flatten('F').reshape((c, r)).T
