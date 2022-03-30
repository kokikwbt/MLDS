import time
import warnings
import numpy as np
import tensorly as tl

try:
    from optimizer import update_multilinear_operator
except:
    from .optimizer import update_multilinear_operator


def vec_to_tensor(vector, ranks, sequential=False):
    if sequential == True:
        return tl.to_numpy([vec_to_tensor(vt, ranks) for vt in vector])
    else:
        return vector.reshape(ranks, order='F')


def tensor_to_vec(tensor, sequential=False):
    if sequential == True:
        return tl.to_numpy([tensor_to_vec(tt) for tt in tensor])
    else:
        return tensor.flatten('F')


def update_mu0(Ez):
    return Ez[0]


def update_Q0(Ez, Ezz, J, covariance_type):

    Q0 = Ezz[0] - np.outer(Ez[0], Ez[0])

    if covariance_type == 'full':
        pass
    elif covariance_type == 'diag':
        Q0 = np.diag(np.diag(Q0))
    elif covariance_type == 'isotropic':
        Q0 = np.diag(np.tile(np.trace(Q0) / J, (J, 1)))

    return Q0


def update_A(A, Ezz, Ez1z, Sz1z, SzzT, Q, J, covariance_type):

    A_ = Sz1z @ np.linalg.pinv(SzzT)

    if len(J) == 1:
        A = [A_]

    else:
        if covariance_type == 'full':
            omg = np.eye(np.prod(J)) @ np.linalg.pinv(Q)
        elif covariance_type == 'diag':
            omg = 1 / np.diag(Q)
        elif covariance_type == 'isotropic':
            omg = 1 / Q

        psi = sum(Ezz[:-1])
        phi = sum(Ez1z[:-1])
        A = update_multilinear_operator(
            A, omg, psi, phi, covariance_type)

    return A


def update_C(C, Ezz, Szz, Sxz, R, I, covariance_type):

    C_ = Sxz @ np.linalg.pinv(Szz)

    if len(I) == 1:
        C = [C_]

    else:
        if covariance_type == 'full':
            omg = np.eye(np.prod(I)) @ np.linalg.pinv(R)
        elif covariance_type == 'diag':
            omg = 1 / np.diag(R)
        elif covariance_type == 'isotropic':
            omg = 1 / R

        psi = sum(Ezz)
        phi = Sxz
        C = update_multilinear_operator(
            C, omg, psi, phi, covariance_type)

    return C


def update_b(Ez, A):
    T = len(Ez)
    matA = tl.tenalg.kronecker(A, reverse=True)
    return sum([Ezt - matA @ Ezt1 for Ezt, Ezt1 in zip(Ez[1:], Ez[:-1])]) / (T - 1)


def update_d(vecX, Ez, C):
    matC = tl.tenalg.kronecker(C, reverse=True)
    return sum([vecXt - matC @ Ezt for vecXt, Ezt in zip(vecX, Ez)]) / vecX.shape[0]


def update_Q(Ezz, Szz, Sz1z, SzzT, J, A, covariance_type):
    T = len(Ezz)
    matA = tl.tenalg.kronecker(A, reverse=True)

    if covariance_type == "diag":
        Q = np.diag(Szz) - np.diag(Ezz[0])
        Q -= 2 * np.diag(matA @ Sz1z.T)
        Q += np.diag(matA @ SzzT @ matA.T)
        Q = np.diag(Q / (T - 1))

    elif covariance_type == "full":
        val = matA @ Sz1z.T
        Q = Szz - Ezz[0] - val - val.T + matA @ SzzT @ matA.T
        Q /= T - 1

    elif covariance_type == "isotropic":
        delta = np.trace(Szz) - np.trace(Ezz[0])
        delta -= 2 * np.trace(matA @ Sz1z.T)
        delta += np.trace(matA @ SzzT @ matA.T)
        delta = delta / (T - 1) / np.prod(J)
        Q = np.diag(np.matlib.repmat(delta, np.prod(J), 1))

    return Q


def update_R(vecX, Sxz, Szz, C, covariance_type):
    T, N = vecX.shape
    matC = tl.tenalg.kronecker(C, reverse=True)

    if covariance_type == "full":
        val = matC @ Sxz.T
        R = (vecX.T @ vecX - val - val.T + matC @ Szz @ matC.T) / T

    elif covariance_type == "diag":
        R = np.diag((
                np.diag(vecX.T @ vecX)
                - 2 * np.diag(matC @ Sxz.T)
                + np.diag(matC @ Szz @ matC.T)
            ) / T)

    elif covariance_type == "isotropic":
        delta = (np.trace(vecX.T @ vecX) - 2 * np.trace(matC @ Sxz.T) + np.trace(matC @ Szz @ matC.T)) / T / N
        R = np.diag(np.matlib.repmat(delta, N, 1))
        
    return R


class MLDS:
    def __init__(self,
        initial_state_cov='diag',
        transition_cov='diag',
        observation_cov='diag',
        transition_offset=False,
        observation_offset=False,
    ):
        covariance_types = ['full', 'isotropic', 'diag']

        assert initial_state_cov in covariance_types
        assert transition_cov    in covariance_types
        assert observation_cov   in covariance_types

        self.init_state_cov = initial_state_cov
        self.trans_cov = transition_cov
        self.obs_cov = observation_cov
        self.trans_offset = transition_offset
        self.obs_offset = observation_offset
        self.history = []

    @staticmethod
    def initialize_multilinear_operator(shape, ranks):

        multi_linear_operator = []

        for mode in range(len(shape)):
            size = (shape[mode], shape[mode])
            row = np.random.normal(0, 1, size=size)
            while np.linalg.matrix_rank(row) < shape[mode]:
                row = np.random.normal(0, 1, size=size)

            u, _, _ = np.linalg.svd(row)
            multi_linear_operator.append(u[:, :ranks[mode]])
        
        return multi_linear_operator

    def initialize(self, shape, ranks, init='random', random_state=None):

        self.T = T = shape[0]
        self.M = M = len(shape) - 1
        self.N = N = np.prod(shape[1:])
        self.L = L = np.prod(ranks)
        self.I = I = shape[1:]
        self.J = J = ranks

        prod_I = np.prod(I)
        prod_J = np.prod(J)

        self.Q0 = np.eye(prod_J)
        self.Q = np.eye(prod_J)
        self.R = np.eye(prod_I)

        if init == 'random':
            rand = np.random.RandomState(random_state)
            self.mu0 = rand.normal(0, 1, size=prod_J)
            self.A = self.initialize_multilinear_operator(J, J)
            self.C = self.initialize_multilinear_operator(I, J)
            self.b = rand.normal(0, 1, size=prod_J)
            self.d = rand.normal(0, 1, size=prod_I)
        
        else:
            self.mu0 = np.zeros(prod_J)
            self.A = [np.eye(J[m], J[m]) for m in range(M)]
            self.C = [np.eye(I[m], J[m]) for m in range(M)]
            self.b = np.zeros(prod_J)
            self.d = np.zeros(prod_I)

        # Allocation for the forward algorithm
        self.Ih = np.eye(L)
        self.mu = np.zeros((T, L))
        self.V = np.zeros((T, L, L))
        self.P = np.zeros((T, L, L))
        
        # Allocation for the backward algorithm
        self.Ez = np.zeros((T, L))
        self.Ezz = np.zeros((T, L, L))
        self.Ez1z = np.zeros((T, L, L))

    def forward(self, vecX):
        
        llh = 0.
        M = self.M
        Ih = self.Ih
        mu = self.mu
        V = self.V
        P = self.P
        A = tl.tenalg.kronecker(self.A, reverse=True)
        C = tl.tenalg.kronecker(self.C, reverse=True)
        Q = self.Q
        R = self.R
        b = self.b
        d = self.d

        for t in range(self.T):

            if t == 0:
                KP = self.Q0
                V[0] = self.Q0
                mu[0] = self.mu0

            else:
                P[t - 1] = A @ V[t - 1] @ A.T + Q
                KP = P[t - 1]
                mu[t] = A @ mu[t - 1] + b

            sgm = C @ KP @ C.T + R
            inv_sgm = np.linalg.pinv(sgm)

            K = KP @ C.T @ inv_sgm
            u = C @ mu[t] + d
            dlt = vecX[t] - u
            mu[t] = mu[t] + K @ dlt
            V[t] = (Ih - K @ C) @ KP

            # log-likelihood
            df = dlt @ inv_sgm @ dlt / 2
            sign, logdet = np.linalg.slogdet(inv_sgm)
            llh -= 0.5 * M * np.log(2 * np.pi)
            llh += sign * logdet * 0.5 - df

        return llh

    def backward(self):

        A = tl.tenalg.kronecker(self.A, reverse=True)
        V = self.V  # result of the forward algorithm
        P = self.P  # reuslt of the forward algorithm
        mu = self.mu
        Ez = self.Ez
        Ezz = self.Ezz
        Ez1z = self.Ez1z
        Vhat = self.V[-1]

        Ez[-1] = self.mu[-1]
        Ezz[-1] = Vhat + np.outer(Ez[-1], Ez[-1])

        for t in reversed(range(self.T - 1)):
            J = V[t] @ A.T @ np.linalg.pinv(P[t])
            Ez[t] = mu[t] + J @ (Ez[t + 1] - A @ mu[t])
            Ez1z[t] = Vhat @ J.T + np.outer(Ez[t + 1], Ez[t])
            Vhat = V[t] + J @ (Vhat - P[t]) @ J.T
            Ezz[t] = Vhat + np.outer(Ez[t], Ez[t])

        # return Ez, Ezz, Ez1z

    def solve(self, vecX):

        T = self.T
        Ez = self.Ez
        Ezz = self.Ezz
        Ez1z = self.Ez1z
        Szz  = sum(Ezz)
        Sz1z = sum(Ez1z[:-1])
        Sxz  = sum(np.outer(vecX[t], Ez[t]) for t in range(T))
        SzzT = Szz - Ezz[-1]

        # Initial state mean/covariance
        self.mu0 = update_mu0(Ez)
        self.Q0 = update_Q0(Ez, Ezz, self.J, self.init_state_cov)

        # Multilinear projections
        self.A = update_A(self.A, Ezz, Ez1z, Sz1z, SzzT, self.Q, self.J, self.trans_cov)
        self.C = update_C(self.C, Ezz, Szz, Sxz, self.R, self.I, self.obs_cov)

        # Offsets (optional)
        if self.trans_offset:
            self.b = update_b(Ez, self.A)
        if self.obs_offset:
            self.d = update_d(vecX, Ez, self.C)
        
        # Covariances
        self.Q = update_Q(Ezz, Szz, Sz1z, SzzT, self.J, self.A, self.trans_cov)
        self.R = update_R(vecX, Sxz, Szz, self.C, self.obs_cov)

    def fit(self, tensor, ranks, temporal_mode=0,
            max_iter=50, tol=1e-4, init='random', verbose=0):

        assert tensor.ndim == len(ranks) + 1

        if not temporal_mode == 0:
            tensor = np.moveaxis(tensor, temporal_mode, 0)

        self.initialize(tensor.shape, ranks)
        vecX = tensor_to_vec(tensor, sequential=True)

        # EM algorithm

        for iteration in range(max_iter):

            tic = time.process_time()
            # E-step
            llh = self.forward(vecX)
            self.backward()
            # M-step
            self.solve(vecX)

            toc = time.process_time() - tic
            self.history.append(llh)

            print('iter= {}; llh= {:.3f}; time= {:.3f} [sec]'.format(
                iteration+1, llh, toc))

            # convergence check
            if iteration > 2:
                if np.abs(self.history[-1] - self.history[-2]) < tol:
                    print('converged!!')
                    break

        else:
            warnings.warn("EM-algorithm did not converge")

    def predict(self, tensor):
        """ """
        assert tensor.shape[1:] == self.I
        raise NotImplementedError

    def smooth(self, tensor):
        assert tensor.shape[1:] == self.I
        raise NotImplementedError

