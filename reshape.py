#!/bin/env python3
import numpy as np
from tensorly import to_numpy


def vec_to_tensor(vector, ranks, sequential=False):
    """
    *** IF sequential is TRUE:

    *** IF sequential is FALSE:

    """
    if sequential == True:
        return to_numpy([vec_to_tensor(vt, ranks) for vt in vector])
    else:
        return vector.reshape(ranks, order='F')


def tensor_to_vec(tensor, sequential=False):
    """
    *** IF sequential is TRUE:
        vectorize tensor Xt for each time tick

    *** IF sequential is FALSE:
        vectorize input tensor X

    NOTE
    ----
    *** https://jp.mathworks.com/help/matlab/math/matrix-indexing.html
    *** matlab: A(:) = python: A.T.flatten() = A.flatten('F')

    """
    if sequential == True:
        return to_numpy([tensor_to_vec(tt) for tt in tensor])
    else:
        return tensor.flatten('F')


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
    """Concatenate column vectors
    """
    return np.concatenate([factor.flatten('F') for factor in factors])


def mat_to_tensor(matrix, shape):
    """
    """
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
