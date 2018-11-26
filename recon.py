import numpy as np
from tensorly import vec_to_tensor
from tensorly.tenalg import kronecker
import matplotlib.pyplot as plt

outdir = './out/results/'

X = np.load(outdir+'X.npy')
vecX = np.loadtxt(outdir+'vecX.txt')
T = X.shape[0]
N = X.shape[1:]
M = X.ndim
A = [np.loadtxt(outdir+f'A_{m}.txt') for m in range(M)]
C = [np.loadtxt(outdir+f'C_{m}.txt') for m in range(M)]
z = np.loadtxt(outdir+'vecz.txt')

matC = kronecker(C, reverse=True)
pred = np.zeros((T, np.prod(N)))

for t in range(T):
    pred[t, :] = matC @ z[t, :]

for i in range(6):
    plt.figure()
    plt.plot(vecX[:, i])
    plt.plot(pred[:, i])

plt.show()
