import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
sns.set()

def plot(X, figure=True, title='', xlabel='', ylabel='',
         tight_layout=False, show=False, close=False, outfn=''):
    if figure: plt.figure()
    plt.plot(X)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if tight_layout: plt.gcf().tight_layout()
    if show: plt.show()
    if outfn:
        plt.savefig(outfn)
        plt.close()
    if close: plt.close()

def fit_plot(X, Y, outfn=''):
    plt.figure()
    plt.subplot(211)
    plt.title("Fitting result")
    plt.plot(X)
    plt.xlabel("Time")
    plt.ylabel("Original value")
    plt.subplot(212)
    plt.plot(Y)
    plt.xlabel("Time")
    plt.ylabel("Reconstructed value")
    if outfn:
        plt.savefig(outfn)
        plt.close()

def fit_scatter(X, Y, outfn=''):
    for i in range(X.shape[1]):
        plt.figure()
        plt.title(f"X_{i}")
        plt.plot(X[:, i])
        plt.plot(Y[:, i])
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.savefig(outfn+str(i)+'.png')
        plt.close()

def heatmap(M, figure=True, title='', xlabel='', ylabel='',
            show=False, close=False, outfn=''):
    if figure: plt.figure()
    sns.heatmap(M)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if show: plt.show()
    if outfn:
        plt.savefig(outfn)
        plt.close()
    if close: plt.close()

def bar(M, figure=True, title='', xlabel='', ylabel='',
            show=False, close=False, outfn=''):
    if figure: plt.figure()
    plt.bar(np.arange(len(M), dtype=int), M)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if show: plt.show()
    if outfn:
        plt.savefig(outfn)
        plt.close()
    if close: plt.close()
