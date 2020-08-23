import torch


__all__ = ['dynamics']


def dynamics(W, X, tol=1e-6, max_iter=5, mode='replicator', **kwargs):
    """
    Selector for dynamics
    Input:
    W:  the pairwise nxn similarity matrix (with zero diagonal)
    X:  an (n,m)-array whose rows reside in the n-dimensional simplex
    tol:  error tolerance
    max_iter:  maximum number of iterations
    mode: 'replicator' to run the replicator dynamics
    """

    if mode == 'replicator':
        X = _replicator(W, X, tol, max_iter)
    else:
        raise ValueError('mode \'' + mode + '\' is not defined.')

    return X


def _replicator(W, X, tol, max_iter):
    """
    Replicator Dynamics
    Output:
    X:  the population(s) at convergence
    i:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """

    i = 0
    while i < max_iter:
        X = X * torch.matmul(W, X)
        X /= X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1)
        i += 1

    return X
