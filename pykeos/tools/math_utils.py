import numpy as np
from scipy.special import gamma
import pandas as pd
from sklearn.metrics import mutual_info_score
from typing import Union


# def n_ball_volume(dim, norm):
#     if norm == 1:
#         return 2**dim/np.math.factorial(dim)
#     elif norm == 2:
#         return np.pi**(dim/2.) / (gamma(dim/2. + 1))
#     elif norm == float('inf') or norm == "inf":
#         return 2 ** dim


def n_ball_volume(dim, norm_p):
    if norm_p == 1:
        from scipy.special import factorial
        return 2.**dim / factorial(dim)
    elif norm_p == 2:
        return (np.sqrt(np.pi))**dim / gamma(dim/2 + 1)
    elif norm_p == float('inf') or norm_p == "inf":
        return 2 ** dim
    elif 0 < norm_p < float('inf'):
        return (2 * gamma(1 / norm_p + 1))**dim / gamma(dim/norm_p + 1)


def n_sphere_area(dim, norm):
    if norm == 1:
        return 2 ** dim / np.math.factorial(dim - 1)
    elif norm == 2:
        return 2 * np.pi ** ((dim + 1) / 2.) / (gamma((dim + 1) / 2.))
    elif norm == float('inf') or norm == "inf":
        return dim * 2 ** dim


def nd_rand_init(*tuples_lo_hi):
    return np.random.uniform(low=[t[0] for t in tuples_lo_hi], high=[t[1] for t in tuples_lo_hi])


def make_uniform_kernel(dim, norm_p):
    return lambda u: 1./n_ball_volume(dim, norm_p) if u <= 1 else 0.


def mutual_information(x, y, size_orig, n_bins="smooth") -> float:
    assert(x.size > 0 and y.size > 0)
    assert(n_bins in [None, 'smooth'] or isinstance(n_bins, int))
    if n_bins is None:
        n_bins = int(np.floor(np.sqrt(size_orig / 5)))

    elif n_bins is 'smooth':
        n_bins = int(np.floor(np.sqrt(size_orig / 3)))

    c_xy = np.histogram2d(x, y, n_bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def lagged_mi(x, lag=1, *args, **kwargs) -> float:
    assert(lag >= 0)
    if lag > 0:
        return mutual_information(x[:-lag], x[lag:], x.shape[0], *args, **kwargs)
    else:
        return mutual_information(x, x, x.shape[0], *args, **kwargs)


def _lstsqr_design_matrix(x, degree=1):
    if degree > 1:
        raise NotImplementedError()
    return np.concatenate([x[:,None], np.ones((x.shape[0], 1))], axis=1)


def lstsqr(X, y):
    Xt = np.transpose(X)
    return np.linalg.pinv(Xt.dot(X)).dot(Xt).dot(y)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
