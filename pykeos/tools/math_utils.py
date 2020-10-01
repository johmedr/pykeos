import numpy as np
from scipy.special import gamma

from sklearn.metrics import mutual_info_score
from typing import Union

def n_ball_volume(dim, norm):
    if norm == 1:
        return 2**dim/np.math.factorial(dim)
    elif norm == 2:
        return np.pi**(dim/2.) / (gamma(dim/2. + 1))
    elif norm == float('inf') or norm == "inf":
        return 2 ** dim


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


def mutual_information(x, y, n_bins=None) -> float:
    assert(x.size > 0 and y.size > 0)
    assert(n_bins is None or isinstance(n_bins, int))
    if n_bins is None:
        n_bins = int(np.floor(np.sqrt(x.shape[0] / 5)))

    c_xy = np.histogram2d(x, y, n_bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def lagged_mi(x, lag=1, n_bins=None) -> float:
    assert(lag >= 0)
    if lag > 0:
        return mutual_information(x[:-lag], x[lag:], n_bins=n_bins)
    else:
        return mutual_information(x, x, n_bins=n_bins)