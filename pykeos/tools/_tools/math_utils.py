import numpy as np
from scipy.special import gamma


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


def delay_coordinates(ts: np.ndarray, dim: int, lag: int = 1, axis: int = 0) -> np.ndarray:
    return np.vstack([ts[i * lag:(i - dim) * lag, axis] for i in range(dim)]).T