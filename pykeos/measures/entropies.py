from ..tools import correlation_sum, delay_coordinates, reference_rule
from ..tools.io_conversion import _make_array
from ..tools.nd_utils import nd_function, windowed_function
from ..tools.correlation_sum import _fast_count_row
import numpy as np


@nd_function
@windowed_function
def sample_entropy(x, radius=None, dim=2, norm_p=float('inf'), axis=0):
    _x = _make_array(x)

    Xm = delay_coordinates(_x, dim=dim, lag=1, return_array=True, axis=axis)
    Xm_1 = delay_coordinates(_x, dim=dim + 1, lag=1, return_array=True, axis=axis)

    if radius is None:
        radius = reference_rule(Xm, dim=dim, norm_p=float('inf'))

    B = correlation_sum(Xm, r=radius, norm_p=norm_p)
    A = correlation_sum(Xm_1, r=radius, norm_p=norm_p)

    return - np.log(A / B)


@nd_function
@windowed_function
def approximate_entropy(x, radius=None, dim=2, norm_p=float('inf'), axis=0):
    _x = _make_array(x)

    Xm = delay_coordinates(_x, dim=dim, lag=1, return_array=True, axis=axis)
    Xm_1 = delay_coordinates(_x, dim=dim + 1, lag=1, return_array=True, axis=axis)

    if radius is None:
        radius = reference_rule(Xm, dim=dim, norm_p=float('inf'))

    Bi = np.apply_along_axis(_fast_count_row, 1, Xm, Xm, radius, norm_p).astype(np.float64)
    Ai = np.apply_along_axis(_fast_count_row, 1, Xm_1, Xm_1, radius, norm_p).astype(np.float64)

    logAi = np.log(Ai)
    logAi = logAi[logAi == logAi]

    logBi = np.log(Bi)
    logBi = logBi[logBi == logBi]

    return np.mean(logBi) - np.mean(logAi)


def ks_entropy():
    pass
