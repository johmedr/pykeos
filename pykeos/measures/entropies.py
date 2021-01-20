from ..tools import correlation_sum, delay_coordinates, reference_rule
from ..tools.conv_utils import _make_array
import numpy as np


def sample_entropy(x, radius=None, dim=2, norm_p=float('inf'), axis=0):
    _x = _make_array(x)

    Xm = delay_coordinates(_x, dim=dim, lag=1, return_array=True, axis=axis)

    if radius is None:
        radius = reference_rule(Xm, dim=dim, norm_p=float('inf'))

    B = correlation_sum(Xm, r=radius, norm_p=norm_p)
    Xm_1 = delay_coordinates(_x, dim=dim + 1, lag=1, return_array=True, axis=axis)
    A = correlation_sum(Xm_1, r=radius, norm_p=norm_p)

    return - np.log(A / B)


def approximate_entropy():
    pass


def ks_entropy():
    pass
