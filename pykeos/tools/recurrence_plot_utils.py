from ._impl.impl import _localized_diagline_histogram, \
    _localized_vertline_histogram, _localized_white_vertline_histogram
import numpy as np
from scipy.sparse import lil_matrix


def localized_diagline_histogram(recmat):
    n_times = int(recmat.shape[0])
    diagline_d = _localized_diagline_histogram(n_times, recmat)
    return diagline_d


def localized_vertline_histogram(recmat):
    n_times = int(recmat.shape[0])
    vertline_d = _localized_vertline_histogram(n_times, recmat)
    return vertline_d


def localized_white_vertline_histogram(recmat):
    n_times = int(recmat.shape[0])
    white_vertline_d = _localized_white_vertline_histogram(n_times, recmat)
    return white_vertline_d

#
# def weighting_kernel(t, i, j, scale):
#     return np.exp(- (max(abs(t - i), abs(t - j))/scale) ** 2)
#
#
# def weighted_diagline_histogram(recmat, scale):
#     recmat = recmat.astype(np.int8)
#     n_times = recmat.shape[0]
#     diag_hist = localized_vertline_histogram(recmat)
#     weighted_histogram = lil_matrix((n_times, int(np.sqrt(2 * n_times**2))))
#     for t in range(n_times):
#         for (i, j, k) in diag_hist:
#             weight = weighting_kernel(t, (i + k) / 2, (j + k) / 2, scale)
#             weighted_histogram[t, k] += weight
#
#     return weighted_histogram






