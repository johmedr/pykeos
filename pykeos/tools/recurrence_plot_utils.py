from ._impl.impl import _localized_diagline_histogram, \
    _localized_vertline_histogram, _localized_white_vertline_histogram, _weighted_diagline_histogram
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


from tqdm import tqdm

def weighted_diagline_histogram(recmat, scale, order=4):
    recmat = recmat.astype(np.int8)
    n_times = recmat.shape[0]
    diag_hist = localized_diagline_histogram(recmat)
    max_diag_length = int(max(_[2] for _ in diag_hist) + 1)
    if max_diag_length > n_times:
        raise RuntimeError(f"max_diag_length ({max_diag_length}) is greater that n_times ({n_times})")
    weighted_histogram = np.zeros((n_times, max_diag_length), dtype=float)
    _weighted_diagline_histogram(n_times, diag_hist, weighted_histogram, scale, order)
    return weighted_histogram






