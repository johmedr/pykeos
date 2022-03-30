from ._impl.impl import _localized_diagline_histogram, \
    _localized_vertline_histogram, _localized_white_vertline_histogram, \
    _weighted_diagline_histogram, _weighted_vertline_histogram, _weighted_white_vertline_histogram

import numpy as np


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


def weighted_diagline_histogram(recmat, scale, order=2):
    recmat = recmat.astype(np.int8)
    n_times = recmat.shape[0]
    diag_hist = localized_diagline_histogram(recmat)
    max_diag_length = int(max(_[2] for _ in diag_hist) + 1)
    if max_diag_length > n_times:
        raise RuntimeError(f"max_diag_length ({max_diag_length}) is greater that n_times ({n_times})")
    weighted_histogram = np.zeros((n_times, max_diag_length), dtype=float)
    _weighted_diagline_histogram(n_times, diag_hist, weighted_histogram, scale, order)
    return weighted_histogram


def weighted_vertline_histogram(recmat, scale, order=2):
    recmat = recmat.astype(np.int8)
    n_times = recmat.shape[0]
    vert_hist = localized_vertline_histogram(recmat)
    max_vert_length = int(max(_[2] for _ in vert_hist) + 1)
    if max_vert_length > n_times:
        raise RuntimeError(f"max_vert_length ({max_vert_length}) is greater that n_times ({n_times})")
    weighted_histogram = np.zeros((n_times, max_vert_length), dtype=float)
    _weighted_vertline_histogram(n_times, vert_hist, weighted_histogram, scale, order)
    return weighted_histogram


def weighted_white_vertline_histogram(recmat, scale, order=2):
    recmat = recmat.astype(np.int8)
    n_times = recmat.shape[0]
    white_vert_hist = localized_white_vertline_histogram(recmat)
    max_white_vert_length = int(max(_[2] for _ in white_vert_hist) + 1)
    if max_white_vert_length > n_times:
        raise RuntimeError(f"max_white_vert_length ({max_white_vert_length}) is greater that n_times ({n_times})")
    weighted_histogram = np.zeros((n_times, max_white_vert_length), dtype=float)
    _weighted_white_vertline_histogram(n_times, white_vert_hist, weighted_histogram, scale, order)
    return weighted_histogram






