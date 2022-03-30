from ._impl.impl import _localized_diagline_histogram, \
    _localized_vertline_histogram, _localized_white_vertline_histogram, \
    _weighted_diagline_histogram, _weighted_vertline_histogram, _weighted_white_vertline_histogram

from pyunicorn.timeseries import RecurrencePlot
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


class TimeResolvedRecurrencePlot(RecurrencePlot): 
    def __init__(self, time_series, metric="supremum", normalize=False,
                 missing_values=False, sparse_rqa=False, silence_level=0,
                 **kwds):
        super().__init__(time_series, metric, normalize, missing_values, sparse_rqa, silence_level, **kwds)

        self._time_resolved_diagline_dist = None
        self._time_resolved_diagline_dist_scale = 0
        self._time_resolved_diagline_dist_order = 0

        self._time_resolved_vertline_dist = None
        self._time_resolved_vertline_dist_scale = 0
        self._time_resolved_vertline_dist_order = 0

        self._time_resolved_white_vertline_dist = None
        self._time_resolved_white_vertline_dist_scale = 0
        self._time_resolved_white_vertline_dist_order = 0

    def time_resolved_diagline_dist(self, scale=None, order=2):
        if scale is not None:
            assert(scale > 0)
        if (self._time_resolved_diagline_dist_scale == scale or scale is None) and self._time_resolved_diagline_dist_order == order:
            return self._time_resolved_diagline_dist
        else:
            if scale is None: 
                scale = int(0.05 * self.N) 
            recmat = self.recurrence_matrix()
            self._time_resolved_diagline_dist = weighted_diagline_histogram(recmat, scale, order)
            self._time_resolved_diagline_dist_scale = scale
            self._time_resolved_diagline_dist_order = order

    def time_resolved_vertline_dist(self, scale=None, order=2): 
        if scale is not None:
            assert(scale > 0)
        if (self._time_resolved_vertline_dist_scale == scale or scale is None) and self._time_resolved_vertline_dist_order == order:
            return self._time_resolved_vertline_dist
        else:
            if scale is None: 
                scale = int(0.05 * self.N) 
            recmat = self.recurrence_matrix()
            self._time_resolved_vertline_dist = weighted_vertline_histogram(recmat, scale, order)
            self._time_resolved_vertline_dist_scale = scale
            self._time_resolved_vertline_dist_order = order

    def time_resolved_white_vertline_dist(self, scale=None, order=2): 
        if scale is not None:
            assert(scale > 0)
        if (self._time_resolved_white_vertline_dist_scale == scale or scale is None) and self._time_resolved_white_vertline_dist_order == order:
            return self._time_resolved_white_vertline_dist
        else:
            if scale is None: 
                scale = int(0.05 * self.N) 
            recmat = self.recurrence_matrix()
            self._time_resolved_white_vertline_dist = weighted_white_vertline_histogram(recmat, scale, order)
            self._time_resolved_white_vertline_dist_scale = scale
            self._time_resolved_white_vertline_dist_order = order


    def time_resolved_determinism(self, l_min=2, scale=None, order=2): 
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            return np.zeros((self.N,))
        
        partial_sum = (np.arange(l_min, diagline.shape[1])[None, :] * diagline[:, l_min:]).sum(axis=1)
        full_sum = (np.arange(diagline.shape[1])[None] * diagline).sum(axis=1)

        return partial_sum / float(full_sum + self._epsilon)

    def time_resolved_max_diaglength(self, scale=None, order=2): 
        diagline = self.time_resolved_diagline_dist(scale, order)
        return diagline.argmax(axis=1)

    def time_resolved_average_diaglength(self, l_min=2, scale=None, order=2): 
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            return np.zeros((self.N,))
        
        partial_sum = (np.arange(l_min, diagline.shape[1])[None, :] * diagline[:, l_min:]).sum(axis=1)
        full_sum = diagline[:, l_min:].sum(axis=1)
        return partial_sum / float(full_sum + self._epsilon)

    def time_resolved_diag_entropy(self, l_min=2, scale=None, order=2): 
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,)) 

        diagline = diagline[:, l_min]
        diagline = diagline[diagline != 0]

        diagnorm = diagline / float(diagline.sum(axis=1, keepdims=True) + self._epsilon)
        return -(diagnorm * np.log(diagnorm)).sum(axis=1)

    def time_resolved_laminarity(self, v_min=2, scale=None, order=2): 
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            return np.zeros((self.N,))
        
        partial_sum = (np.arange(v_min, vertline.shape[1])[None, :] * vertline[:, v_min:]).sum(axis=1)
        full_sum = (np.arange(vertline.shape[1])[None] * vertline).sum(axis=1)
        
        return partial_sum / float(full_sum + self._epsilon)

    def time_resolved_max_vertlength(self, scale=None, order=2): 
        vertline = self.time_resolved_vertline_dist(scale, order)
        return vertline.argmax(axis=1)

    def time_resolved_average_vertlength(self, v_min=2, scale=None, order=2): 
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            return np.zeros((self.N,))
        
        partial_sum = (np.arange(v_min, vertline.shape[1])[None, :] * vertline[:, v_min:]).sum(axis=1)
        full_sum = vertline[:, v_min:].sum(axis=1)
        return partial_sum / float(full_sum + self._epsilon)

    def time_resolved_trapping_time(self, v_min=2, scale=None, order=2): 
        return self.time_resolved_average_vertlength(v_min, scale, order)

    def time_resolved_vert_entropy(self, v_min=2, scale=None, order=2): 
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,)) 

        vertline = vertline[:, v_min]
        vertline = vertline[vertline != 0]

        vertnorm = vertline / float(vertline.sum(axis=1, keepdims=True) + self._epsilon)
        return -(vertnorm * np.log(vertnorm)).sum(axis=1)

    def time_resolved_max_white_vertlength(self, scale=None, order=2): 
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)
        return white_vertline.argmax(axis=1)

    def time_resolved_average_white_vertlength(self, w_min=2, scale=None, order=2): 
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)

        if w_min > white_vertline.shape[1]:
            return np.zeros((self.N,))
        
        partial_sum = (np.arange(w_min, white_vertline.shape[1])[None, :] * white_vertline[:, w_min:]).sum(axis=1)
        full_sum = white_vertline[:, w_min:].sum(axis=1)
        return partial_sum / float(full_sum + self._epsilon)

    def time_resolved_mean_recurrence_time(self, w_min=2, scale=None, order=2): 
        return self.time_resolved_average_white_vertlength(self, w_min, scale, order)

    def time_resolved_white_vert_entropy(self, w_min=2, scale=None, order=2): 
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)

        if w_min > white_vertline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,)) 

        white_vertline = white_vertline[:, w_min]
        white_vertline = white_vertline[white_vertline != 0]

        white_vertnorm = white_vertline / float(white_vertline.sum(axis=1, keepdims=True) + self._epsilon)
        return -(white_vertnorm * np.log(white_vertnorm)).sum(axis=1)