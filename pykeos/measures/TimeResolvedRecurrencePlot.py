import numpy as np
from pyunicorn.timeseries import RecurrencePlot
from ..tools import weighted_white_vertline_histogram, weighted_vertline_histogram, weighted_diagline_histogram


class TimeResolvedRecurrencePlot(RecurrencePlot):
    def __init__(self, time_series, metric="supremum", normalize=False,
                 missing_values=False, sparse_rqa=False, silence_level=0, default_scale=None,
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

        if default_scale is None:
            default_scale = int(0.05 * self.N)

        self._default_scale = default_scale

    def time_resolved_diagline_dist(self, scale=None, order=2):
        if scale is not None:
            assert (scale > 0)
        if (
                self._time_resolved_diagline_dist_scale == scale or scale is None) and self._time_resolved_diagline_dist_order == order:
            return self._time_resolved_diagline_dist
        else:
            if scale is None:
                scale = self._default_scale
            recmat = self.recurrence_matrix()
            self._time_resolved_diagline_dist = weighted_diagline_histogram(recmat, scale, order)
            self._time_resolved_diagline_dist_scale = scale
            self._time_resolved_diagline_dist_order = order

            return self._time_resolved_diagline_dist

    def time_resolved_vertline_dist(self, scale=None, order=2):
        if scale is not None:
            assert (scale > 0)
        if (
                self._time_resolved_vertline_dist_scale == scale or scale is None) and self._time_resolved_vertline_dist_order == order:
            return self._time_resolved_vertline_dist
        else:
            if scale is None:
                scale = self._default_scale
            recmat = self.recurrence_matrix()
            self._time_resolved_vertline_dist = weighted_vertline_histogram(recmat, scale, order)
            self._time_resolved_vertline_dist_scale = scale
            self._time_resolved_vertline_dist_order = order
            return self._time_resolved_vertline_dist

    def time_resolved_white_vertline_dist(self, scale=None, order=2):
        if scale is not None:
            assert (scale > 0)
        if (
                self._time_resolved_white_vertline_dist_scale == scale or scale is None) and self._time_resolved_white_vertline_dist_order == order:
            return self._time_resolved_white_vertline_dist
        else:
            if scale is None:
                scale = self._default_scale
            recmat = self.recurrence_matrix()
            self._time_resolved_white_vertline_dist = weighted_white_vertline_histogram(recmat, scale, order)
            self._time_resolved_white_vertline_dist_scale = scale
            self._time_resolved_white_vertline_dist_order = order
            return self._time_resolved_white_vertline_dist

    def time_resolved_determinism(self, l_min=2, scale=None, order=2):
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            return np.zeros((self.N,))

        partial_sum = (np.arange(l_min, diagline.shape[1])[None, :] * diagline[:, l_min:]).sum(axis=1)
        full_sum = (np.arange(diagline.shape[1])[None] * diagline).sum(axis=1).astype(float)

        return partial_sum / (full_sum + self._epsilon)

    def time_resolved_max_diaglength(self, scale=None, order=2):
        diagline = self.time_resolved_diagline_dist(scale, order)
        return diagline.argmax(axis=1)

    def time_resolved_average_diaglength(self, l_min=2, scale=None, order=2):
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            return np.zeros((self.N,))

        partial_sum = (np.arange(l_min, diagline.shape[1])[None, :] * diagline[:, l_min:]).sum(axis=1)
        full_sum = diagline[:, l_min:].sum(axis=1).astype(float)
        return partial_sum / full_sum + self._epsilon

    def time_resolved_diag_entropy(self, l_min=2, scale=None, order=2):
        diagline = self.time_resolved_diagline_dist(scale, order)

        if l_min > diagline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,))

        diagline = diagline[:, l_min:]
        diagnorm = diagline / (diagline.sum(axis=1, keepdims=True).astype(float) + self._epsilon)
        valid_entries = diagline > 0
        entr = np.zeros_like(diagline)
        entr[valid_entries] = -(diagnorm[valid_entries] * np.log(diagnorm[valid_entries]))

        return entr.sum(axis=1)

    def time_resolved_laminarity(self, v_min=2, scale=None, order=2):
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            return np.zeros((self.N,))

        partial_sum = (np.arange(v_min, vertline.shape[1])[None, :] * vertline[:, v_min:]).sum(axis=1)
        full_sum = (np.arange(vertline.shape[1])[None] * vertline).sum(axis=1).astype(float)

        return partial_sum / (full_sum + self._epsilon)

    def time_resolved_max_vertlength(self, scale=None, order=2):
        vertline = self.time_resolved_vertline_dist(scale, order)
        return vertline.argmax(axis=1)

    def time_resolved_average_vertlength(self, v_min=2, scale=None, order=2):
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            return np.zeros((self.N,))

        partial_sum = (np.arange(v_min, vertline.shape[1])[None, :] * vertline[:, v_min:]).sum(axis=1)
        full_sum = vertline[:, v_min:].sum(axis=1).astype(float)
        return partial_sum / full_sum + self._epsilon

    def time_resolved_trapping_time(self, v_min=2, scale=None, order=2):
        return self.time_resolved_average_vertlength(v_min, scale, order)

    def time_resolved_vert_entropy(self, v_min=2, scale=None, order=2):
        vertline = self.time_resolved_vertline_dist(scale, order)

        if v_min > vertline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,))

        vertline = vertline[:, v_min:]
        vertnorm = vertline / (vertline.sum(axis=1, keepdims=True).astype(float) + self._epsilon)
        valid_entries = vertline > 0
        entr = np.zeros_like(vertline)
        entr[valid_entries] = -(vertnorm[valid_entries] * np.log(vertnorm[valid_entries]))

        return entr.sum(axis=1)

    def time_resolved_max_white_vertlength(self, scale=None, order=2):
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)
        return white_vertline.argmax(axis=1)

    def time_resolved_average_white_vertlength(self, w_min=2, scale=None, order=2):
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)

        if w_min > white_vertline.shape[1]:
            return np.zeros((self.N,))

        partial_sum = (np.arange(w_min, white_vertline.shape[1])[None, :] * white_vertline[:, w_min:]).sum(axis=1)
        full_sum = white_vertline[:, w_min:].sum(axis=1)
        return partial_sum / (full_sum.astype(float) + self._epsilon)

    def time_resolved_mean_recurrence_time(self, w_min=2, scale=None, order=2):
        return self.time_resolved_average_white_vertlength(w_min, scale, order)

    def time_resolved_white_vert_entropy(self, w_min=2, scale=None, order=2):
        white_vertline = self.time_resolved_white_vertline_dist(scale, order)

        if w_min > white_vertline.shape[1]:
            # TODO: check this
            return np.zeros((self.N,))

        white_vertline = white_vertline[:, w_min:]
        white_vertnorm = white_vertline / (white_vertline.sum(axis=1, keepdims=True).astype(float) + self._epsilon)
        valid_entries = white_vertline > 0
        entr = np.zeros_like(white_vertline)
        entr[valid_entries] = -(white_vertnorm[valid_entries] * np.log(white_vertnorm[valid_entries]))

        return entr.sum(axis=1)