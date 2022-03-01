from ..tools import reference_rule

from scipy.special import erf
from scipy.signal import correlate
from scipy.stats import multivariate_normal

import numpy as np


def autocov(x, axis=-1):
    if len(x.shape) > 1:
        return np.apply_along_axis(autocov, axis, x)
    else:
        return correlate(x, x, mode='full', method='auto')[-x.size:] / x.size


class ParametricRQA:
    def __init__(self, x, axis=-1, threshold=None, lmin=2, vmin=2):
        self.x = x
        self.axis = axis
        if threshold is None:
            threshold = reference_rule(x)
        self.threshold = threshold
        self.lmin = lmin
        self.vmin = vmin
        self._internals = dict(diag_probs=dict(), vert_probs=dict())

    def _get_autocov(self):
        try:
            return self._internals['autocov']
        except KeyError:
            self._internals['autocov'] = autocov(self.x, self.axis)
            return self._internals['autocov']

    def _get_diag_prob(self, k):
        try:
            return self._internals['diag_probs'][k]
        except KeyError:
            self._internals['diag_probs'][k] = self._compute_diag_prob(k)
            return self._internals['diag_probs'][k]

    def _get_vert_prob(self, k):
        try:
            return self._internals['vert_probs'][k]
        except KeyError:
            self._internals['vert_probs'][k] = self._compute_vert_prob(k)
            return self._internals['vert_probs'][k]

    def recurrence_rate(self):
        return erf(self.threshold / (2 * self.x.std(ddof=1, axis=self.axis)))

    def determinism(self, lmin=None):
        if lmin is None:
            lmin = self.lmin
        Pk = self._get_diag_prob(lmin)
        Pk_1 = self._get_diag_prob(lmin + 1)
        P1 = self._get_diag_prob(1)
        return (lmin * Pk - (lmin - 1) * Pk_1) / P1

    def avg_diagline(self, lmin=None):
        if lmin is None:
            lmin = self.lmin
        Pk = self._get_diag_prob(lmin)
        Pk_1 = self._get_diag_prob(lmin + 1)
        return (lmin * Pk - (lmin - 1) * Pk_1) / (Pk - Pk_1)

    def laminarity(self, vmin=None):
        if vmin is None:
            vmin = self.vmin
        Tk = self._get_vert_prob(vmin)
        Tk_1 = self._get_vert_prob(vmin + 1)
        T1 = self._get_vert_prob(1)
        return (vmin * Tk - (vmin - 1) * Tk_1) / T1

    def avg_vertline(self, vmin=None):
        if vmin is None:
            vmin = self.vmin
        Tk = self._get_vert_prob(vmin)
        Tk_1 = self._get_vert_prob(vmin + 1)
        return (vmin * Tk - (vmin - 1) * Tk_1) / (Tk - Tk_1)

    def rqa_summary(self, lmin=None, vmin=None):
        return dict(
            rec=self.recurrence_rate(),
            det=self.determinism(lmin),
            lavg=self.avg_diagline(lmin),
            lam=self.laminarity(vmin),
            vavg=self.avg_vertline(vmin)
        )

    def _compute_diag_prob(self, k):
        axis = self.axis
        autocov = self._get_autocov()
        epsilon = self.threshold

        if axis != -1:
            autocov = np.swapaxes(autocov, axis, -1)

        r = np.repeat(np.arange(k)[:, None], k, axis=1)
        s = r.T
        idxs = r - s
        q = autocov.shape[-1] - k

        c1 = np.take(autocov, np.abs(idxs), axis=-1)
        c2 = np.take(autocov, np.abs(idxs + q), axis=-1)
        c3 = np.take(autocov, np.abs(idxs - q), axis=-1)

        cov = 2 * c1[..., :, :] - c2 - c3

        eps = np.repeat(epsilon, k)

        shape = cov.shape[:-2]
        cov = cov.reshape((-1, k, k))
        cov = (cov + np.swapaxes(cov, 1, 2)) / 2

        prob = []
        for _cov in cov:
            try:
                res = multivariate_normal.cdf(eps, cov=_cov)
            except ValueError:
                res = np.nan
            except np.linalg.LinAlgError:
                res = np.nan
            prob.append(res)

        if len(prob) > 1:
            prob = np.asarray(prob).reshape(shape)

            if axis != -1:
                prob = np.swapaxes(prob, -1, axis)
        else:
            prob = prob[0]

        return prob

    def _compute_vert_prob(self, k):
        axis = self.axis
        autocov = self._get_autocov()
        epsilon = self.threshold

        if axis != -1:
            autocov = np.swapaxes(autocov, axis, -1)

        r = np.repeat(np.arange(k)[:, None], k, axis=1)
        s = r.T
        idxs = r - s

        # todo: decrement q iterativelly until cov is SPD
        q = autocov.shape[-1] - k

        c1 = autocov[..., 0]
        # Note: in the two following lines, s - 1 and r - 1 from the paper is replaced
        # with s and r as they range from 0 to k-1 instead of 1 to k. This does not change
        # anything for the difference r - s
        c2 = np.take(autocov, np.abs(q - s), axis=-1)
        c3 = np.take(autocov, np.abs(r - q), axis=-1)
        c4 = np.take(autocov, np.abs(idxs), axis=-1)

        cov = c1 - c2 - c3 + c4

        eps = np.repeat(epsilon, k)

        shape = cov.shape[:-2]
        cov = cov.reshape((-1, k, k))
        cov = (cov + np.swapaxes(cov, 1, 2)) / 2

        prob = []
        for _cov in cov:
            try:
                res = multivariate_normal.cdf(eps, cov=_cov)
            except ValueError:
                res = np.nan
            except np.linalg.LinAlgError:
                res = np.nan
            prob.append(res)

        if len(prob) > 1:
            prob = np.asarray(prob).reshape(shape)

            if axis != -1:
                prob = np.swapaxes(prob, -1, axis)
        else:
            prob = prob[0]

        return prob
