import numpy as np
from .._tools import n_ball_volume
from nolds.measures import poly_fit

import plotly.graph_objs as go


def _fast_count_row_1d(x, traj, r, norm_p):
    return np.count_nonzero(traj - x[np.newaxis, :] <= r)


def _fast_count_row(x, traj, r, norm_p):
    """
    :param x: (dim, )
    :param traj: (n_points, dim)
    :param norm_p: int, float('inf')
    :return:
    """
    return np.count_nonzero(np.linalg.norm(traj - x[np.newaxis, :], ord=norm_p, axis=1) <= r)


def _fast_count_traj(x, r, norm_p):
    """
    :param x: (n_points, dim)
    :param norm_p: int, float('inf')
    :return:
    """
    if x.shape[1] > 1:
        return np.sum(np.apply_along_axis(_fast_count_row, 1, x, x, r, norm_p))
    elif x.shape[1] == 1:
        return np.sum(np.apply_along_axis(_fast_count_row_1d, 1, x, x, r, norm_p))


def corr_sum(traj, r, norm_p=1, allow_equals=False):
    if allow_equals:
        return _fast_count_traj(traj, r, norm_p).astype(np.float64) / traj.shape[0] ** 2
    else:
        return (_fast_count_traj(traj, r, norm_p).astype(np.float64) - traj.shape[0]) /\
               (traj.shape[0] * (traj.shape[0] - 1))

def dsty_est(x, samples, r, norm_p=1):
    if len(samples.shape) == 1:
        samples = samples[:, np.newaxis]

    dim = samples.shape[1]

    # x is (k,)
    if len(x.shape) == 1:
        # if k = d, then x is a point in phase space
        if x.shape[0] == dim:
            x = x[np.newaxis, :]
        # x is a collection of sample points in 1d phase space
        elif dim == 1:
            x = x[:, np.newaxis]
    elif len(x.shape) == 2:
        assert(x.shape[1] == samples.shape[1])

    return np.apply_along_axis(_fast_count_row, 1, x, samples, r, norm_p).astype(np.float64) / (n_ball_volume(dim, norm_p) * r**dim * samples.shape[0])


def rule_of_thumb(x: np.ndarray) -> float:
    n = x.shape[0]
    d = 1
    if len(x.shape) == 2:
        d = x.shape[1]

    std = x.std(axis=0) if d == 1 else np.sqrt(np.trace(np.cov(x.T)))

    return std * ((18 * (2 * np.sqrt(np.pi)) ** d)/ ((d + 2) * n)) ** (1/(d+4))


def grassberger_proccacia(x: np.ndarray, rvals=None, rmin=None, rmax=None, omit_plateau=True, hack_filter_rvals=None,  nr=20, plot=False, fig=None, show=True) -> float:
    """
    Estimates the correlation dimension using the Grassberger-Proccacia algorithm. The code is greatly inspired by
    nolds: https://github.com/CSchoel/nolds/blob/master/nolds/measures.py and makes use of nolds version of poly_fit
    :param x: time-series (n_points, dim) or (n_points, )
    :param rvals: the threshold values to use
    :return: the correlation dimension
    """
    if rvals is None:
        if rmin is not None and rmax is not None:
            rvals = np.linspace(rmin, rmax, nr)

        else:
            rvals = np.logspace(- np.log10(x.shape[0]) + np.log10(x.std()), 1 + np.log10(x.std()), nr)
    # print(rvals)
    csums = np.asarray([corr_sum(x, r) for r in rvals])
    # print(csums)
    orig_log_csums = np.log10(csums[csums > 0])
    orig_log_rvals = np.log10(rvals[csums > 0])

    log_csums = np.asarray(orig_log_csums)
    log_rvals = np.asarray(orig_log_rvals)

    if hack_filter_rvals is not None:
        log_rvals = log_rvals[hack_filter_rvals]
        log_csums = log_csums[hack_filter_rvals]

    if omit_plateau:
        filter = np.zeros_like(log_rvals, dtype=np.bool)
        for i in range(log_rvals.size - 1):
            delta = log_csums[i+1] - log_csums[i]
            if delta > 0:
                filter[i] = True

        log_rvals = log_rvals[filter]
        log_csums = log_csums[filter]

    poly = poly_fit(log_rvals, log_csums, degree=1)

    if plot:
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(x=orig_log_rvals, y=orig_log_csums, name="log(C(r)) vs log(r)"))
        fig.add_trace(go.Scatter(x=orig_log_rvals, y=poly[1] + log_rvals * poly[0],
                                 name="%.2f log(r) + %.2f"%(poly[0], poly[1])))

        if show:
            fig.show()

    return poly[0]


# def corr_dim(x):
#