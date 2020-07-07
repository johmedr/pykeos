import numpy as np
from .._tools import n_ball_volume, n_sphere_area
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


def rule_of_thumb(x: np.ndarray, norm_p=2) -> float:
    n = x.shape[0]
    d = 1
    if len(x.shape) == 2:
        d = x.shape[1]

    std = np.sqrt(x.var(axis=0).mean())


    # version 1
    # return std * ((18 * (2 * np.sqrt(np.pi)) ** d)/ ((d + 2) * n)) ** (1/(d+4))
    # version 2
    return std * (((d + 2)**2 * (2*np.sqrt(np.pi))**d) / (n * n_ball_volume(d, norm_p) * (1/2. * d + 1/4. * d**2))) ** (1/(d+4))


def grassberger_proccacia(x: np.ndarray, rvals=None, rmin=None, rmax=None, omit_plateau=True,
                          hack_filter_rvals=None,  nr=20, plot=False, fig=None, show=True, full_output=False):
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
            rvals = np.logspace(- np.log(x.shape[0]) + np.log(x.std()) - 2, 5 + np.log(x.std()), nr, base=np.exp(1))
    # print(rvals)
    csums = np.asarray([corr_sum(x, r) for r in rvals])
    # print(csums)
    orig_log_csums = np.log(csums[csums > 0])
    orig_log_rvals = np.log(rvals[csums > 0])

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
        fig.add_trace(go.Scatter(x=orig_log_rvals, y=poly[1] + orig_log_rvals * poly[0],
                                 name="%.2f log(r) + %.2f"%(poly[0], poly[1]), line=dict(dash="dash"),
                                 marker=dict(symbol="x-thin")))

        if show:
            fig.show()

    if full_output:
        return [orig_log_rvals, orig_log_csums, poly]
    else:
        return poly[0]


def approximate_corr_dim(x: np.ndarray, r_opt: float = None, norm_p=1, full_output=False, plot=False, fig=None, show=True):

    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    dim = x.shape[1]
    n_points = x.shape[0]

    if r_opt is None:
        r_opt = rule_of_thumb(x, norm_p=norm_p)
        # print("selecting r_opt=%.3f"%r_opt )

    rho_x = dsty_est(x, samples=x, r=r_opt, norm_p=norm_p) - 1 / n_points
    if not full_output:
        return (r_opt / corr_sum(x, r_opt, norm_p=norm_p)) * (np.sum(rho_x) * n_sphere_area(dim - 1, norm=norm_p) * r_opt**(dim - 1) / n_points)
    else:
        raise NotImplementedError

# def corr_dim(x):
#