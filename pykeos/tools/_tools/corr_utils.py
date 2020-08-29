import numpy as np
from .._tools import n_ball_volume, n_sphere_area, delay_coordinates
from nolds.measures import poly_fit
from tqdm import tqdm

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


def rule_of_thumb(x: np.ndarray, norm_p=2, version: str = 'normal') -> float:
    n = x.shape[0]
    d = 1
    if len(x.shape) == 2:
        d = x.shape[1]

    if norm_p in ['manhattan', 'euclidean', 'supremum']:
        norm_p = ["manhattan", "euclidean"].index(norm_p) + 1 if norm_p != "supremum" else float("inf")

    std = np.sqrt(x.var(axis=0, ddof=1).mean())


    # version 1
    if version == 'normal':
        return std * ((9. * n_ball_volume(d, norm_p) * (2 * np.sqrt(np.pi)) ** d)/ ((d + 2) * n)) ** (1/(d+4))
    elif version == 'scott':
        return std * 3.5 * n ** (-1 / (d + 2))
    # version 2
    # return std * (((d + 2)**2 * (2*np.sqrt(np.pi))**d) / (n * n_ball_volume(d, norm_p) * (1/2. * d + 1/4. * d**2))) ** (1/(d+4))


def grassberger_proccacia(x: np.ndarray, rvals=None, rmin=None, rmax=None, omit_plateau=True,
                          hack_filter_rvals=None,  nr=20, plot=False, fig=None, show=True, full_output=False, log_base=10, remove_tail=True, verbose=True):
    """
    Estimates the correlation dimension using the Grassberger-Proccacia algorithm. The code is greatly inspired by
    nolds: https://github.com/CSchoel/nolds/blob/master/nolds/measures.py and makes use of nolds version of poly_fit
    :param x: time-series (n_points, dim) or (n_points, )
    :param rvals: the threshold values to use
    :return: the correlation dimension
    """
    if log_base == 10:
        log = np.log10
    elif log_base == np.exp(1):
        log = np.log
    else:
        log = lambda x: np.log(x)/np.log(log_base)

    if rvals is None:
        if rmin is not None and rmax is not None:
            rvals = np.logspace(rmin, rmax, nr, base=log_base)
        else:
            rvals = np.logspace(- log(x.shape[0]) + log(x.std()) - 2, 5 + log(x.std()), nr, base=log_base)
    # print(rvals)
    csums = np.asarray([corr_sum(x, r) for r in tqdm(rvals, desc="Computing correlation sums")])
    # print(csums)
    orig_log_csums = log(csums[csums > 0])
    orig_log_rvals = log(rvals[csums > 0])

    log_csums = np.asarray(orig_log_csums)
    log_rvals = np.asarray(orig_log_rvals)

    if remove_tail:
        filter = log_csums > -log(x.shape[0])
        log_csums = log_csums[filter]
        log_rvals = log_rvals[filter]

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


def corrdim_tangent_approx(x: np.ndarray, r_opt: float = None, norm_p=1, r_opt_ratio: float = 0.1, base: float = 10.0, full_output=False):

    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    dim = x.shape[1]
    n_points = x.shape[0]

    if r_opt is None:
        r_opt = rule_of_thumb(x, norm_p=norm_p)

    if base == 10:
        log = np.log10
    elif base == np.e:
        log = np.log
    else:
        raise AttributeError()

    r1 = r_opt
    r2 = r_opt * base**(-r_opt_ratio)
    # print(log(r1))
    # print(log(r2))
    c1 = corr_sum(x,  r1, norm_p=norm_p, allow_equals=False)
    c2 = corr_sum(x, r2, norm_p=norm_p, allow_equals=False)
    alpha = (log(c1) - log(c2)) / r_opt_ratio
    if full_output:
        r0 = log(r1)
        beta = log(c1) - alpha * r0

        return alpha, beta, r0
    else:
        return alpha


def approximate_k2(x: np.ndarray, r='auto', max_l=15, full_output=False, mrange: tuple=None):
    try:
        from pyunicorn.timeseries import RecurrencePlot
    except ImportError:
        raise ImportError('The pyunicorn module is required to approximate K2. ')

    if r == 'auto':
        r = rule_of_thumb(x, norm_p=float('inf'))
    if max_l is None:
        max_l = -1
        
    N_l = RecurrencePlot(x, threshold=r, silence_level=3).diagline_dist()[:max_l]
    D_l = np.zeros_like(N_l, dtype=float)

    if mrange is None:
        mrange = (2 + 1 , max_l - 1)

    for i in range(mrange[0], mrange[1]):
        if np.log(N_l[i]) > - np.inf and np.log(N_l[i + 1]) > - np.inf:
            D_l[i] = np.log(N_l[i]) - np.log(N_l[i + 1])
    D_l = np.nan_to_num(D_l)
    D_l = D_l[D_l != 0]

    if full_output:
        return np.mean(D_l), np.std(D_l, ddof=1) / D_l.size
    else:
        return np.mean(D_l)

# def ks_estimation(x: np.ndarray, dims: list, rvals=None, rmin=None, rmax=None, omit_plateau=True, lag:int=1):
#     if isinstance(dims, int):
#         dims = [dims]
#     for dim in dims:
#         ts0 = delay_coordinates(x, dim, lag)
#         ts1 = delay_coordinates(x, dim + 1, lag)
#
#         corr_sum(ts0, r, norm_p=float('inf'), allow_equals=False)
#         corr_sum(ts1, r, norm_p=float('inf'), allow_equals=False)