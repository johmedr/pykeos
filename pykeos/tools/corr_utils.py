import numpy as np
from ..tools import n_ball_volume, n_sphere_area, delay_coordinates, lstsqr
from .math_utils import _lstsqr_design_matrix
from scipy.special import gamma
from nolds.measures import poly_fit
from tqdm import tqdm
from typing import Union

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


def reference_rule_alpha(p: Union[float, int], d: int):
    # if d > 1:
    #     return (
    #                    ((d * n_ball_volume(d, p) * (2 * np.sqrt(np.pi))**d) / (d + 2))
    #                  * ((3 * gamma((d + 2) / p + 1))/(2 * gamma((d-1)/p + 1) * gamma(3 / p + 1)))**2
    #            ) ** (1./(d+4))
    # else:
    #     return (12 * np.sqrt(np.pi)) ** (1 / 5.)
    if d == 1:
        return 1.843

    return (
        (4* (2 * np.sqrt(np.pi))**d *(3 * gamma(1+(d+2)/p) * gamma(1+1./p))**2) / ((d + 2) * n_ball_volume(d,p) * (gamma(3/p + 1) * gamma(d/p + 1))**2)
    )**(1/(d+4))


def reference_rule(x: np.ndarray, dim:Union[int, str] = 'auto', norm_p: Union[int, float, str] = 2) -> float:
    n = x.shape[0]
    if dim == 'auto':
        d = 1
        if len(x.shape) == 2:
            d = x.shape[1]

    elif isinstance(dim, int):
        d = dim

    # print(d)
    std = np.sqrt(x.var(axis=0, ddof=1).mean())
    from scipy import stats
    iqr = stats.iqr(x)
    scale = min(std, iqr/1.34)

    gamma_n = n ** (-1/(d+4))

    if norm_p in ['manhattan', 'euclidean', 'supremum']:
        norm_p = ["manhattan", "euclidean"].index(norm_p) + 1 if norm_p != "supremum" else float("inf")
    alpha_p_d = reference_rule_alpha(norm_p, d)
    return gamma_n * alpha_p_d * scale

# def rule_of_thumb(x: np.ndarray, norm_p=2, version: str = 'normal') -> float:
#     n = x.shape[0]
#     d = 1
#     if len(x.shape) == 2:
#         d = x.shape[1]
#
#     if norm_p in ['manhattan', 'euclidean', 'supremum']:
#         norm_p = ["manhattan", "euclidean"].index(norm_p) + 1 if norm_p != "supremum" else float("inf")
#
#     std = np.sqrt(x.var(axis=0, ddof=1).mean())
#
#
#     # version 1
#     if version == 'normal':
#         return std * ((9. * n_ball_volume(d, norm_p) * (2 * np.sqrt(np.pi)) ** d)/ ((d + 2) * n)) ** (1/(d+4))
#     elif version == 'scott':
#         return std * 3.5 * n ** (-1 / (d + 2))
#     # version 2
#     # return std * (((d + 2)**2 * (2*np.sqrt(np.pi))**d) / (n * n_ball_volume(d, norm_p) * (1/2. * d + 1/4. * d**2))) ** (1/(d+4))


def grassberger_proccacia(x: np.ndarray, rvals=None, rmin=None, rmax=None, omit_plateau=True, norm_p=2, method='lstsqr',
                          hack_filter_rvals=None,  nr=20, plot=False, fig=None, show=True, full_output=False,
                          log_base=10, remove_tail=True, verbose=False):
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
    csums = np.asarray([corr_sum(x, r, norm_p=norm_p)
                        for r in (tqdm(rvals, desc="Computing correlation sums") if verbose else rvals)])
    # print(csums)
    orig_log_csums =log(csums[csums > 0])
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

    # poly = poly_fit(log_rvals, log_csums, degree=1)
    if len(log_rvals) == 0:
        raise ValueError('bad estim')

    if method == 'lstsqr':
        poly = lstsqr(_lstsqr_design_matrix(log_rvals), log_csums)
    else:
        poly = np.polyfit(log_rvals, log_csums, deg=1)

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

def approximate_d2(x: object, r_opt: object = None, meaningfull_range: object = (0.5, 1.), n_evals: object = 10, base: object = 10.,
                   norm_p: float = float('inf'),
                   method: object = 'fit',
                   full_output: object = False, output_curve = False) -> float:
    if r_opt is None:
        r_opt = reference_rule(x, norm_p=norm_p)
    if base ==10:
        log = np.log10
    elif base == np.e:
        log = np.log
    else:
        log = lambda x: np.log(x) / np.log(base)

    rvals = np.logspace(log(r_opt * meaningfull_range[0]), log(r_opt*meaningfull_range[1]), n_evals, base=base)
    _csum =lambda r : corr_sum(x, r, norm_p=norm_p)
    csums = np.vectorize(_csum)(rvals)
    csums = log(csums)
    csums = csums[csums == csums]
    rvals = log(rvals)

    if method == 'tangent':
        slopes = []
        for i in range(1, len(rvals)):
            rsup = rvals[i]
            for j in range(i):
                rinf = rvals[j]
                slopes.append((csums[i] - csums[j])/(rsup - rinf))

        return np.mean(slopes)
    elif method == 'fit':
        poly = lstsqr(_lstsqr_design_matrix(rvals), csums)
        # poly = poly_fit(rvals, csums, degree=1)
        if full_output:
            if output_curve:
                return poly[0], poly[1],  log(r_opt), rvals, csums
            else:
                return poly[0], poly[1],  log(r_opt)
        else:
            return poly[0]



def corrdim_tangent_approx(x: np.ndarray, r_opt: float = None, norm_p=1, r_opt_ratio: float = 0.1, base: float = 10.0, full_output=False):

    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    dim = x.shape[1]
    n_points = x.shape[0]

    if r_opt is None:
        r_opt = reference_rule(x, norm_p=norm_p)

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


def approximate_k2(x: np.ndarray=None, r='auto', L_min=None, L_max=None, min_diag_number=0,
                   min_consecutive_nonzero_values=5, split_nonzero_slices=True, take_zero_splitted_slice='all', dt=1, method='avg', rp=None,
                   full_output=False, raise_if_empty=True, ordering=True):
# def approximate_k2(x: np.ndarray, r='auto', L_min=None, max_l=15, full_output=False, mrange: tuple=None, dt=1):
    assert(take_zero_splitted_slice in {'first', 'all'})
    assert(method in {'avg', 'fit'})

    try:
        from pyunicorn.timeseries import RecurrencePlot
    except ImportError:
        raise ImportError('The pyunicorn module is required to approximate K2. ')

    if r == 'auto':
        r = reference_rule(x, norm_p=float('inf'))
    # if max_l is None:
    #     max_l = -1

    if rp is None:
        if x is None:
            raise AttributeError("Either x or a pyunicorn RecurrencePlot must be supplied")
        rp = RecurrencePlot(x, threshold=r, silence_level=3, metric="supremum")

    if L_max is None:
        L_max = rp.max_diaglength()
    else:
        L_max = min(L_max, rp.max_diaglength())

    if L_min is not None:
        if isinstance(L_min, str):
            L_min = int(float(L_min.split()[0]) * rp.average_diaglength())

        L_min = min(L_min, 2)
    else:
        L_min = 2


    import itertools
    import operator
    # Entry i corresponds to diagline of size i
    diagline_dist = rp.diagline_dist()[L_min:L_max]
    diagline_dist_slices = []

    if split_nonzero_slices:
        nonzero_diagline_idx = [[i for i,value in it]
                                for key,it in itertools.groupby(enumerate(diagline_dist > min_diag_number), key=operator.itemgetter(1))
                                if key != 0]
        if take_zero_splitted_slice == 'all':
            diagline_dist_slices = [diagline_dist[s] for s in nonzero_diagline_idx
                                    if len(s) > min_consecutive_nonzero_values]
        elif take_zero_splitted_slice == "first":
            for s in nonzero_diagline_idx:
                if len(s) > min_consecutive_nonzero_values:
                    diagline_dist_slices = [diagline_dist[s]]
                    break

    else:
        diagline_dist_slices = [diagline_dist]



    if len(diagline_dist_slices) == 0:
        if raise_if_empty:
            raise ValueError('Cannot find sufficient support to estimate K2. Please check the timeseries for NaN or inf '
                             'values or manually provide a valid recurrence plot.')
        else:
            return np.nan

    k2_list = []

    if method == 'fit':
        x_vals = []
        y_vals = []
        for N_l in diagline_dist_slices:
            # bad_vals_filter = N_l > min_diags
            # bad_vals_filter[1:] = np.logical_and(bad_vals_filter[1:], N_l[:-1] - N_l[1:] != 0)
            # bad_vals_filter = N_l[:-1] - N_l[1:] != 0
            x_vals.append(np.arange(N_l.shape[0])[::-1])
            # x_vals = x_vals[bad_vals_filter]

            # N_l = N_l[bad_vals_filter]
            y_vals.append(np.log(N_l))
            # N_l = np.log(N_l)
        x_vals = np.concatenate(x_vals)
        y_vals = np.concatenate(y_vals)

        poly = lstsqr(_lstsqr_design_matrix(x_vals), y_vals)
        k2_list.append(poly[0])

    elif method == 'avg':
        # D_l = np.zeros((np.sum((len(s) for s in diagline_dist_slices)) - 1, ), dtype=float)

        for N_l in diagline_dist_slices:
            # D_l = np.zeros((N_l.shape[0]-1,), dtype=float)
            D_l = []
            for i in range(N_l.shape[0] - 1):
                if N_l[i + 1] != 0 and N_l[i] != 0:
                    if ordering:
                        # li = np.log(N_l[i])
                        # lip1 = np.log(N_l[i+1])

                        # D_l[i] = li - lip1
                        # D_l.append(li - lip1)
                        if  N_l[i+1] <= N_l[i]:
                            D_l.append(np.log(N_l[i] / N_l[i+1]))
                    else:
                        D_l.append(np.log(N_l[i] / N_l[i+1]))
            # D_l = np.nan_to_num(D_l)
            # D_l = D_l[D_l == D_l]

            k2_list.append(np.mean(D_l))

    if full_output:
        return [k2 / dt for k2 in k2_list]
    else:
        return np.mean(k2_list) / dt


