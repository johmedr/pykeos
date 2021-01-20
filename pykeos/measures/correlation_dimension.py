from ..tools import reference_rule, correlation_sum
from ..tools.math_utils import _lstsqr_design_matrix
from ..tools.conv_utils import _make_array

import numpy as np


def correlation_dimension(x, radius=None, radius_range=(0.5, 1), use_relative_range=True,
                          n_radius=10, log_base=np.e, norm_p=float('inf'), debug_plot=False):

    _x = _make_array(x)

    if radius is None:
        radius = reference_rule(_x, norm_p=norm_p)

    if log_base == 10:
        log = np.log10
    elif log_base == np.e:
        log = np.log
    else:
        log = lambda x: np.log(_x) / np.log(log_base)

    if use_relative_range:
        absolute_range = list(b *  radius for b in radius_range)
    else:
        absolute_range = list(radius_range)
    absolute_range = [log(r) for r in absolute_range]

    radius_values = np.logspace(absolute_range[0], absolute_range[1], n_radius, base=log_base)
    corr_sums = np.vectorize(lambda r: correlation_sum(_x, r, norm_p=norm_p))(radius_values)

    radius_values = log(radius_values[corr_sums != 0])
    corr_sums = log(corr_sums[corr_sums != 0])

    poly = np.linalg.lstsq(_lstsqr_design_matrix(radius_values), corr_sums, rcond=None)
    poly = poly[0]
    corr_dim = poly[0]

    if debug_plot:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=radius_values, y=corr_sums, name="log(C(r)) vs log(r)"))
        fig.add_trace(go.Scatter(x=radius_values, y=poly[1] + radius_values * poly[0],
                                 name="%.2f log(r) + %.2f"%(poly[0], poly[1]), line=dict(dash="dash"),
                                 marker=dict(symbol="x-thin")))
        fig.show()

    return corr_dim

