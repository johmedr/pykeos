from ..tools import lagged_mi
from ..systems import AbstractBaseSys, SysWrapper

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from typing import Union


def delay_coordinates(
        ts: np.ndarray,
        dim: int,
        lag: Union[int, str] = "auto",
        axis: int = 0,
        return_array: bool = False,
        *args, **kwargs
    ) -> Union[np.ndarray, SysWrapper]:
    if len(ts.shape) > 1:
        ts = ts[:, axis]

    if lag == "auto":
        lag = select_embedding_lag(ts[:, axis] if len(ts.shape) > 1 else ts, *args, **kwargs)

    if len(ts.shape) == 1:
        data = np.vstack([ts[i * lag:(i - dim) * lag] for i in range(dim)]).T
    else:
        data = np.vstack([ts[i * lag:(i - dim) * lag, axis] for i in range(dim)]).T

    if return_array:
        return data
    else:
        return SysWrapper(data)


def select_embedding_lag(
        data: Union[np.ndarray, AbstractBaseSys, pd.Series],
        lag_range: Union[int, tuple] = (2, 500),
        method: str = "acf",
        criterion: float = 1/np.e,
        interactive: bool = False,
        plot: bool = False
    ) -> int:

    assert(method in ['acf', 'mi'])

    if interactive:
        raise NotImplementedError()

    ts = None
    if isinstance(data, np.ndarray):
        ts = pd.Series(data)
    elif isinstance(data, AbstractBaseSys):
        from ..tools.conv_utils import to_pandas_series
        ts = to_pandas_series(data)
    elif isinstance(data, pd.Series):
        ts = data.copy()

    dcurve = None
    ref = None
    lag = -1

    if method == "acf":
        ref = ts.autocorr(0)
        dcurve = [ts.autocorr(lag) for lag in range(*lag_range)]
    elif method == "mi":
        ref = lagged_mi(ts, 0)
        dcurve = [lagged_mi(ts, lag) for lag in range(*lag_range)]

    for i in range(len(dcurve) - 1):
        if dcurve[i] > ref * criterion and dcurve[i + 1] <= ref * criterion:
            lag = i if abs(dcurve[i]) < abs(dcurve[i + 1]) else i + 1
            break

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=dcurve, mode="lines"))
        fig.add_shape(x0=lag, x1=lag, y0=np.min(dcurve), y1=np.max(dcurve), type="line", line_color="indianred")
        fig.show()

    return lag







