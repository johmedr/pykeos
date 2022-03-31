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
    if len(ts.shape) == 1:
        ts = ts[:, None]
        axis = 0

    if lag == "auto":
        lag = select_embedding_lag(ts, axis=axis, *args, **kwargs)
    if lag * dim > ts.shape[0]:
        raise ValueError

    data = np.vstack([ts[i * lag:(i - dim) * lag, axis] for i  in range(dim)]).T


    if return_array:
        return data
    else:
        return SysWrapper(data)


def select_embedding_lag(
        data: Union[np.ndarray, AbstractBaseSys, pd.Series],
        axis: int = 0,
        lag_range: Union[int, tuple] = (1, 500),
        method: str = "mi",
        criterion: Union[float, str] = 'firstmin',
        interactive: bool = False,
        plot: bool = False
    ) -> int:

    assert(method in ['acf', 'mi'])
    if isinstance(criterion, str):
        assert(criterion == 'firstmin')

    if interactive:
        raise NotImplementedError()

    ts = None
    if method == 'acf': # convert to pandas
        if isinstance(data, np.ndarray):
            ts = pd.Series(data[:,axis])
        elif isinstance(data, AbstractBaseSys):
            from ..tools.io_conversion import to_pandas_series
            ts = to_pandas_series(data, axis=axis)
        elif isinstance(data, pd.Series):
            ts = data.copy()
    else: # convert to np
        if isinstance(data, np.ndarray):
            if len(data.shape) > 1:
                data = data[:, axis]
            ts = np.asarray(data)
        elif isinstance(data, AbstractBaseSys):
            ts = data.states[:, axis]
        elif isinstance(data, pd.Series):
            ts = data.to_numpy(copy=True)


    lag_range = (lag_range[0], min(lag_range[1], ts.size) - 1)
    dcurve = None
    ref = None
    lag = -1

    if method == "acf":
        ref = ts.autocorr(0)
        dcurve = [ts.autocorr(lag) for lag in range(*lag_range)]
    elif method == "mi":
        ref = lagged_mi(ts, 0)
        dcurve = [lagged_mi(ts, lag) for lag in range(*lag_range)]

    if isinstance(criterion, str):
        _dcurve = [dcurve[i+1] - dcurve[i] for i in range(len(dcurve) - 1)]

        for i in range(len(_dcurve) - 1):
            if _dcurve[i] < 0 and _dcurve[i + 1] > 0:
                lag = i if abs(dcurve[i]) < abs(dcurve[i + 1]) else i + 1
                break
    else:
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







