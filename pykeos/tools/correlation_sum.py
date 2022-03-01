from .conv_utils import _make_array
import numpy as np
from typing import Union


def _fast_count_row(x, traj, r, norm_p):
    """
    :param x: (dim, )
    :param traj: (n_points, dim)
    :param norm_p: int, float('inf')
    :return:
    """
    return np.count_nonzero(np.linalg.norm(traj - x[np.newaxis, :], ord=norm_p, axis=1) <= r)


def _corr_sum_numpy_backend(x: np.ndarray, r: float, norm_p: float) -> float:
    return np.sum(
        np.apply_along_axis(_fast_count_row, 1, x, x, r, norm_p)).astype(np.float64) / (x.shape[0]**2)


def _corr_sum_pyunicorn_backend(x: np.ndarray, r: float, norm_p: Union[float, str]) -> float:
    import pyunicorn.timeseries as puts
    from .conv_utils import _to_pyunicorm_metric

    return puts.RecurrencePlot(
        x, threshold=r, metric=_to_pyunicorm_metric(norm_p), silence_level=3
    ).recurrence_rate()


try:
    import pyunicorn
    _corr_sum_backend = _corr_sum_pyunicorn_backend
except ImportError:
    print("Cannot import pyunicorn backend. Custom Numpy backend will be use instead (could be"
          "slower.")
    _corr_sum_backend = _corr_sum_numpy_backend


def correlation_sum(x, r=None, norm_p=float('inf'), allow_equals=False) -> float:
    _x = _make_array(x)
    csum = _corr_sum_backend(x, r, norm_p)
    if not allow_equals:
        N = _x.shape[0]
        csum = (csum * N - 1)/(N-1)
    return csum