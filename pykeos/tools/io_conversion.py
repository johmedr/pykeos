from ..systems import AbstractBaseSys, SysWrapper
import pandas as pd
import numpy as np
import mne
from typing import Union


def to_data_frame(sys: AbstractBaseSys, columns=None) -> pd.DataFrame:
    return pd.DataFrame(data=sys.states, index=sys.time_vec, columns=columns)


def from_data_frame(df: pd.DataFrame) -> SysWrapper:
    return SysWrapper(df, dim=len(df.columns), n_points=len(df.index), time_vec=np.asarray(df.index))


def to_pandas_series(sys: AbstractBaseSys, axis: int = 0) -> pd.Series:
    if sys.dim > 1:
        return pd.Series(data=sys.states[:, axis], index=sys.time_vec)
    else:
        return pd.Series(data=sys.states, index=sys.time_vec)


def from_pandas_series(ts: pd.Series) -> SysWrapper:
    return SysWrapper(ts, dim=1, n_points=len(ts.index), time_vec=np.asarray(ts.index))


def to_array(sys: AbstractBaseSys) -> np.ndarray:
    return sys.states


def from_array(a: np.ndarray, *args, **kwargs):
    return SysWrapper(a, *args, **kwargs)


def from_mne_raw(raw: mne.io.BaseRaw) -> SysWrapper:
    return SysWrapper(raw.get_data().T, time_vec=np.asarray(raw.times))


def _make_array(ts_or_sys: Union[np.ndarray, AbstractBaseSys]) -> np.ndarray:
    return ts_or_sys if isinstance(ts_or_sys, np.ndarray) else to_array(ts_or_sys)


_pyunicorn_np_map = {'manhattan':1, 'euclidean':2, 'supremum':float('inf')}
_np_pyunicorn_map = {v:k for k, v in _pyunicorn_np_map.items()}

_pyunicorn_np_map.update({v:v for v in _pyunicorn_np_map.values()})
_np_pyunicorn_map.update({v:v for v in _np_pyunicorn_map.values()})


def _to_pyunicorm_metric(norm_p):
    return _np_pyunicorn_map[norm_p]


def _from_pyunicorn_metric(metric):
    return _pyunicorn_np_map[metric]
