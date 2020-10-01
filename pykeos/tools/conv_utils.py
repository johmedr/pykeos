from ..systems import AbstractBaseSys, SysWrapper
import pandas as pd
import numpy as np


def to_data_frame(sys: AbstractBaseSys, columns = None) -> pd.DataFrame:
    return pd.DataFrame(data=sys.states, index=sys.time_vec, columns=columns)


def from_data_frame(df: pd.DataFrame) -> SysWrapper:
    return SysWrapper(df, dim=len(df.columns), n_points=len(df.index), time_vec=np.asarray(df.index))


def to_pandas_series(sys: AbstractBaseSys) -> pd.Series:
    assert(sys.dim == 1)
    return pd.Series(data=sys.states, index=sys.time_vec)


def from_pandas_series(ts: pd.Series) -> SysWrapper:
    return SysWrapper(ts, dim=1, n_points=len(ts.index), time_vec=np.asarray(ts.index))


def to_array(sys: AbstractBaseSys) -> np.ndarray:
    return sys.states


def from_array(a: np.ndarray, *args, **kwargs):
    return SysWrapper(a, *args, **kwargs)

