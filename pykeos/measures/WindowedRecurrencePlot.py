from pyunicorn.timeseries import RecurrencePlot
from ..tools.io_conversion import _make_array, _from_pyunicorn_metric
from ..tools.corr_utils import reference_rule
from tqdm import tqdm
import numpy as np
import functools


class WindowedRecurrencePlot:
    def __init__(self, x, window_size, n_overlap=None, time_axis=None, state_axis=None, **kwargs):
        x = _make_array(x)
        self._time_axis = time_axis
        self._state_axis = state_axis

        self._is_ndim = False
        if len(x.shape) > 2 or (len(x.shape) == 2 and state_axis == None):
            assert(time_axis is not None)
            self._is_ndim = True

        if time_axis is not None:

            self._original_shape = x.shape
            
            if state_axis is None: 
                self._has_state_dim = False
                x = np.swapaxes(x, time_axis, -1)
                x = np.expand_dims(x, -1)
            else:
                self._has_state_dim = True
                x = np.moveaxis(x, (time_axis, state_axis), (-2, -1))

            self._unrelated_shape = x.shape[:-2]
            self._related_shape = x.shape[-2:]

            x = x.reshape((-1, *self._related_shape))

        else: 
            assert(state_axis is None)
            x = np.expand_dims(x, 0)

        self._window_size = window_size
        if n_overlap is None:
            n_overlap = window_size // 2
        self._n_overlap = n_overlap
        self._step_size = window_size - n_overlap
        self._n_windows = x.shape[-2] // self._step_size

        self._time_series = [
            _x[i * self._step_size:min(x.shape[-2]+1, i * self._step_size + self._window_size), :]
            for _x in x for i in range(self._n_windows) 
        ]

        if not any(
            k in ('threshold', 'threshold_std', 'recurrence_rate', 'local_recurrence_rate', 'local_recurrence_rate') 
                for k in kwargs.keys()): 

            norm_p = _from_pyunicorn_metric(kwargs.pop('metric')) if 'metric' in kwargs else _from_pyunicorn_metric('supremum')

            self._rps = [RecurrencePlot(ts, threshold=reference_rule(ts, norm_p=norm_p), silence_level=10, **kwargs)
                for ts in tqdm(self._time_series)]

        else: 
            self._rps = [RecurrencePlot(ts, silence_level=10, **kwargs)
                for ts in tqdm(self._time_series)
            ]

        for name in dir(RecurrencePlot):
            method = getattr(RecurrencePlot, name)
            if callable(method) and name[0] != '_':
                self._bind_rps_method(name, method)

    def _bind_rps_method(self, name, method):
        @functools.wraps(method)
        def _wrapped_func(*args, **kwargs): 
            results = []
            for rp in tqdm(self._rps):
                results.append(method(rp, *args, **kwargs))
            results = np.asarray(results)
            if self._is_ndim:
                if self._has_state_dim: 
                    results = results.reshape((*self._unrelated_shape, -1, self._related_shape[1]))
                    results = np.moveaxis(results, (-1, -2), (self._state_axis, self._time_axis))
                else:
                    results = results.reshape((*self._unrelated_shape, -1))
                    results = np.swapaxes(results, -1, self._time_axis)
                        
                if isinstance(results.flat[0], dict): 
                    unpack = lambda key: np.vectorize(lambda _: _[key])
                    results = {k: unpack(k)(results) for k in results.flat[0].keys()}
            else: 
                results = results[0]

            return results

        setattr(self, method.__name__, _wrapped_func)

    