import joblib
import numpy as np
import functools

from ..tools.conv_utils import _make_array


def nd_function(func): 
    @functools.wraps(func)
    def _wrapped_func(x, *args, **kwargs):
        if 'axis' in kwargs.keys(): 
            if not isinstance(kwargs['axis'], int):
                raise ValueError('Axis argument must have integer type.')

            axis = kwargs.pop('axis')
            if 'keepdims' in kwargs.keys(): 
                keepdims = kwargs.pop('keepdims')
            else: 
                keepdims = False


            _x = _make_array(x)

            _x = np.swapaxes(_x, axis, -1)
            _shape = _x.shape
            _x = _x.reshape((-1, _x.shape[-1]))

            results = joblib.Parallel(-1)(
                joblib.delayed(func)(_x[i], *args, **kwargs) 
                for i in range(_x.shape[0]))


            result_shape = np.shape(results[0])
            if len(result_shape) == 0:
                result_shape = (1,)

            results = np.asarray(results)

            if result_shape[0] == 1 and len(result_shape) == 1 and not keepdims: 
                results = results.reshape((*_shape[:-1],))
            else: 
                results = results.reshape((*_shape[:-1], *result_shape))
                results = np.swapaxes(results, axis, -1)

            return results

        else: 
            return func(_x, *args, **kwargs)

    return _wrapped_func

def windowed_function(func): 
    @functools.wraps(func)
    def _wrapped_func(x, *args, **kwargs):
        if 'window_size' in kwargs.keys(): 
            if not isinstance(kwargs['window_size'], int):
                raise ValueError('window size argument must have integer type.')
            window_size = kwargs.pop('window_size')

            if 'n_overlap' in kwargs.keys():

                if not isinstance(kwargs['n_overlap'], int):
                    raise ValueError('window size argument must have integer type.')
                n_overlap = kwargs.pop('n_overlap')
            else: 
                n_overlap = window_size // 2

            _x = _make_array(x)

            step_size = window_size - n_overlap
            n_windows = int(np.ceil(x.shape[0] / step_size))


            results = [
                func(_x[i * step_size:min(i * step_size + window_size, x.shape[0]+1)], *args, **kwargs)
                for i in range(n_windows)
            ]

            results = np.asarray(results)

            return results

        else: 
            return func(_x, *args, **kwargs)

    return _wrapped_func