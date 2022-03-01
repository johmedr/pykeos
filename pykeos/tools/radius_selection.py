from ..tools.nd_utils import nd_function
from ..tools.math_utils import n_ball_volume
from typing import Union

from scipy.special import gamma
from scipy import stats

import numpy as np


def reference_rule_alpha(p: Union[float, int], d: int):
    if d == 1:
        alpha = (12. * np.sqrt(np.pi)) ** (1./5)
    else:
        alpha = (
            (4. * (2. * np.sqrt(np.pi))**d * (3. * gamma(1+(d+2.)/p) * gamma(1+1./p))**2)
            / ((d + 2.) * n_ball_volume(d, p) * (gamma(1+3./p) * gamma(1.+d/p)) ** 2)
        ) ** (1./(d+4))
    return alpha


@nd_function
def reference_rule(x: np.ndarray, dim: Union[int, str] = 'auto', norm_p: Union[int, float, str] = 2) -> float:
    n = x.shape[0]
    if dim == 'auto':
        d = x.shape[1] if len(x.shape) == 2 else 1
    elif isinstance(dim, int):
        d = dim
    else:
        raise ValueError(f'dim must be "auto" or int. Got: {type(dim)} ({dim})')

    std = np.sqrt(x.var(axis=0, ddof=1).mean())
    iqr = stats.iqr(x)
    scale = min(std, iqr/1.34)

    gamma_n = n ** (-1/(d+4))

    if norm_p in ['manhattan', 'euclidean', 'supremum']:
        norm_p = ['manhattan', 'euclidean'].index(norm_p) + 1 if norm_p != 'supremum' else float('inf')

    alpha_p_d = reference_rule_alpha(norm_p, d)
    return gamma_n * alpha_p_d * scale

