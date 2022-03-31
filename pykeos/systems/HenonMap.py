from ..systems import DiscreteSys
from ..tools import nd_rand_init

import numpy as np


class HenonMap(DiscreteSys):
    def __init__(self, a=1.4, b=0.3, n_points=100):
        def map_func(X):
            x, y = X
            return np.asarray([1. - a * x**2 + y, b * x])

        def rand_init():
            return nd_rand_init((-0.5, 0.5), (-0.5, 0.5))

        super().__init__(dim=2, map_func=map_func, init_func=rand_init, n_points=n_points)
