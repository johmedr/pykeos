from .._systems import DiscreteSys
from pykeos.tools import nd_rand_init

import numpy as np


class LogisticMap(DiscreteSys):
    def __init__(self, r=4, n_points=100):
        def map_func(x):
            return r * x * (1 - x)

        def rand_init():
            return nd_rand_init((0, 1))

        super().__init__(dim=1, map_func=map_func, init_func=rand_init, n_points=n_points)
