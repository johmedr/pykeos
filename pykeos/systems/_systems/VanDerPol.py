from .._systems import ContinuousSys
from pykeos.tools import nd_rand_init

import numpy as np


class VanDerPol(ContinuousSys):
    def __init__(self, mu=2., n_points=1000, t_min=0, t_max=100):
        def ode(X, t):
            x, y = X
            return np.asarray([
                y,
                (1. - x**2) * mu * y - x
            ])

        def rand_init():
            return nd_rand_init((-2, 2), (-2, 2))

        super().__init__(dim=2, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)

