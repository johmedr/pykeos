from .._systems import ContinuousSys
from pykeos.tools import nd_rand_init

import numpy as np


class Rossler(ContinuousSys):
    def __init__(self, a=0.1, b=0.1, c=14, n_points=2000, t_min=0, t_max=100):
        def ode(X, t):
            x,y,z = X
            return np.asarray([
                - y - z,
                x + a * y,
                b + z * (x - c)
            ])

        def rand_init():
            return nd_rand_init([-10,10],[-10,10],[0,25])

        super().__init__(dim=3, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)
