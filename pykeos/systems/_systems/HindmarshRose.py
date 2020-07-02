from .._systems import ContinuousSys
from pykeos.tools import nd_rand_init

import numpy as np


class HindmarshRose(ContinuousSys):
    def __init__(self, I=3.25, a=1, b=3, c=1, d=5, r=1e-3, s=4, x_r=-(1 + np.sqrt(5))/2., n_points=5000, t_min=0, t_max=30):

        def ode(X, t):
            x, y, z = X
            return np.asarray([
                y - a * x**3 + b * x**2 - z + I,
                c - d * x**2 - y,
                r * (s * (x - x_r) - z)
            ])

        def rand_init():
            return nd_rand_init((-1, 2), (-1, 1), (-1, 1))

        super().__init__(dim=3, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)

