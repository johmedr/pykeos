# http://thor.physics.wisc.edu/pubs/paper255.pdf
import numpy as np


from .._systems import ContinuousSys
from pykeos.tools import nd_rand_init


class Ueda(ContinuousSys):
    def __init__(self, k=0.05, A=7.5, n_points=10000, t_min=0, t_max=100):
        def ode(X, t):
            _x, _y, _z = X
            return np.asarray([
                _y,
                _x **3 - k * _y + A * np.sin(_z),
                1
            ])

        def rand_init():
            return nd_rand_init([-10, 10], [-10, 10], [-np.pi, np.pi])

        super().__init__(dim=3, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)
