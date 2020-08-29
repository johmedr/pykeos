import numpy as np


from .._systems import ContinuousSys
from pykeos.tools import nd_rand_init


class Lorenz(ContinuousSys):
    def __init__(self, sigma=10., beta=8. / 3, rho=28., n_points=10000, t_min=0, t_max=100):
        def ode(X, t):
            _x, _y, _z = X
            return np.asarray([
                sigma * (_y - _x),
                _x * (rho - _z) - _y,
                _x * _y - beta * _z
            ])

        def rand_init():
            return nd_rand_init([-20, 20], [-30, 30], [10, 40])

        super().__init__(dim=3, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)

    # def observe(self, omit=0, axis=0):
    #     if self._last_ts is None:
    #         self.integrate()
    #     return self._last_ts[omit:, axis]
    #
    # def delay_coordinates(self, tau=None, axis=0):
    #     if tau is None:
    #         tau = int(0.15 * self.sampling_rate())
    #
    #     ts = self.observe(axis=0)
    #
    #     return np.vstack([ts[:-2 * tau], ts[tau:-tau], ts[2 * tau:]]).T
    #
    # def sampling_rate(self):
    #     return float(self._timevec.size) / (self._timevec[-1] - self._timevec[0])
