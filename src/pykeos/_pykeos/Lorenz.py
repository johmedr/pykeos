import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go


class Lorenz(object):
    def __init__(self, X0=None, sigma=10., beta=8. / 3, rho=28., npoints=1000, tmin=0, tmax=20, seed=None):
        self._s = sigma
        self._b = beta
        self._r = rho
        self._x0 = X0
        self._last_ts = None
        self._timevec = np.linspace(tmin, tmax, npoints)

        if seed is not None:
            np.random.seed(seed)

    def ode(self, X, t=None):
        _x, _y, _z = X
        return np.asarray([
            self._s * (_y - _x),
            _x * (self._r - _z) - _y,
            _x * _y - self._b * _z
        ])

    def rand_init(self, Ux=[-20, 20], Uy=[-30, 30], Uz=[10, 40]):
        U = [Ux, Uy, Uz]
        mins = [u[0] for u in U]
        maxs = [u[1] for u in U]
        return np.random.uniform(low=mins, high=maxs)

    def integrate(self, timevec=None, init=True):
        if init:
            self._x0 = self.rand_init()

        if timevec is not None:
            self._timevec = timevec

        self._last_ts = odeint(self.ode, self._x0, self._timevec)
        return self._last_ts

    def observe(self, omit=0, axis=0):
        if self._last_ts is None:
            self.integrate()
        return self._last_ts[omit:, axis]

    def delay_coordinates(self, tau=None, axis=0):
        if tau is None:
            tau = int(0.15 * self.sampling_rate())

        ts = self.observe(axis=0)

        return np.vstack([ts[:-2 * tau], ts[tau:-tau], ts[2 * tau:]]).T

    def sampling_rate(self):
        return float(self._timevec.size) / (self._timevec[-1] - self._timevec[0])

    def plot(self):
        if self._last_ts is None:
            self.integrate()

        _x, _y, _z = self._last_ts.transpose()
        f = go.Figure(
            go.Scatter3d(x=_x, y=_y, z=_z,
                         mode='lines+markers',
                         marker=dict(size=3)))
        f.show()
