from abc import ABC, abstractmethod
import plotly.graph_objs as go
import numpy as np
from scipy.integrate import odeint

from pykeos.tools import delay_coordinates


class AbstractBaseSys(ABC):
    def __init__(self, dim, map_func, init_func=None, n_points=None):
        """
        :param dim:
        :param x0: (npoints, dim)
        """
        if dim < 1:
            raise ValueError("Incorrect number of dimensions")

        self.dim = dim
        self._states = None
        self.n_points = n_points
        self.time_vec = None

        self.rand_init = init_func if callable(init_func) else None
        self._initial_state = None

        if not callable(map_func):
            raise ValueError()
        self.time_map = map_func

    @property
    def states(self):
        if self._states is None:
            try:
                self.integrate()
            except ValueError:
                raise ValueError("Cannot integrate - call integrate manually with keyword update_states=True.")
        return self._states

    @states.setter
    def states(self, new_states):
        assert (new_states.shape[-1] == self.dim)
        self._states = new_states

    @property
    def initial_state(self):
        if self._initial_state is None:
            if not callable(self.rand_init):
                raise ValueError("Either x0 or an initialization function must be provided")
            else:
                self._initial_state = self.rand_init()
        return self._initial_state

    @initial_state.setter
    def initial_state(self, x0):
        self._initial_state = x0

    @abstractmethod
    def integrate(self, *args, **kwargs):
        pass

    def make_plot(self, show=True, fig=None, fig_argdict=dict(), **trace_kwargs):
        if self.dim > 3:
            raise NotImplementedError()

        if fig is None:
            fig = go.Figure()

        if self.dim == 1:
            fig.add_trace(go.Scatter(x=self.time_vec if self.time_vec is not None else np.arange(self.n_points),
                                     y=self.states[:, 0], mode='lines', **trace_kwargs), **fig_argdict)
        elif self.dim == 2:
            fig.add_trace(go.Scatter(x=self.states[:, 0], y=self.states[:, 1], mode='lines', **trace_kwargs),
                          **fig_argdict)
        elif self.dim == 3:
            fig.add_trace(go.Scatter3d(x=self.states[:, 0], y=self.states[:, 1], z=self.states[:, 2], mode='lines',
                                       **trace_kwargs), **fig_argdict)

        if show:
            fig.show()

        return fig

    def delay_coordinates(self, dim: int, lag: int = 1, axis: int = 0) -> np.ndarray:
        return delay_coordinates(self.states, dim=dim, lag=lag, axis=axis)


class DiscreteSys(AbstractBaseSys):
    def integrate(self, n_points=None, x0=None, update_states=True):
        if x0 is not None:
            self.initial_state = x0

        if n_points is None:
            if self.n_points is None:
                raise ValueError("n_points must be provided")
            else:
                n_points = self.n_points

        states = [self.initial_state]

        for i in range(n_points - 1):
            states.append(self.time_map(states[i]))

        states = np.asarray(states)
        if update_states:
            self.states = states
            self.n_points = n_points
            return self

        else:
            return states


class ContinuousSys(AbstractBaseSys):
    def __init__(self, dim: int, map_func, init_func=None, n_points=None, t_min=None, t_max=None):
        super().__init__(dim, map_func, init_func, n_points=n_points)

        if None not in [n_points, t_min, t_max]:
            self.time_vec = np.linspace(t_min, t_max, n_points)
        else:
            self.time_vec = None

    def integrate(self, time_vec=None, x0=None, update_states=True, **odeint_kwargs):
        if x0 is not None:
            self.initial_state = x0

        if time_vec is None:
            if self.time_vec is None:
                raise ValueError("time_vec must be provided")
            else:
                time_vec = self.time_vec

        states = odeint(self.time_map, self.initial_state, time_vec, **odeint_kwargs)

        if update_states:
            self.states = states
            self.time_vec = time_vec
            return self

        else:
            return states
