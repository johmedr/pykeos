from abc import ABC, abstractmethod
import plotly.graph_objs as go
import numpy as np
from scipy.integrate import odeint


class AbstractBaseSys(ABC):
    def __init__(self, dim, map_func, init_func=None, n_points=None):
        """
        :param dim:
        :param x0: (npoints, dim)
        """
        if dim < 1:
            raise ValueError("Incorrect number of dimensions")

        self.dim = dim
        self.states = None
        self.n_points = n_points
        self.time_vec = None

        self.rand_init = init_func if callable(init_func) else None

        if not callable(map_func):
            raise ValueError()
        self.time_map = map_func

    @abstractmethod
    def integrate(self, *args, **kwargs):
        pass

    def make_plot(self, show=True, fig=None, **trace_kwargs):
        if self.dim > 3:
            raise NotImplementedError()

        if self.states is None:
            try:
                self.integrate()
            except ValueError:
                raise ValueError("Cannot integrate - call integrate manually with keyword update_states=True.")

        if fig is None:
            fig = go.Figure()

        if self.dim == 1:
            fig.add_trace(go.Scatter(x=self.time_vec if self.time_vec is not None
                                    else np.arange(self.n_points), y=self.states, mode='lines', **trace_kwargs))
        elif self.dim == 2:
            fig.add_trace(go.Scatter(x=self.states[:, 0], y=self.states[:, 1], mode='lines', **trace_kwargs))
        elif self.dim == 3:
            fig.add_trace(go.Scatter3d(x=self.states[:, 0], y=self.states[:, 1], z=self.states[:, 2], mode='lines',
                                       **trace_kwargs))

        if show:
            fig.show()

        return fig


class DiscreteSys(AbstractBaseSys):
    def integrate(self, n_points=None, x0=None, update_states=True):
        if x0 is None:
            if not callable(self.rand_init):
                raise ValueError("Either x0 or an initialization function must be provided")
            else:
                x0 = self.rand_init()

        if n_points is None:
            if self.n_points is None:
                raise ValueError("n_points must be provided")
            else:
                n_points = self.n_points

        states = [x0 if x0 is not None else self.rand_init]

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
        super().__init__(dim, map_func, init_func)

        if None not in [n_points, t_min, t_max]:
            self.time_vec = np.linspace(t_min, t_max, n_points)
        else:
            self.time_vec = None

    def integrate(self, time_vec=None, x0=None, update_states=True, **odeint_kwargs):

        if x0 is None:
            if not callable(self.rand_init):
                raise ValueError("Either x0 or an initialization function must be provided")
            else:
                x0 = self.rand_init()

        if time_vec is None:
            if self.time_vec is None:
                raise ValueError("time_vec must be provided")
            else:
                time_vec = self.time_vec

        states = odeint(self.time_map, x0, time_vec, **odeint_kwargs)

        if update_states:
            self.states = states
            self.time_vec = time_vec
            return self

        else:
            return states
