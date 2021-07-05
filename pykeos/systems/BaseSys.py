from abc import ABC, abstractmethod
from typing import Union, Dict, Callable, Optional
from numbers import Number
from scipy.integrate import odeint

import plotly.graph_objs as go
import numpy as np
import warnings


class AbstractBaseSys(ABC):
    """ Abstract class representing common attributes of pykeos systems.

    This class is not supposed to be instancied but is a common template for
    pykeos systems. Defines common functions that are usually applicable to
    either formally-defined discrete or continuous (sampled) systems, as well
    as to unformally-defined (wrapped) systems.

    Parameters
    ----------
    dim : int
        The dimension of the system
    map_func : Callable
        A callable map that will be integrated with the defined integration scheme. Will set the attribute time_map.
    init_func : Callable, optional
        The initialization function that must be called to get a random initial point to the system. Will set the
        attribute rand_init.
    n_points : int, optional
        The number of points.

    Attributes
    ----------
    dim : int
        The dimension of the system.
    n_points : int
        The number of points.
    states : np.ndarray
        An array of shape (n_points, dim) by convention. The first time states is called, it will integrate the time_map
        (if defined) from an initial_state (returned by rand_init()) for a duration of n_points.
    time_vec : Optional[np.ndarray]
        The time vector associated with the time series. Is usually None for discrete systems.
    time_map : Callable
        A callable map that will be integrated with the defined integration scheme. See DiscreteSys and ContinuousSys.
    rand_init : Optional[Callable]
        The initialization function that must be called to get a random initial point to the system.
    """
    def __init__(self, dim: int, map_func: Callable, init_func: Optional[Callable] = None,
                 n_points: Optional[int] = None):
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

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def states(self) -> np.ndarray:
        """ Getter for the states timeseries of the system, calls integrate if the states are not yet defined.

        Returns
        -------
        np.ndarray
            An array of shape (n_points, dim) representing a timeseries of states for the system. If the
            serie is not defined yet, the system integrates its map function with its default parameters
            to create a timeserie.
        """
        if self._states is None:
            try:
                self.integrate()
            except ValueError:
                raise ValueError("Cannot integrate - call integrate manually with keyword update_states=True.")
        return self._states

    @states.setter
    def states(self, new_states: np.ndarray):
        """ Setter for the system's states.

        Parameters
        ----------
        new_states : np.ndarray
            A new timeseries of states of shape (new_n_points, dim)."""
        assert(len(new_states.shape) in [1, 2])
        if len(new_states.shape) == 2:
            assert(new_states.shape[1] == self.dim)
        self._states = new_states
        self.n_points = new_states.shape[0]

    @property
    def initial_state(self) -> Union[np.ndarray, float]:
        """ Getter for the initial state.

        Returns
        -------
        Union[np.ndarray, float]
            An array representing an initial point for the system. Will be used to solve the Initial
            Value Problem in the case of DiscreteSys and ContinuousSys.
        """
        if self._initial_state is None:
            if not callable(self.rand_init):
                raise ValueError("Either x0 or an initialization function must be provided")
            else:
                self._initial_state = self.rand_init()
        return self._initial_state

    @initial_state.setter
    def initial_state(self, x0: Union[np.ndarray, float]):
        """ Setter for the initial state.

        Parameters
        ----------
        x0 : Union[np.ndarray, float]
            An initial value that will be used to solve the Initial Value Problem in the case of DiscreteSys and
            ContinuousSys.
        """
        self._initial_state = x0

    @abstractmethod
    def integrate(self, *args, **kwargs) -> np.ndarray:
        """ The integration scheme to use in the case of formally-defined systems.

        This method is intended to do the all of the necessary integration work for typical types of systems.
        
        See also
        --------
        DiscreteSys.integrate
        ContinuousSys.integrate

        """
        pass

    def plot(self, show: bool = True, fig: Optional[go.Figure] = None, transient_index: int = 0, mode: str = "lines",
             fig_kwargs: Optional[Dict] = None, **trace_kwargs) -> go.Figure:
        """ Plots the states attribute using plotly for 1-, 2- or 3-dimensional systems.

        Parameters
        ----------
        show : bool
            Whether to show the figure or just return it.
        fig : Optional[go.Figure]
            An optional figure to draw on.
        mode : std
            A scatterplot mode compatible with plotly (e.g. 'lines', 'markers', 'lines+markers')
        transient_index : int
            Stop index for transients. We will omit [:transient_index].
        fig_kwargs : Optional[Dict]
            Kwargs to unpack to the fig.add_trace function.
        trace_kwargs :
            kwargs to unpack to the go.Scatter function.
        """
        if self.dim > 3:
            raise NotImplementedError()

        if fig is None:
            fig = go.Figure()

        if fig_kwargs is None:
            fig_kwargs = {}

        if transient_index is None:
            states = self.states
        else:
            states = self.states[transient_index:]

        if self.dim == 1:
            fig.add_trace(go.Scatter(x=self.time_vec if self.time_vec is not None else np.arange(self.n_points),
                                     y=states[:, 0], mode=mode, **trace_kwargs), **fig_kwargs)
        elif self.dim == 2:
            fig.add_trace(go.Scatter(x=states[:, 0], y=states[:, 1], mode=mode, **trace_kwargs),
                          **fig_kwargs)
        elif self.dim == 3:
            fig.add_trace(go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], mode=mode,
                                       **trace_kwargs), **fig_kwargs)

        if show:
            fig.show()

        return fig

    def delay_coordinates(self, dim: Optional[int] = None, lag: Union[int, str] = "auto", axis: int = 0, *args, **kwargs):
        """ A convenient wrapper around pykeos delay_coordinates function.

        See also
        --------
        pykeos.tools.ebdg_utils.delay_coordinates
        """
        if dim is None:
            dim = self.dim
        from ..tools import delay_coordinates
        return delay_coordinates(self.states, dim=dim, lag=lag, axis=axis, *args, **kwargs)


class SysWrapper(AbstractBaseSys):
    """ A convenient wrapper to easily interface external data in pykeos framework.

    SysWrapper aims at wrapping a little bit of everything. It may represent partially
    observed systems, delay coordinates systems as well as real data from sensors. The usage is straightforward :
    any timeseries in form of a 1- or 2-dimensional array can be wrapped into a pykeos system using SysWrapper

    See also
    --------
    pykeos.tools.conv_utils : Contains helper function to convert data from and to pykeos.
    """
    def __init__(self, ts: np.ndarray, dim: Optional[int] = None, n_points: Optional[int] = None,
                 time_vec: Optional[np.ndarray] = None, t_min: Optional[Number] = None, t_max: Optional[Number] = None):
        """
        Parameters
        ----------
        ts : np.ndarray
            The
        :param ts:
        :param dim:
        :param n_points:
        :param time_vec:
        :param t_min:
        :param t_max:
        """
        if dim is None: 
            dim = ts.shape[1] if len(ts.shape) > 1 else 1
        if n_points is None: 
            n_points = ts.shape[0]
        if time_vec is None: 
            if t_min is None: 
                t_min = 0
            if t_max is None: 
                t_max = n_points
            time_vec = np.linspace(t_min, t_max, n_points)

        super(SysWrapper, self).__init__(
            dim=dim,
            map_func=lambda x: warnings.warn("The map function of a <SysWrapper> is not supposed to be called."),
            n_points=n_points
        )

        self.dim = dim
        self.n_points = n_points
        self.time_vec = time_vec
        self._states = ts if len(ts.shape) > 1 else ts[:, None]
        self._initial_state = self._states[0]

    def integrate(self, *args, **kwargs):
        raise RuntimeError("Instance of <SysWrapper> is not supposed to be integrated.")

    
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
        """ Solves an initial value problem with odeint.
        
        See Also
        --------
        DiscreteSys.integrate
        """
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
