from pytraits import Trait
from abc import abstractmethod

import numpy as np


class AbstractIntegrate(Trait):

    @abstractmethod
    def _time_map(self, *args, **kwargs):
        pass

    @abstractmethod
    def integrate(self, *args, **kwargs):
        pass


class DiscreteMap(AbstractIntegrate):
    def integrate(self, n_points, x0, update_states=True):
        states = [x0]
        for i in n_points:
            states.append(self._time_map(x0))
        states = np.asarray(states)

        self.states = states

        return states


class ODEMap(AbstractIntegrate):
    def integrate(self, timevec, x0, update_states=True, **odeint_kwargs):
        from scipy.integrate import odeint

        states = odeint(self._time_map, x0, timevec, **odeint_kwargs)

        self.states = states

        return states
