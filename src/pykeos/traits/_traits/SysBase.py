
class AbstractStateHolder(object):
    def __init__(self):
        self._states = None

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        self._states = states


class SysBase(AbstractStateHolder):
    pass

