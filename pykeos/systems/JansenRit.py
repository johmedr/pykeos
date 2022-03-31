from ..systems import ContinuousSys
from ..tools import nd_rand_init, sigmoid

import numpy as np



class JansenRit(ContinuousSys):
    def __init__(self, A=3.25, B=22, a_inv=10, b_inv=20, C=135, Crep=[1., 0.8, 0.25, 0.25], vmax=5, v0=6, r=0.56, n_points=5000, t_min=0, t_max=30):
        self.A = A
        self.B = B
        self.C = C
        self.a = 1. / a_inv
        self.b = 1. / b_inv
        self.C = C
        self.C1 = Crep[0] * C
        self.C2 = Crep[1] * C
        self.C3 = Crep[2] * C
        self.C4 = Crep[3] * C
        self.vmax = vmax
        self.v0 = v0
        self.r = r

        def sigm(x): 
            return self.vmax * sigmoid(self.r * (x - v0))


        def ode(X, t, p=None):
            x0, x1, x2, x3, x4, x5 = X
            p = p(t) if callable(p) else 0
            return np.asarray([
                x3, x4, x5, 
                self.A * self.a * sigm(x1 - x2) - 2 * self.a * x3 - self.a**2 * x0,
                self.A * self.a * (p + self.C2  * sigm(x1 * self.C1)) - 2 * self.a * x4 - self.a**2 * x1, 
                self.B * self.b * self.C4 * sigm(self.C3 * x0) - 2 * self.b * x5 - self.b ** 2 * x2
            ])

        def rand_init():
            return nd_rand_init(*[(-5,5)]*6)

        super().__init__(dim=6, map_func=ode, init_func=rand_init, n_points=n_points, t_min=t_min, t_max=t_max)

