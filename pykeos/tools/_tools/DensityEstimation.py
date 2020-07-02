import numpy as np

from .._tools import make_uniform_kernel


class DensityEstimation:
    def __init__(self, Xi, h='auto', kernel=None, norm_p=None):
        '''
        Xi is (npoints, dim)
        '''
        self.Xi = Xi if len(Xi.shape) == 2 else Xi.reshape((-1, 1))
        self.n_points, self.dim = self.Xi.shape
        self.norm_p = norm_p if norm_p is not None else 2
        self.kernel = kernel if kernel is not None else make_uniform_kernel(self.dim, self.norm_p)
        self.h = h if h != "auto" else DensityEstimation._rule_of_thumb(self.Xi)

    @staticmethod
    def _rule_of_thumb(Xi, verbose=True):
        n, d= Xi.shape
        if d > 1:
            sigma = np.sqrt(np.mean(np.diag(np.cov(Xi, rowvar=False))))
        else:
            sigma = np.sqrt(np.var(Xi))
        h = (18. * (2 * np.sqrt(np.pi))**d / (n * (d + 2))) ** (1./(d+4)) * sigma

        if verbose:
            print("Rule of thumbs: bandwidth = %.4f"%h)

        return h

    def _row_call(self, x, h):
        return np.sum(np.vectorize(self.kernel)(
            np.linalg.norm(x - self.Xi, ord=self.norm_p, axis=-1) / h)) \
               / (self.n_points * (h ** self.dim))

    def __call__(self, x, h=None):
        if h is None:
            assert(self.h is not None)
            h = self.h

        return np.apply_along_axis(self._row_call, -1, x.reshape((-1, self.dim)), h)