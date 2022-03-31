from pykeos.tools import correlation_sum
from pykeos.tools.correlation_sum import _corr_sum_numpy_backend
import unittest
import numpy as np


def _make_ts(size=100):
    p = np.random.permutation(size)
    x = np.asarray([0] * int(size / 2) + [1] * int(size / 2))
    return x[p]


class TestCorrelationSum(unittest.TestCase):
    x = _make_ts()

    def test_correlation_sum(self):
        csum = correlation_sum(self.x, 0.5, allow_equals=True)
        self.assertAlmostEqual(csum, 0.5)

    def test_correlation_sum_numpy_backend(self):
        csum = _corr_sum_numpy_backend(self.x[:, None], 0.5, float('inf'))
        self.assertAlmostEqual(csum, 0.5)


if __name__ == '__main__':
    unittest.main()