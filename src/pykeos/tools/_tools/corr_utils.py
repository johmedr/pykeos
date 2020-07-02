import numpy as np


def _fast_count_row_1d(x, traj, r, norm_p):
    return np.count_nonzero(traj - x[np.newaxis, :] <= r)


def _fast_count_row(x, traj, r, norm_p):
    """
    :param x: (dim, )
    :param traj: (n_points, dim)
    :param norm_p: int, float('inf')
    :return:
    """
    return np.count_nonzero(np.linalg.norm(traj - x[np.newaxis, :], ord=norm_p, axis=1) <= r)


def _fast_count_traj(x, r, norm_p):
    """
    :param x: (n_points, dim)
    :param norm_p: int, float('inf')
    :return:
    """
    if x.shape[1] > 1:
        return np.sum(np.apply_along_axis(_fast_count_row, 1, x, x, r, norm_p))
    elif x.shape[1] == 1:
        return np.sum(np.apply_along_axis(_fast_count_row_1d, 1, x, x, r, norm_p))


def corr_sum(traj, r, norm_p=1):
    return _fast_count_traj(traj, r, norm_p).astype(np.float64) / traj.shape[0] ** 2


def dsty_est(x, samples, r, norm_p=1):
    if len(samples.shape) == 1:
        samples = samples[:, np.newaxis]

    dim = samples.shape[1]

    # x is (k,)
    if len(x.shape) == 1:
        # if k = d, then x is a point in phase space
        if x.shape[0] == dim:
            x = x[np.newaxis, :]
        # x is a collection of sample points in 1d phase space
        elif dim == 1:
            x = x[:, np.newaxis]
    elif len(x.shape) == 2:
        assert(x.shape[1] == samples.shape[1])

    return np.apply_along_axis(_fast_count_row, 1, x, samples, r, norm_p).astype(np.float64)  / samples.shape[0]
