from ..tools import reference_rule
import numpy as np

_CORRDIM_METHODS = {
    'fit',
    'tangent',
    'gp'
}




def correlation_dimension(x, radius=None, radius_range=(0.5, 1), range_is_relative=True,
                          n_radius=10, log_base=10, norm_p=float('inf'), method='fit', ):
    assert(method in _CORRDIM_METHODS)

    if radius is None:
        radius = reference_rule(x, norm_p=norm_p)

    if range_is_relative:
        absolute_range = tuple(b *  radius for b in radius_range)
    else:
        absolute_range = tuple(radius_range)

    radius_values = np.linspace(absolute_range[0], absolute_range[1], n_radius)



