from ..systems import AbstractBaseSys
from ..tools import reference_rule

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots



import numpy as np

from typing import Union

black_white_colorscale = [[0, "rgb(255, 255, 255)"],[1, "rgb(0, 0, 0)"]]


def plot_rp(sys: Union[AbstractBaseSys, None] = None,
            rp: Union[np.ndarray, None] = None,
            sig: Union[np.ndarray, None] = None,
            time_vec: Union[np.ndarray, None] = None,
            width: int = 1080,
            rp_height_ratio: float = 0.8,
            vspacing_ratio: float = 0.1,
            axis: int = 0
        ) -> go.Figure:

    _time_vec = None
    if time_vec is None:
        _time_vec = sys.time_vec if sys is not None else None
    print(_time_vec)

    _ts = None
    if sys is None:
        assert(sig is not None)
        _ts = sig
    else:
        _ts = sys.states

    _rp = None
    if rp is None:
        import pyunicorn.timeseries as puts
        _rp = puts.RecurrencePlot(_ts, threshold=reference_rule(_ts, norm_p='inf')).R
    elif isinstance(rp, np.ndarray):
        _rp = rp
    # elif isinstance(rp, puts.RecurrencePlot):
    #     _rp = rp.R

    fig = make_subplots(rows=2, row_heights=[rp_height_ratio, 1-rp_height_ratio-vspacing_ratio], vertical_spacing=vspacing_ratio)

    fig.add_trace(go.Heatmap(z=_rp, colorscale=black_white_colorscale,  showscale=False, yaxis="y"), row=1, col=1)
    fig.add_trace(go.Scatter(x=_time_vec, y=_ts[:, axis], line=dict(color="rgb(0,0,0)")), row=2, col=1)

    fig.update_yaxes(col=1, row=1, autorange='reversed')
    fig.update_layout(width=width, height=int(width / rp_height_ratio))

    return fig


class Scale4Latex:
    DEFAULTS = {
        'img_width': 1920,
        'img_height': 1080,
        'font_size': 10,
        'title_font_size': 11,
        'font_family': 'Computer Modern',
        'line_width': 1,
        'marker_size': 4,
    }

    def __init__(self, output_img_width=1920, latex_img_width=624, **kwargs):
        self._scale = output_img_width / latex_img_width
        self._img_width = output_img_width
        self._latex_img_width = latex_img_width
        for k, v in self.__class__.DEFAULTS.items():
            value = kwargs[k] if k in kwargs.keys() else v
            if isinstance(value, int) or isinstance(value, float):
                self.__dict__.update({k: self._scale_object(value)})
            else:
                self.__dict__.update({k:value})

    def _scale_object(self, o):
        return int(self._scale * o)

    @property
    def img_width(self):
        return self._img_width

    @property
    def latex_img_width(self):
        return self._latex_img_width

    def scale_font(self, font_size):
        return self._scale_object(font_size)