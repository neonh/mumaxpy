"""
Plot functions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from typing import Tuple, Callable, Optional


# %% Types
Path = str


# %% Constants
# Figure size in inches
AX_SIZE_MAX = 7.0
DELTA_W = 2.0
DELTA_H = 1.0

# Figure margin in percents
H_MARGIN = 0.1
V_MARGIN = 0.05

# Slider height in percent
H_SLIDER = 0.05

FONTSIZE = 14


# %% Functions
def get_fig_size(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    aspect_ratio = abs((x[-1] - x[0]) / (y[-1] - y[0]))

    if aspect_ratio >= 1:
        w = AX_SIZE_MAX
        h = w / aspect_ratio
    else:
        h = AX_SIZE_MAX
        w = h * aspect_ratio
    return (w + DELTA_W, h + DELTA_H)


def plot_2D(x: np.ndarray, y: np.ndarray, data: np.ndarray,
            xlabel: str = '', ylabel: str = '', title: str = '',
            cmap: str = 'OrRd',
            vmin: Optional[float] = None, vmax: Optional[float] = None,
            add_plot_func: Optional[Callable] = None,
            file: Path = None) -> plt.Axes:
    fig, ax = plt.subplots(figsize=get_fig_size(x, y))
    xm, ym = np.meshgrid(x, y)
    im = ax.pcolormesh(xm, ym, np.transpose(data),
                       cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       shading='auto')
    ax.set_aspect('equal')
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)

    if add_plot_func is not None:
        add_plot_func(ax)

    fig.tight_layout()

    if file is not None:
        fig.savefig(file)

    return ax


def animate_2D(time: np.ndarray, time_unit: str,
               x: np.ndarray, y: np.ndarray, data: np.ndarray,
               xlabel: str = '', ylabel: str = '', title: str = '',
               cmap: str = 'bwr',
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               add_plot_func: Optional[Callable] = None,
               frame_interval: int = 100,
               file: Optional[Path] = None) -> plt.Axes:
    # Select min/max ranges for all data
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    fig, (ax, axtime) = plt.subplots(nrows=2, figsize=get_fig_size(x, y),
                                     gridspec_kw={'height_ratios':
                                                  [1-H_SLIDER, H_SLIDER]})
    xm, ym = np.meshgrid(x, y)
    im = ax.pcolormesh(xm, ym, np.transpose(data[0]),
                       cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       shading='auto')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    fig.colorbar(im, ax=ax)

    if add_plot_func is not None:
        add_plot_func(ax)

    # Create slider
    slider = Slider(axtime, f't, {time_unit}',
                    time[0], time[-1],
                    valinit=time[0],
                    initcolor='none')

    def update(val):
        # ax.set_title(f'{title} [t = {time[i]:.2f} {time_unit}]')
        i = np.searchsorted(time, slider.val, side='left')
        im.set_array(np.transpose(data[i]).flatten())
        fig.canvas.draw()

    slider.on_changed(update)

    fig.subplots_adjust(left=H_MARGIN, right=1-H_MARGIN,
                        bottom=V_MARGIN, top=1-H_MARGIN)

    if file is not None:
        # Create animation
        def animate(i):
            slider.set_val(time[i])
            return im

        anim = animation.FuncAnimation(fig, animate,
                                       interval=frame_interval,
                                       frames=len(data)-1)
        anim.save(file)

        # Stop animation
        fig.canvas.flush_events()
        anim.event_source.stop()

    # Go to end time
    slider.set_val(time[-1])

    return ax