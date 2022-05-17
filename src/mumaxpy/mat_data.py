"""
Process data from .mat file
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from scipy.io import loadmat
from astropy import units as u
from dataclasses import dataclass
from .utilities import extract_parameters, get_filename


# %% Constants
IX, IY, IZ = 0, 1, 2
T = 't'
X, Y, Z = 'x', 'y', 'z'
COMP = 'component'

XYZ = {X: IX, Y: IY, Z: IZ}
AX_NUM = {T: 0, X: 1, Y: 2, Z: 3, COMP: 4}


# %% Functions
def plot_2D(x, y, data,
            xlabel='', ylabel='', title='',
            cmin=None, cmax=None, cmap='OrRd',
            add_plot_func=None,
            file=None):
    if cmax is None:
        cmax = np.max(data)
    if cmin is None:
        cmin = np.min(data)
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 8))
    xm, ym = np.meshgrid(x, y)
    im = ax.pcolormesh(xm, ym, np.transpose(data),
                       vmin=cmin, vmax=cmax,
                       cmap=cmap,
                       shading='auto')
    ax.set_aspect('equal')
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if add_plot_func is not None:
        add_plot_func(ax)

    fig.tight_layout()

    if file is not None:
        fig.savefig(file)


def animate_2D(x, y, data,
               xlabel='', ylabel='', title='',
               cmin=None, cmax=None, cmap='bwr',
               add_plot_func=None,
               frame_interval=100,
               file=None):
    if cmax is None:
        cmax = np.max(data)
    if cmin is None:
        cmin = np.min(data)
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 8))
    xm, ym = np.meshgrid(x, y)
    im = ax.pcolormesh(xm, ym, np.transpose(data[0]),
                       vmin=cmin, vmax=cmax,
                       cmap=cmap,
                       shading='auto')
    ax.set_aspect('equal')
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if add_plot_func is not None:
        add_plot_func(ax)

    fig.tight_layout()

    def animate(i):
        im.set_array(np.transpose(data[i]).flatten())
        return im

    anim = animation.FuncAnimation(fig, animate,
                                   interval=frame_interval,
                                   frames=len(data)-1)
    anim.save(file)


# %% Classes
@dataclass
class Time:
    values: np.ndarray
    unit: str


@dataclass
class Grid:
    nodes: np.ndarray
    size: np.ndarray
    step: np.ndarray
    unit: str

    def __post_init__(self):
        """ Coordinates of grid corner """
        self.coord = np.array([-self.size[i]/2 for i in (IX, IY, IZ)])

    def set_coord(self, x, y, z):
        self.coord = np.array([x, y, z])


@dataclass
class Quantity:
    name: str
    unit: str
    dimension: int

    def __post_init__(self):
        if self.unit == '1':
            self.unit = ''


class MatFileData:
    def __init__(self, file):
        self._file = file
        mat_dict = loadmat(file)

        time = mat_dict['time'][0][0]
        self.time = Time(time['values'][0],
                         time['unit'][0])

        grid = mat_dict['grid'][0][0]
        self.grid = Grid(grid['nodes'][0],
                         grid['size'][0],
                         grid['step'][0],
                         grid['unit'][0])

        quantity = mat_dict['quantity'][0][0]
        self.quantity = Quantity(quantity['name'][0],
                                 quantity['unit'][0],
                                 quantity['dimension'][0][0])

        if 'extra' in mat_dict:
            self.extra = mat_dict['extra'][0][0]
        else:
            self.extra = None

        self.data = mat_dict['data']

        self.filename = get_filename(file)
        self.parameters = extract_parameters(self.filename)

    def delete(self):
        # Delete file to save space
        os.remove(self._file)

    def set_coord(self, coord=(0, 0, 0)):
        self.grid.set_coord(coord)

    def convert_units(self, time_unit=None, space_unit=None):
        if time_unit is not None:
            cur_unit = u.Unit(self.time.unit)
            self.time.unit = time_unit
            multiplier = cur_unit.to(time_unit)
            self.time.values *= multiplier

        if space_unit is not None:
            cur_unit = u.Unit(self.grid.unit)
            self.grid.unit = space_unit
            multiplier = cur_unit.to(space_unit)
            self.grid.size *= multiplier
            self.grid.step *= multiplier
            self.grid.coord *= multiplier

    # TODO
    # def apply(self, func):
    def convert_quantity(self, convert_func):
        self.data, self.quantity = convert_func(self.data, self.quantity)

    def get_axis_data(self, axis, mesh=False):
        """ Return nodes coordinates for selected axis
            If mesh is False
            then return N values for coordinates of cell centers,
            else return N+1 values for coordinates of cell boundaries"""
        i = XYZ[axis]
        if mesh is False:
            base = self.grid.coord[i] + self.grid.step[i]/2
            ax_data = np.r_[base: base + self.grid.size[i]:
                            self.grid.step[i]]
        else:
            base = self.grid.coord[i]
            ax_data = np.r_[base: base + self.grid.size[i]:
                            1j*(self.grid.nodes[i] + 1)]
        return ax_data

    def crop_time(self, t_start=np.nan, t_end=np.nan):
        t = self.time.values
        mask = (t >= max(t[0], t_start)) & (t <= min(t[-1], t_end))
        self.data = self.data[mask]
        self.time.values = t[mask]

    def crop(self, axis, start=np.nan, end=np.nan):
        # TODO units check
        if axis == 't':
            self.crop_time(self, t_start=start, t_end=end)
        else:
            i = XYZ[axis]

            # Coordinates of cells (their centers)
            c = self.get_axis_data(axis)

            mask = (c >= max(c[0], start)) & (c <= min(c[-1], end))
            mask_idxs = np.nonzero(mask)[0]
            self.data = np.take(self.data, mask_idxs, axis=AX_NUM[axis])

            c = c[mask]
            self.grid.size[i] = len(c) * self.grid.step[i]
            self.grid.nodes[i] = len(c)
            self.grid.coord[i] = c[0] - self.grid.step[i]/2

    def plot_amplitude(self, normal=Z, layer_index=None, component=None,
                       cmin=None, cmax=None, cmap='OrRd',
                       add_plot_func=None, save_path=None):
        axes = [ax for ax in XYZ if ax != normal]
        axes_data = [self.get_axis_data(ax, mesh=True) for ax in axes]

        qname = self.quantity.name
        if self.quantity.dimension == 1:
            data = self.data
            title = f'{qname}'
        else:
            if component is not None:
                data = self.data[:, :, :, :, XYZ[component]]
                title = f'{qname}_{component}'
            else:
                data = (self.data[:, :, :, :, XYZ[axes[0]]]**2
                        + self.data[:, :, :, :, XYZ[axes[1]]]**2)
                title = f'|({qname}_{axes[0]}, {qname}_{axes[1]})|'

        if layer_index is None:
            # Mean over normal
            data = np.mean(data, axis=AX_NUM[normal])
            layer_coord = None
        else:
            # Get layer data
            # data = data[:, :,  ... , layer_index]
            slice_tuple = (slice(None),) * AX_NUM[normal] + (layer_index,)
            data = data[slice_tuple]
            # Get coordinate of layer
            layer_coord = self.get_axis_data(normal)[layer_index]

        if self.quantity.unit != '':
            title += f', {self.quantity.unit}'
        if len(self.parameters) > 0:
            p = []
            for key, val in self.parameters.items():
                p += [f'{key}={val[0]}{val[1]}']
            title += ' [' + ', '.join(p) + ']'

        if layer_coord is not None:
            title += f' @ {normal} = {layer_coord:.2f}{self.grid.unit}'

        if save_path is not None and os.path.isdir(save_path):
            file = os.path.join(save_path, f'{self.filename}.png')
        else:
            file = save_path

        # Amplitude = maximum - minimum over time
        data_amp = (data.max(axis=AX_NUM[T])
                    - data.min(axis=AX_NUM[T])) / 2

        plot_2D(axes_data[0], axes_data[1], data_amp,
                xlabel=f'{axes[0]}, {self.grid.unit}',
                ylabel=f'{axes[1]}, {self.grid.unit}',
                title=title,
                cmin=cmin,
                cmax=cmax,
                cmap=cmap,
                add_plot_func=add_plot_func,
                file=file)

    def create_animation(self, normal=Z, layer_index=None, component=None,
                         cmin=None, cmax=None, cmap='bwr',
                         add_plot_func=None,
                         time_factor=2e9,
                         save_path=None, extension='mp4'):
        # TODO move doubled code to func
        axes = [ax for ax in XYZ if ax != normal]
        axes_data = [self.get_axis_data(ax, mesh=True) for ax in axes]

        qname = self.quantity.name
        if self.quantity.dimension == 1:
            data = self.data
            title = f'{qname}'
        else:
            if component is not None:
                data = self.data[:, :, :, :, XYZ[component]]
                title = f'{qname}_{component}'
            else:
                data = (self.data[:, :, :, :, XYZ[axes[0]]]**2
                        + self.data[:, :, :, :, XYZ[axes[1]]]**2)
                title = f'|({qname}_{axes[0]}, {qname}_{axes[1]})|'

        if layer_index is None:
            # Mean over normal
            data = np.mean(data, axis=AX_NUM[normal])
            layer_coord = None
        else:
            # Get layer data
            # data = data[:, :,  ... , layer_index]
            slice_tuple = (slice(None),) * AX_NUM[normal] + (layer_index,)
            data = data[slice_tuple]
            # Get coordinate of layer
            layer_coord = self.get_axis_data(normal)[layer_index]

        if self.quantity.unit != '':
            title += f', {self.quantity.unit}'
        if len(self.parameters) > 0:
            p = []
            for key, val in self.parameters.items():
                p += [f'{key}={val[0]}{val[1]}']
            title += ' [' + ', '.join(p) + ']'

        if layer_coord is not None:
            title += f' @ {normal} = {layer_coord:.2f}{self.grid.unit}'

        if save_path is not None and os.path.isdir(save_path):
            file = os.path.join(save_path, f'{self.filename}.{extension}')
        else:
            file = save_path

        delta_t = ((self.time.values[-1] - self.time.values[0])
                   / (len(self.time.values) - 1)) * u.Unit(self.time.unit)
        frame_interval = delta_t.to('ms') * time_factor

        animate_2D(axes_data[0], axes_data[1], data,
                   xlabel=f'{axes[0]}, {self.grid.unit}',
                   ylabel=f'{axes[1]}, {self.grid.unit}',
                   title=title,
                   cmin=cmin,
                   cmax=cmax,
                   cmap=cmap,
                   add_plot_func=add_plot_func,
                   frame_interval=frame_interval.value,
                   file=file)
