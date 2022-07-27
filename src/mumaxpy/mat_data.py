"""
Process data from .mat file
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from scipy.io import loadmat, savemat
from astropy import units as u
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
from .utilities import extract_parameters, get_filename
from .signal import Signal


# %% Types
# X, Y or Z
Ax = str


# %% Constants
IX, IY, IZ = 0, 1, 2
T = 't'
X, Y, Z = 'x', 'y', 'z'
COMP = 'component'

XYZ = {X: IX, Y: IY, Z: IZ}
AX_NUM = {COMP: 0, T: 1, X: 2, Y: 3, Z: 4}


# %% Functions
def probe_mask(ax_data, probe_coord, probe_D):
    probe_mask = np.exp(-2*((ax_data[0] - probe_coord[0])[:, np.newaxis]**2 +
                            (ax_data[1] - probe_coord[1])[np.newaxis, :]**2)
                        / probe_D**2)
    # Normalize
    probe_mask /= np.sum(probe_mask)
    return probe_mask


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
    base: np.ndarray
    unit: str

    def __post_init__(self) -> None:
        """ Coordinates of grid corner """
        self.coord = np.array([-self.size[i]/2 for i in (IX, IY, IZ)])

    def set_coord(self, x: float, y: float, z: float) -> None:
        self.coord = np.array([x, y, z])


@dataclass
class Quantity:
    name: str
    unit: str
    dimension: int
    components: Optional[List[str]]

    def __post_init__(self) -> None:
        if self.unit == '1':
            self.unit = ''

    def __repr__(self) -> str:
        s = self.name
        if self.unit != '':
            s += f', {self.unit}'
        return s


class MatFileData:
    """
    Class for vector field data:

        [vector component, time, x, y, z]

    Attributes:
        time (Time): time object
        grid (Grid): grid object
        quantity (Quantity): quantity object
        data (np.ndarray): 5-dimensional array with the following shape:
            [vector components, time steps, x nodes, y nodes, z nodes]
        extra (dict): any extra data
        parameters (dict): parameters names and values from the filename
    """

    def __init__(self, file) -> None:
        self._file = file
        mat_dict = loadmat(file)

        self.title = mat_dict['title']

        time = mat_dict['time'][0][0]
        self.time = Time(time['values'][0],
                         time['unit'][0])

        grid = mat_dict['grid'][0][0]
        self.grid = Grid(grid['nodes'][0],
                         grid['size'][0],
                         grid['step'][0],
                         grid['base'][0],
                         grid['unit'][0])

        quantity = mat_dict['quantity'][0][0]
        q_labels = list(quantity['labels'])
        # Quantity and unit should be the same
        q_name = os.path.commonprefix(q_labels).strip('_')
        q_unit = os.path.commonprefix(list(quantity['units']))
        q_dim = quantity['dimension'][0][0]
        if q_dim > 1:
            # Vector components
            q_components = [lbl[-1] for lbl in q_labels]
        else:
            q_components = None
        self.quantity = Quantity(q_name,
                                 q_unit,
                                 q_dim,
                                 q_components)

        if 'extra' in mat_dict:
            self.extra = mat_dict['extra'][0][0]
        else:
            self.extra = None

        self.data = mat_dict['data']

        self._filename = get_filename(file)
        self.parameters = extract_parameters(self._filename)

    def save(self, file: Optional[str] = None) -> None:
        if file is None:
            file = self._file
        if self.quantity.unit == '':
            q_unit = '1'
        else:
            q_unit = self.quantity.unit

        if self.quantity.dimension > 1:
            q_labels = [f'{self.quantity.name}_{c}'
                        for c in self.quantity.components]
            q_units = [q_unit] * self.quantity.dimension
        else:
            q_labels = [self.quantity.name]
            q_units = [self.quantity.unit]

        # Save modified data
        mat_dict = {'title': self.title,
                    'time': {'values': self.time.values,
                             'unit': self.time.unit},
                    'grid': {'nodes': self.grid.nodes,
                             'size': self.grid.size,
                             'step': self.grid.step,
                             'base': self.grid.base,
                             'unit': self.grid.unit},
                    'quantity': {'dimension': self.quantity.dimension,
                                 'labels': q_labels,
                                 'units': q_units,
                                 },
                    'data': self.data}
        if self.extra is not None:
            mat_dict['extra'] = self.extra
        savemat(file, mat_dict, do_compression=False)

    def delete(self) -> None:
        # Delete file to save space
        os.remove(self._file)

    def set_coord(self, coord: Tuple[float, float, float] = (0, 0, 0)) -> None:
        self.grid.set_coord(coord)

    def convert_units(self,
                      time_unit: Optional[str] = None,
                      space_unit: Optional[str] = None) -> None:
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
    def convert_quantity(self,
                         convert_func:
                             Callable[[Tuple[np.ndarray, Quantity]],
                                      Tuple[np.ndarray, Quantity]]) -> None:
        self.data, self.quantity = convert_func(self.data, self.quantity)

    def get_shape(self) -> Tuple:
        return self.data.shape

    def get_time_data(self) -> None:
        return self.time.values

    def get_axis_data(self, axis: Ax, mesh: bool = False) -> np.ndarray:
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

    def get_data(self, component: Optional[Ax] = None) -> np.ndarray:
        """
        Returns data array keeping dimensions count.

        Parameters
        ----------
        component: vector component, if None return all components.
            If quantity is scalar - ignore this parameter.

        Returns
        -------
        data: data array
        """
        if component is not None and self.quantity.dimension > 1:
            if component in self.quantity.components:
                data = self.data[[XYZ[component]]]
            else:
                err = f'{self.quantity.name} have no {component}-component'
                raise RuntimeError(err)
        else:
            data = self.data
        return data

    def get_plane_data(self,
                       normal: Ax,
                       layer_index: Optional[int] = None,
                       component: Optional[Ax] = None) -> np.ndarray:
        """
        Returns plane section of data array (without normal-axis dimension).

        Parameters
        ----------
        normal: normal of the plane
        layer_index: index of layer, if None return mean of all layers
        component: vector component, if None return all components

        Returns
        -------
        data: data array
        """
        data = self.get_data(component)

        if layer_index is None:
            data = np.mean(data, axis=AX_NUM[normal])
        else:
            data = np.take(data, layer_index, axis=AX_NUM[normal])
        return data

    def crop_time(self,
                  t_start: float = np.nan, t_end: float = np.nan) -> None:
        t = self.time.values
        mask = (t >= max(t[0], t_start)) & (t <= min(t[-1], t_end))
        self.data = self.data[mask]
        self.time.values = t[mask]

    def crop(self,
             axis: str, start: float = np.nan, end: float = np.nan) -> None:
        if axis == T:
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

    def _get_plot_title(self, component: Ax, normal: Ax,
                        layer_index: int) -> str:
        if self.quantity.dimension == 1:
            title = f'{self.quantity.name}'
        else:
            title = f'{self.quantity.name}_{component}'
        if self.quantity.unit != '':
            title += f', {self.quantity.unit}'

        if len(self.parameters) > 0:
            p = []
            for key, val in self.parameters.items():
                p += [f'{key}={val[0]}{val[1]}']
            title += ' [' + ', '.join(p) + ']'

        if layer_index is not None:
            layer_coord = self.get_axis_data(normal)[layer_index]
            title += f' @ {normal} = {layer_coord:.2f}{self.grid.unit}'

        return title

    def _get_plot_filepath(self, save_path: str, extension: str) -> str:
        if save_path is None:
            file = os.path.join(os.path.dirname(self._file),
                                f'{self._filename}.{extension}')
        elif os.path.isdir(save_path):
            file = os.path.join(save_path, f'{self._filename}.{extension}')
        else:
            file = save_path
        return file

    def plot_amplitude(self, component, normal=Z, layer_index=None,
                       cmin=None, cmax=None, cmap='OrRd',
                       add_plot_func=None,
                       save_path=None,
                       extension='png') -> None:
        title = self._get_plot_title(component, normal, layer_index)
        file = self._get_plot_filepath(save_path, extension)
        axes = [ax for ax in XYZ if ax != normal]
        axes_data = [self.get_axis_data(ax, mesh=True) for ax in axes]

        # Amplitude = maximum - minimum over time
        data = self.get_plane_data(normal, layer_index, component)
        data_amp = ((data.max(axis=AX_NUM[T])
                     - data.min(axis=AX_NUM[T])) / 2)[0]

        plot_2D(axes_data[0], axes_data[1], data_amp,
                xlabel=f'{axes[0]}, {self.grid.unit}',
                ylabel=f'{axes[1]}, {self.grid.unit}',
                title=title,
                cmin=cmin,
                cmax=cmax,
                cmap=cmap,
                add_plot_func=add_plot_func,
                file=file)

    def create_animation(self, component, normal=Z, layer_index=None,
                         cmin=None, cmax=None, cmap='bwr',
                         add_plot_func=None,
                         time_factor=2e9,
                         save_path=None,
                         extension='mp4') -> None:
        title = self._get_plot_title(component, normal, layer_index)
        file = self._get_plot_filepath(save_path, extension)

        delta_t = ((self.time.values[-1] - self.time.values[0])
                   / (len(self.time.values) - 1)) * u.Unit(self.time.unit)
        frame_interval = delta_t.to('ms') * time_factor

        axes = [ax for ax in XYZ if ax != normal]
        axes_data = [self.get_axis_data(ax, mesh=True) for ax in axes]
        data = self.get_plane_data(normal, layer_index, component)[0]
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

    def get_probe_signals(self,
                          probe_coordinates: List[Tuple[float, float]],
                          probing_component: Ax,
                          probe_axis: Ax = Z,
                          probe_D: float = 0) -> Signal:
        """ Get signals by probing """
        sig = Signal(self.time.values, 'Time', self.time.unit,
                     self.quantity.name, self.quantity.unit,
                     'Probe coordinates', self.grid.unit)

        # Axes perpendicular to probe axis
        axes = [a for a in XYZ if a != probe_axis]
        ax_data = [self.get_axis_data(i) for i in axes]
        data = self.get_plane_data(normal=probe_axis,
                                   component=probing_component)[0]

        for p_coord in probe_coordinates:
            if probe_D == 0:
                # Get near indexes
                idx = [[i for i in range(len(ax_data[n]))
                        if ax_data[n][i] >= p_coord[n]][0] for n in range(2)]
                signal = data[:, idx[0], idx[1]]
            else:
                signal = np.sum(data * probe_mask(ax_data, p_coord, probe_D),
                                axis=(1, 2))

            sig.add(signal, p_coord)

        return sig
