"""
Signal class
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
from dataclasses import dataclass
from typing import Optional, Any


# %% Constants
TIME_TO_FREQ_UNITS = {'s': 'Hz',
                      'ms': 'kHz',
                      'us': 'MHz',
                      'ns': 'GHz',
                      'ps': 'THz'}


# %% Types
Path = str


# %% Classes
@dataclass
class Label:
    name: str
    unit: str

    def __repr__(self) -> str:
        s = str(self.name)
        if self.unit != '':
            s += f', {self.unit}'
        return s


class Signal:
    def __init__(self, x_data: np.ndarray,
                 x_name: str = '', x_unit: str = '',
                 y_name: str = '', y_unit: str = '',
                 title: str = '',
                 legend_name: str = '', legend_unit: str = '') -> None:

        self.df = pd.DataFrame(index=x_data)
        self.x = Label(x_name, x_unit)
        self.y = Label(y_name, y_unit)
        self.title = title
        self.legend = Label(legend_name, legend_unit)

    def _get_file_path(self, save_path: Path,
                       default_filename: str) -> Path:
        if os.path.isdir(save_path):
            file = os.path.join(save_path, default_filename)
        else:
            file = save_path
        return file

    def set_title(self, title: str) -> None:
        self.title = title

    def set_xlabel(self, name: str, unit: str = '') -> None:
        self.x = Label(name, unit)

    def set_ylabel(self, name: str, unit: str = '') -> None:
        self.y = Label(name, unit)

    def set_legend(self, name: str, unit: str = '') -> None:
        self.legend = Label(name, unit)

    def add(self, data: np.ndarray, label: Optional[Any] = None) -> None:
        if label is not None:
            self.df[label] = data
        else:
            i = len(self.df.columns)
            self.df[str(i)] = data

    def get_x(self) -> np.ndarray:
        return self.df.index.values

    def get_y(self, label: Optional[Any] = None) -> np.ndarray:
        if label is not None:
            data = self.df[label].values
        else:
            # Return first
            data = self.df.iloc[:, 0].values
        return data

    def get(self) -> pd.DataFrame:
        return self.df

    def plot(self, save_path: Optional[Path] = None) -> plt.Axes:
        fig, ax = plt.subplots()
        self.df.plot.line(ax=ax, xlabel=str(self.x), ylabel=str(self.y))
        ax.legend(title=f'{self.legend}:')
        ax.set_title(self.title)
        fig.tight_layout()

        if save_path is not None:
            file = self._get_file_path(save_path, f'signal_{self.title}.png')
            fig.savefig(file)

        return ax

    def plot_2D(self, save_path: Optional[Path] = None,
                cmap: str = 'bwr',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None) -> plt.Axes:
        x_values = self.df.columns.values
        if all(isinstance(x, (int, float)) for x in x_values) is False:
            x_values = [str(x) for x in x_values]

        fig, ax = plt.subplots()
        im = ax.pcolormesh(x_values, self.df.index.values,
                           self.df,
                           cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(self.title)
        ax.set_xlabel(str(self.legend))
        ax.set_ylabel(str(self.x))
        cbar = fig.colorbar(im)
        cbar.set_label(str(self.y))
        fig.tight_layout()

        if save_path is not None:
            file = self._get_file_path(save_path,
                                       f'signal_2D_{self.title}.png')
            fig.savefig(file)

        return ax

    def save(self, save_path: Path) -> None:
        df = self.df.copy()
        col_qty = len(df.columns)
        df.columns = pd.MultiIndex.from_tuples(
                        zip([self.y.name]*col_qty,
                            [self.y.unit]*col_qty,
                            df.columns),
                        names=(self.x.name, self.x.unit, f'{self.legend}:'))

        file = self._get_file_path(save_path, f'signal_{self.title}.dat')
        df.to_csv(file, sep='\t')

    def fft(self, amp_mul: float = 1.0,
            amp_name: str = 'FFT Amplitude',
            amp_unit: str = '',
            freq_mul: float = 1.0,
            freq_name: str = 'Frequency',
            freq_unit: Optional[str] = None) -> 'Signal':
        """ Returns FFT of signal

        Parameters
        ----------
        amp_mul: FFT amplitude multiplicator
        freq_mul: frequency multiplicator
        """
        t = self.df.index.values
        time_step = (t[-1] - t[0]) / (t.size - 1)
        freq = fftpack.fftfreq(t.size, d=time_step) * freq_mul

        if freq_unit is None:
            freq_unit = TIME_TO_FREQ_UNITS.get(self.x.unit,
                                               f'1 / {self.x.unit}')

        sig_fft = Signal(freq, freq_name, freq_unit,
                         amp_name, amp_unit,
                         title=f'{self.title}_FFT',
                         legend_name=self.legend.name,
                         legend_unit=self.legend.unit)

        for lbl in self.df:
            sig_fft.add(np.abs(fftpack.fft(self.df[lbl].values)),
                        label=lbl)

        # Remove negative part
        sig_fft.df = sig_fft.df[sig_fft.df.index >= 0]

        return sig_fft
