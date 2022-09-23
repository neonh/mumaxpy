"""
Script for easy data processing
"""
import os
import numpy as np
from typing import Optional
from mumaxpy.utilities import get_filename, get_files_list
from mumaxpy.mat_data import MatFileData


# %% Types
# X, Y or Z
Ax = str
Path = str


# %% Functions
def probe_mask(ax_data, probe_coord, probe_D):
    probe_mask = np.exp(-2*((ax_data[0] - probe_coord[0])[:, np.newaxis]**2 +
                            (ax_data[1] - probe_coord[1])[np.newaxis, :]**2)
                        / probe_D**2)
    # Normalize
    probe_mask /= np.sum(probe_mask)
    return probe_mask


# %% Classes
class MatFilesProcessor:
    """
    Process (calculate, plot, save) data from all mat-files in selected folder

    Arguments for most methods is the same as in corresponding methods of
    MatFileData class
    """

    def __init__(self,
                 filename_mask: str = '*.mat',
                 input_folder: Optional[Path] = None,
                 output_folder: Optional[Path] = None,
                 recursive: bool = True) -> None:
        """
        Set input and output folders
        """
        self.files = get_files_list(filename_mask, input_folder, recursive)
        self.out_folder = output_folder

        self.methods = {}

        # Set default methods arguments
        self.convert_units(time_unit='ns', space_unit='um')

    def process(self) -> None:
        """
        Process all files
        """
        n = len(self.files)
        for i, f in enumerate(self.files):
            print(f'#{i+1} of {n}: {get_filename(f)}')

            if self.out_folder is None:
                out_folder = os.path.dirname(f)
            else:
                out_folder = self.out_folder

            # Load
            m = MatFileData.load(f)

            # Get vector component
            if 'get_component' in self.methods:
                args, kwargs = self.methods['get_component']
                m = m.get_component(*args, **kwargs)

            # Convert units
            if 'convert_units' in self.methods:
                args, kwargs = self.methods['convert_units']
                m.convert_units(*args, **kwargs)

            # Create amplitude plot and animation
            mat_plot_methods_list = ['plot_amplitude',
                                     'create_animation']
            for method_name in mat_plot_methods_list:
                if method_name in self.methods:
                    args, kwargs = self.methods[method_name]
                    kwargs['save_path'] = out_folder
                    method = getattr(m, method_name)
                    method(*args, **kwargs)

            # Get signals by probing
            if 'get_probe_signals' in self.methods:
                args, kwargs = self.methods['get_probe_signals']
                sig = m.get_probe_signals(*args, **kwargs)

                sig.plot(out_folder)
                sig.plot_2D(out_folder)
                sig.save(out_folder)

                # Get signals FFT
                if 'get_probe_signals' in self.methods:
                    sig_fft = sig.fft()
                    sig_fft.plot(out_folder)
                    sig_fft.plot_2D(out_folder)
                    sig_fft.save(out_folder)

    def delete(self) -> None:
        """
        Delete all files
        """
        for f in self.files:
            os.remove(f)

    # Methods to enable corresponding processing function for each file
    def get_component(self, *args, **kwargs) -> None:
        self.methods['get_component'] = (args, kwargs)

    def convert_units(self, *args, **kwargs) -> None:
        self.methods['convert_units'] = (args, kwargs)

    def plot_amplitude(self, *args, **kwargs) -> None:
        self.methods['plot_amplitude'] = (args, kwargs)

    def create_animation(self, *args, **kwargs) -> None:
        self.methods['create_animation'] = (args, kwargs)

    def get_probe_signals(self, *args, **kwargs) -> None:
        self.methods['get_probe_signals'] = (args, kwargs)

    def get_probe_signals_fft(self, *args, **kwargs) -> None:
        self.methods['get_probe_signals_fft'] = (args, kwargs)
