"""
Script for automated running of Mumax3 simulations
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from mumaxpy.simulation import Simulation
from mumaxpy.variables import Variables
from mumaxpy.mx3_script import Parameters
from mumaxpy.material import load_material
from mumaxpy.mat_data import MatFileData
from mumaxpy.run import run


# %% Script template filename
template = 'example.tmpl'
# Comment will be added to results folder name
comment = ''


# %% Parameters
def create_parameters(p: Parameters) -> Parameters:
    """
    Add any parameter to use in mumax script, for example:
        p.Ddisk = 100 * u.um

    Parameters
    ----------
    p : Parameters

    Returns
    -------
    Parameters
    """
    # Times
    p.quasystationaryTime = 10 * u.ns
    p.measureTime = 1 * u.ns
    p.relaxTime = 0 * u.ns

    # Geometry
    p.Ddisk = 50 * u.um
    p.h_film = 1 * u.um

    # External field
    p.B = 200 * u.mT
    # Excitations
    p.Hf = 1 * u.Oe
    p.freq = 2 * u.GHz

    # Solver settings
    p.max_dt = 20 * u.ps
    # Save settings
    p.tableSavePeriod = p.max_dt
    p.ovfSavePeriod = p.max_dt
    p.saveOVF = True

    # Grid
    N = 32
    cells_num = [N, N, 1]
    size = [p.Ddisk, p.Ddisk, p.h_film] * u.um
    p.set_grid(size, cells_num)

    # Material
    m = load_material('YIG', 'materials.yml')
    p.set_material(m)

    return p


# %% Variables
def create_variables(v: Variables) -> Variables:
    """
    Add variables and their values to set before each mumax run.
    Each next added variable will alternate faster than the previous.
    For example:
        v.add('Hf', [1, 2, 3])
        v.add('B', np.r_[200: 300: 2])
    The above is similar to the following nested FOR cycles:
        for Hf in [1, 2, 3]:
            p.Hf = Hf
            for B in np.r_[200: 300: 2]:
                p.B = B
                ...

    NB! Unit of variables will be the same as configured in Parameters.

    Parameters
    ----------
    v : Variables

    Returns
    -------
    Variables
    """
    v.add('Hf', [1, 5])
    v.add('B', np.r_[215: 235.01: 1])

    return v


# %% Callbacks
def pre_run(p: Parameters) -> Parameters:
    """
    Callback function - called BEFORE each next mumax run.
    Change any existing parameter, for example:
        p.gridNx = p.gridNy

    Parameters
    ----------
    p : Parameters

    Returns
    -------
    Parameters
    """

    return p


def post_run(current_results: pd.DataFrame, p: Parameters) -> bool:
    """
    Callback function - called AFTER each next mumax run.

    Parameters
    ----------
    current_results : pd.DataFrame
        Dataframe with current results (generated by data_process_func).
    p : Parameters

    Returns
    -------
    bool
        If returns False - simulation will stop.
    """

    return True


# %% Output data processing
def data_process_func(data: pd.DataFrame, units: dict, p: Parameters) -> dict:
    """
    Process raw data from mumax table.txt file and return some results.
    For example:
        return {'B_z maximum, T': data['B_extz'].max()}

    Parameters
    ----------
    data : pd.DataFrame
        index - t(s),
        columns - mx, my, mz and all other columns from table.txt.
    units : dict
        keys - data column names, values - their units.
    p : Parameters

    Returns
    -------
    dict
        keys - any name of result data with optional unit (comma separated),
        values - result data value.
    """
    # quasystationaryTime in seconds
    qs_time = p.quasystationaryTime.to('s').value

    # Get quasystationary data (t > qs_time)
    qs_data = data[data.index > qs_time]

    # mx in region 1
    mx = qs_data['m.region1x']
    # Calculate maximum angle (theta) of magnetization precession
    theta = np.arcsin((mx.max() - mx.min()) / 2)
    # Mean total energy in nanojoules
    energy = qs_data['E_total'].mean() * 1e9

    result_dict = {r'$\theta$, rad': theta,
                   'E, nJ': energy}
    return result_dict


# %% Add plots and data to save
def add_output(sim: Simulation, p: Parameters):
    """
    Configure output data and plots saving.
    For example:
        sim.add_table_plot(columns=['MaxAngle'])
        sim.add_data_to_save(data='m', comp='z')

    Parameters
    ----------
    sim : Simulation
    p : Parameters

    Returns
    -------
    None
    """
    # Table plots
    sim.add_table_plot(columns=['m.region1x', 'm.region1y'])
    sim.add_table_plot(columns=['MaxAngle'])

    # OVF data
    if p.saveOVF:
        sim.add_data_to_save(data='m', comp='x')

    return None


# %% Postprocessing
def post_processing(result_dir: str, p: Parameters):
    """
    Function to process data in result directory after simulations are done.

    Parameters
    ----------
    result_dir : str
        Path to directory with results.
    p : Parameters

    Returns
    -------
    None
    """
    def add_disk_plot(ax):
        # Plot circles for disks
        d = p.Ddisk.to('um').value
        circle = plt.Circle((0, 0), d/2, color='black', fill=False)
        ax.add_patch(circle)

    # Load all .mat files in result_dir and subdirs
    mat_files = glob.glob(os.path.join(result_dir, "**", "*.mat"),
                          recursive=True)

    for i, f in enumerate(mat_files):
        folder = os.path.dirname(f)
        fname = os.path.splitext(os.path.basename(f))[0]
        print(f'#{i} of {len(mat_files)}: {fname}')

        m = MatFileData(f)
        m.convert_units('ns', 'um')
        m.plot_amplitude(normal='z', add_plot_func=add_disk_plot,
                         save_path=folder)

        m.create_animation(normal='z', add_plot_func=add_disk_plot,
                           save_path=folder, extension='mp4')
        m.delete()
        plt.close('all')

    return None


# %% Verify parameters and variables before simulation
def verify(p: Parameters, v: Variables) -> bool:
    """
    Function to verify parameters and variables before starting the simulation.

    Parameters
    ----------
    p : Parameters
    v : Variables

    Returns
    -------
    bool
        True if verification succeeded.
    """
    is_ok = True
    if p.saveOVF:
        SIZEOF_DATA = 4 * u.B
        VECTOR_COMP_QTY = 3
        iter_qty = v.get_full_length()
        ovf_qty = (p.measureTime + p.relaxTime) / p.ovfSavePeriod
        cells_qty = p.gridN_x * p.gridN_y * p.gridN_z
        ovf_size = cells_qty * VECTOR_COMP_QTY * SIZEOF_DATA
        single_ovf_size = (ovf_qty * ovf_size).to(u.GB)
        all_ovf_size = iter_qty * single_ovf_size

        print('OVF data size:',
              f'{single_ovf_size:g} x {iter_qty} files = {all_ovf_size:g}')

        if ((single_ovf_size > 0.5 * u.GB)
                or (all_ovf_size > 4 * u.GB)):
            print('WARNING! Check OVF files size')
            is_ok = False

    return is_ok


# %% Main
if __name__ == "__main__":
    # Create parameters
    p = create_parameters(Parameters())

    # Create variables
    v = create_variables(Variables())

    init_file = 'init.yml'
    this_file = os.path.realpath(__file__)
    # Run mumax simulations
    result_dir = run(template,
                     this_file,
                     init_file,
                     p, v, pre_run, post_run,
                     data_process_func, add_output,
                     verify, post_processing,
                     comment)
