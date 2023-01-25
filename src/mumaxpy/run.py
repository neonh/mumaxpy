"""
Run mumax
"""
# %% Imports
import os
import shutil
import traceback
import glob
from datetime import datetime
from typing import Callable
import matplotlib.pyplot as plt

from mumaxpy import __version__
from mumaxpy.simulation import Simulation
from mumaxpy.variables import Variables
from mumaxpy.mx3_script import Parameters, Script
from mumaxpy.mat_data import MatFileData
from mumaxpy.utilities import init, msgbox


# %% Constants
MAX_ANGLE_THRESHOLD = 0.35  # rad


# %% Functions
def run(template_filename: str,
        main_script_file: str,
        init_file: str = None,
        parameters: Parameters = None,
        variables: Variables = None,
        pre_run: Callable = None,
        post_run: Callable = None,
        data_process_func: Callable = None,
        add_output: Callable = None,
        verify: Callable = None,
        post_processing: Callable = None,
        comment: str = '',
        plot_geom: bool = True) -> str:
    """
    Configure and run mumax simulations
    """
    # Current directory
    cur_dir = os.path.dirname(main_script_file)
    # Get directories
    if init_file is not None and os.path.isfile(init_file):
        template_dir, work_dir, data_dir = init(init_file)
    else:
        template_dir, work_dir, data_dir = cur_dir, cur_dir, cur_dir

    template_file = os.path.join(template_dir, template_filename)
    if not os.path.isfile(template_file):
        template_file = os.path.join(cur_dir, template_filename)
        if not os.path.isfile(template_file):
            err_msg = f'Template "{template_filename}" is not found!'
            msgbox(message=err_msg, title='Error', icon='error')
            raise FileNotFoundError(err_msg)

    print('Parameters:')
    print(parameters)

    print('Variables:')
    print(variables)

    print(f'\nIterations number: {variables.get_full_length()}\n')

    parameters.print_grid_parameters()

    # Verify
    if verify is not None:
        if verify(parameters, variables) is False:
            dialog_text = 'Verification failed!\n\n'
            icon = 'warning'
        else:
            dialog_text = ''
            icon = 'info'

    # Create script configuration object
    script = Script(template_file)
    script.set_parameters(parameters)
    script.check()

    # Create simulation object
    sim = Simulation(work_dir, data_dir)
    sim.set_callbacks(pre_run, post_run)
    sim.set_process_func(data_process_func)
    sim.set_max_angle(MAX_ANGLE_THRESHOLD)

    # Plot geometry view
    if plot_geom:
        geom_dir = sim.run_geometry_test(script, variables)
        m = MatFileData.load(os.path.join(geom_dir, 'geom.mat'))
        m.convert_units(space_unit=str(parameters.gridSize_x.unit))
        m.plot_volume(opacity=0.5, surf_count=3, save_path=geom_dir)

    # Configure data
    if add_output is not None:
        add_output(sim, parameters)

    dialog_text += 'Check console output...'
    dialog_ret = msgbox(message=dialog_text, title='Continue?', icon=icon,
                        cancel_button=True)

    if dialog_ret is True:
        # Close prevoius plots
        plt.close('all')

        try:
            # Run simulation using selected variables and script configuration
            result_dir = sim.run(script, variables, comment=comment)
        except KeyboardInterrupt as e:
            print(e, '\nTerminated by user!')
            result_dir = sim.get_result_dir()
        except Exception as e:
            err_msg = f'{type(e).__name__}: {e}'
            msgbox(message=err_msg, title='Simulation error', icon='error')
            raise

        # Print errors
        print(sim.get_errors_str())

        if result_dir is not None:
            scripts_dir = sim.get_scripts_dir()
            # Copy template and this script to results folder
            shutil.copy2(main_script_file, scripts_dir)
            shutil.copy2(template_file, scripts_dir)
            yml_files = glob.glob(os.path.join(cur_dir, '*.yml'))
            for f in yml_files:
                shutil.copy2(f, scripts_dir)
            # Save empty file with version in its filename
            version_file = os.path.join(scripts_dir,
                                        f'mumaxpy-{__version__}.version')
            open(version_file, 'w+').close()

        if post_processing is not None:
            print('>>>> Running post_processing function...')
            start_time = datetime.now()

            try:
                # Do postprocessing
                post_processing(result_dir, parameters)
                print('<<<< Post-processing completed')
            except Exception as e:
                err_msg = f'{type(e).__name__}: {e}'
                msgbox(message=err_msg,
                       title='Post-processing error', icon='error')
                print(traceback.format_exc())
                print('<<<< Post-processing FAILED')

            finish_time = datetime.now()
            print(f'Elapsed time: {finish_time - start_time}')

        # Don't close the plots
        plt.show()
    else:
        print('\nCanceled...')
        result_dir = None

    return result_dir
