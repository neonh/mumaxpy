"""
Simulation class
"""
# %% Imports
import os
import re
from datetime import datetime
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mumaxpy.ovf import ovf_to_mat
from mumaxpy.utilities import (get_name_and_unit_from_str,
                               get_valid_dirname, number_to_str,
                               remove_forbidden_chars)


# %% Constants
PRE = 0
POST = 1
TIME_COL = 0

X, Y, Z, ALL = 'x', 'y', 'z', ''
XYZ = [X, Y, Z]

TABLES = 'tables'
RESULTS = 'results'
PLOTS = 'plots'
SCRIPTS = 'scripts'
MAT_FILES = 'mat_files'
SUB_DIRS = [TABLES, SCRIPTS, RESULTS]
OPT_SUB_DIRS = [PLOTS, MAT_FILES]

MAX_ANGLE = 'MaxAngle'


# %% Classes
class Simulation:
    def __init__(self, work_dir, data_dir=None):
        self.callbacks = [None, None]
        self.process_func = None

        self.work_dir = work_dir
        self.cache_dir = os.path.join(self.work_dir, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = self.work_dir

        self.df = None
        self.table_columns_to_plot = []

        self.max_angle = None
        self.errors = {}
        self.warnings = {}

        self.result_dir = None

    def _get_table_data(self, file):
        df = pd.read_csv(file, sep="\t", index_col=TIME_COL)

        n_u_tuple = [get_name_and_unit_from_str(c) for c in list(df.columns)]
        df.columns = [t[0] for t in n_u_tuple]
        return df, dict(n_u_tuple)

    def _check_max_angle(self, df):
        s = None
        if self.max_angle is not None and MAX_ANGLE in df.columns:
            m_angle = df[MAX_ANGLE]
            maximum = max(m_angle)
            print(f'MaxAngle={maximum:.6f} rad')

            if maximum > self.max_angle:
                it_qty = len(df)
                th_it_qty = len(df[df[MAX_ANGLE] > self.max_angle])
                p = th_it_qty / it_qty
                s = ('WARNING! MaxAngle was > '
                     + f'{self.max_angle:.2f} rad during {p:.1%} '
                     + 'of simulation time.')
                print(s)
        return s

    def _plot_table(self, df, output_dir, title='', fname=''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for cols in self.table_columns_to_plot:
            name = '_'.join(cols)
            fig, ax = plt.subplots()
            df[cols].plot.line(ax=ax)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     f'{name}{fname}.png'))
            plt.close(fig)

    def _plot_results(self, df, output_dir):
        if len(df.index.names) > 1:
            # Unstack multiindex to columns
            lvls_to_unstack = list(range(len(df.index.names) - 1))
            tmp_df = df.unstack(lvls_to_unstack)
        else:
            tmp_df = df
            tmp_df.index = df.index.get_level_values(0)

        for col in df.columns:
            fig, ax = plt.subplots()
            name, _ = get_name_and_unit_from_str(col)
            tmp_df[col].plot.line(ax=ax)
            ax.set_ylabel(col)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'results_{name}.png'))

    def _create_result_dir(self, script_name, start_time,
                           var_str='', comment=''):
        # Create directory for results
        sub_dir_name = datetime.strftime(start_time, '%Y%m%d_%H%M')
        if var_str != '':
            sub_dir_name += var_str
        if comment != '':
            sub_dir_name += f'_{comment}'
        result_dir = os.path.join(self.data_dir,
                                  script_name,
                                  get_valid_dirname(sub_dir_name))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    def _create_sub_dirs(self, result_dir):
        sub_dirs_dict = {}
        # Add and create sub directories
        for name in SUB_DIRS:
            sub_dir = os.path.join(result_dir, name)
            sub_dirs_dict.update({name: sub_dir})
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
        # Add optional sub directories (created later if needed)
        for name in OPT_SUB_DIRS:
            sub_dir = os.path.join(result_dir, name)
            sub_dirs_dict.update({name: sub_dir})
        return sub_dirs_dict

    def _run_mx3_script(self, file, script, param_lines_qty = 0):
        with open(file, 'w') as f:
            f.write(script)

        # Run
        command = ['mumax3', '-cache', self.cache_dir, file]
        ret = subprocess.run(command, capture_output=True, text=True)
        if ret.returncode != 0:
            out_str = f'Mumax returned an error:\n{ret.stderr}'

            m = re.search(r'script line (\d+):', ret.stderr)
            if m is not None:
                err_line = int(m.group(1))
                out_str += ('\n---' +
                            f'\nCheck line {err_line} '
                            f'in generated file {file}')

                tmpl_err_line = err_line - param_lines_qty - 1
                if tmpl_err_line > 0:
                    out_str += f'\nor line {tmpl_err_line} in template file'
            raise RuntimeError(out_str)

    def set_max_angle(self, max_angle=None):
        self.max_angle = max_angle

    def set_callbacks(self, pre=None, post=None):
        self.callbacks = [pre, post]

    def set_process_func(self, f):
        self.process_func = f

    def add_table_plot(self, columns):
        self.table_columns_to_plot += [columns]

    def get_result_dir(self):
        return self.result_dir

    def get_scripts_dir(self):
        return os.path.join(self.result_dir, SCRIPTS)

    def get_errors_str(self):
        err_str = ''
        if len(self.warnings) > 0:
            err_str += '\nWARNINGS:\n'
            for key, val in self.warnings.items():
                err_str += f'{key}: {val}\n'
        if len(self.errors) > 0:
            err_str += '\nERRORS:\n'
            for key, val in self.errors.items():
                err_str += f'{key}: {val}\n'
        return err_str

    def get_result_data(self):
        return self.df

    def run_geometry_test(self, script, variables=None):
        output_dir = os.path.join(self.work_dir, f'{script.name}.out')
        script_file = os.path.join(self.work_dir, f'{script.name}.mx3')

        # Set first value for all variables
        if variables is not None and len(variables) > 0:
            v = variables.get_df().index
            for name, val in zip(v.names, v[0]):
                script.modify_parameter(name, val)
        # Run PRE callback
        if self.callbacks[PRE] is not None:
            script.parameters = self.callbacks[PRE](script.parameters)

        # Run test geom script
        self._run_mx3_script(script_file,
                             script.geom_test_text(),
                             script.get_parameter_lines_qty())
        # Convert to mat-file
        ovf_to_mat(input_dir=output_dir)

        return output_dir

    def run(self, script, variables=None, comment=''):
        output_dir = os.path.join(self.work_dir, f'{script.name}.out')
        if os.path.exists(output_dir):
            # Remove previous data from output directory
            shutil.rmtree(output_dir)
        script_file = os.path.join(self.work_dir, f'{script.name}.mx3')
        table_file = os.path.join(output_dir, 'table.txt')

        var_vals_str = ''
        if variables is not None and len(variables) > 0:
            # Dataframe with variables multiindex
            df = variables.get_df()
            var_names = df.index.names
            var_units = [str(script.get_parameter_unit(v))
                         for v in var_names]
            var_qty = len(variables)

            for v in variables.get():
                var_vals_str += '_{' + f'{v.name}={min(v.values):g}'
                if len(v.values) > 1:
                    var_vals_str += f'-{max(v.values):g}'
                var_vals_str += str(script.get_parameter_unit(v.name)) + '}'

            if var_qty > 1:
                # Create dict for colormapping
                v_list = variables.at(-2).values
                c_list = np.linspace(0, 0.9, len(v_list))
                color_dict = dict(zip(v_list, c_list))
        else:
            df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(0,)]))
            var_names = []
            var_units = []
            var_qty = 0

        # %% Start
        start_time = datetime.now()
        print(f'\n>>>> Started at {start_time}\n')
        prev_time = start_time
        iter_num = 1
        iter_qty = len(df)

        # %% Main iterations trough variable values
        for var_value, df_row in df.iterrows():
            # Update script with variables
            v_list = []
            for i, v in enumerate(var_names):
                script.modify_parameter(v, var_value[i])
                v_list += [script.get_parameter_str(v, variables.get(v).fmt)]
            var_str = ', '.join(v_list)
            if len(v_list) > 0 and v_list[0] != '':
                fvar_str = '_{' + '}_{'.join(v_list).replace(' ', '') + '}'
                fvar_str = remove_forbidden_chars(fvar_str)
            else:
                fvar_str = ''

            # %% Callback PRE
            if self.callbacks[PRE] is not None:
                script.parameters = self.callbacks[PRE](script.parameters)

            # Run
            print(f'#{iter_num:2} of {iter_qty} | {var_str}')
            self._run_mx3_script(script_file,
                                 script.text(),
                                 script.get_parameter_lines_qty())

            time = datetime.now()
            duration_str = str(time - prev_time).split('.')[0]
            finish_time = (start_time
                           + ((time - start_time) / iter_num) * iter_qty)
            finish_time_str = datetime.strftime(finish_time, '%H:%M:%S %B %d')
            print(f'  {str(duration_str)} -> finish at ~{finish_time_str}')

            if iter_num == 1:
                self.result_dir = self._create_result_dir(script.name,
                                                          start_time,
                                                          var_vals_str,
                                                          comment)
                sub_dirs = self._create_sub_dirs(self.result_dir)
                # Copy script to results and rename
                shutil.copy2(script_file, os.path.join(sub_dirs[SCRIPTS],
                                                       '__script.mx3'))
                # For plots
                plt.style.use('seaborn-whitegrid')

            # Move table file to output dir and rename
            out_dir = os.path.join(sub_dirs[TABLES], *v_list[:-1])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            table_copy_file = os.path.join(out_dir, f'table{fvar_str}.txt')
            shutil.move(table_file, table_copy_file)

            # %% Convert ovf files
            out_dir = os.path.join(sub_dirs[MAT_FILES], *v_list[:-1])
            ovf_to_mat(input_dir=output_dir,
                       output_dir=out_dir,
                       filename_suffix=fvar_str,
                       extra_data=script.parameters.get_param_dict())

            # %% Get table data
            table_df, table_units = self._get_table_data(table_copy_file)
            warn = self._check_max_angle(table_df)
            if warn is not None:
                self.warnings[f'#{iter_num:2} {var_str}'] = warn

            # Plot table data
            self._plot_table(table_df,
                             os.path.join(sub_dirs[PLOTS], *v_list[:-1]),
                             title=var_str,
                             fname=fvar_str)

            # %% Process table data
            if self.process_func is not None and var_qty > 0:
                res_dict = self.process_func(table_df, table_units,
                                             script.parameters)

                if iter_num == 1:
                    # Create plot for current data
                    fig, axs = plt.subplots(len(res_dict))
                    # In case of only one ax:
                    if not hasattr(axs, '__iter__'):
                        axs = [axs]
                    for i, key in enumerate(res_dict):
                        # Set title and lables
                        axs[i].set_ylabel(key)
                    x_str = script.get_parameter_name_str(var_names[-1])
                    axs[-1].set_xlabel(x_str)

                for i, (key, val) in enumerate(res_dict.items()):
                    df.loc[var_value, key] = val
                    # Plot
                    if var_qty > 1:
                        color = plt.cm.brg(color_dict[var_value[-2]])
                    else:
                        color = 'blue'
                    axs[i].scatter(var_value[-1], val,
                                   c=[color], alpha=0.8)
                    plt.pause(0.05)
                    plt.show(block=False)

            prev_time = time
            iter_num += 1

            # %% Callback POST
            if self.callbacks[POST] is not None:
                if self.callbacks[POST](df, script.parameters) is False:
                    # Stop iterations
                    cb_name = self.callbacks[POST]
                    print(f'STOP: {cb_name} callback returned False!')
                    break
            print('')

        # After all of iterations #
        # %% Plot and save all graphs
        if var_qty > 0:
            self._plot_results(df, sub_dirs[RESULTS])
            # Save current data plot
            fig.tight_layout()
            fig.savefig(os.path.join(self.result_dir, 'data.png'))

        # %% Save results data organized into folders
        if var_qty > 1:
            # Create first column header
            row_var = df.index.names[-1]
            col_var = df.index.names[-2]
            r_var_unit = str(script.get_parameter_unit(row_var))
            c_var_unit = str(script.get_parameter_unit(col_var))
            header_0 = [(row_var,
                         r_var_unit,
                         f'{col_var} ({c_var_unit}) =')]

            # Get var values except last 2 levels
            var_vals = set([x[:-2] for x in df.index.values])

            # Split data by folders (\<var_name>=<var_value> <var_unit>\)
            for val in var_vals:
                sub_dirs_list = [f'{x[0]}='
                                 + number_to_str(x[1], variables.get(x[0]).fmt)
                                 + f' {x[2]}'
                                 for x in zip(var_names[:-2],
                                              val, var_units[:-2])]
                out_dir = os.path.join(sub_dirs[RESULTS], *sub_dirs_list)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                fstr = '}_{'.join(sub_dirs_list).replace(' ', '')
                if var_qty > 2:
                    fstr = '_{' + fstr + '}'

                # Save each result column data
                for d in df.columns:
                    data_name, data_unit = get_name_and_unit_from_str(d)
                    fname = f'results_{data_name}{fstr}'
                    header = header_0 + [(data_name, data_unit, v)
                                         for v in variables.at(-2).values]
                    tmp_df = df[d].loc[val].unstack(df.index.names[-2])

                    # Plot
                    fig, ax = plt.subplots()
                    tmp_df.plot.line(ax=ax)
                    ax.set_ylabel(d)
                    ax.set_xlabel(f'{var_names[-1]}, {var_units[-1]}')
                    legend = ax.get_legend()
                    legend.set_title(f'{var_names[-2]}, {var_units[-2]}')
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, f'{fname}.png'))
                    plt.close(fig)
                    # Save to text file
                    result_file = os.path.join(out_dir, f'{fname}.dat')
                    df_out = tmp_df.reset_index()
                    df_out.columns = pd.MultiIndex.from_tuples(header)
                    df_out.to_csv(result_file, sep='\t', index=False)

        # %% Save full results data
        # Create header with units:
        header = []
        # for variables
        for p in df.index.names:
            unit = str(script.get_parameter_unit(p))
            header += [(p, unit)]
        # for other output parameters
        for col in df.columns:
            header += [get_name_and_unit_from_str(col)]

        result_file = os.path.join(sub_dirs[RESULTS], 'results.dat')
        df_out = df.reset_index()
        df_out.columns = pd.MultiIndex.from_tuples(header)
        df_out.to_csv(result_file, sep='\t', index=False)

        finish_time = datetime.now()
        print(f'<<<< Finished at {finish_time}')
        print(f'Elapsed time: {finish_time - start_time}')
        print('Results at', self.result_dir)

        self.df = df
        return self.result_dir
