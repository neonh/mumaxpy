"""
Simulation class
"""
import os
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import subprocess
import numpy as np
import pandas as pd
from .ovf import ovf_to_mat
from .utilities import get_name_and_unit_from_str, remove_forbidden_chars


# %% Constants
PRE = 0
POST = 1
TIME_COL = 0

X, Y, Z, ALL = 'x', 'y', 'z', ''
XYZ = [X, Y, Z]

TABLES = 'tables'
PLOTS = 'plots'
SCRIPTS = 'scripts'
MAT_FILES = 'mat_files'
SUB_DIRS = [TABLES, SCRIPTS]
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
        self.data_to_save = []
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
        if self.max_angle is not None:
            if MAX_ANGLE in df.columns:
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
            ax = df[cols].plot.line()
            ax.set_title(title)
            fig = ax.get_figure()
            fig.savefig(os.path.join(output_dir,
                                     f'{name}{fname}.png'))
            plt.close(fig)

    def _plot_results(self, df, output_dir):
        var_names = df.index.names
        if len(var_names) > 1:
            # Plot graph
            ax = df.unstack(level=0).plot.line()
        else:
            ax = df.plot.line()
        fig = ax.get_figure()
        fig.savefig(os.path.join(output_dir, 'results.png'))

        if len(var_names) > 1:
            for col in df.columns:
                name, _ = get_name_and_unit_from_str(col)
                ax = df[col].unstack(level=0).plot.line()
                ax.set_ylabel(col)
                fig = ax.get_figure()
                fig.savefig(os.path.join(output_dir, f'results_{name}.png'))

    def _create_result_dir(self, script_name, start_time, comment=''):
        # Create directory for results
        date_str = datetime.strftime(start_time, '%Y%m%d_%H%M')
        sub_dir = "_".join([date_str, remove_forbidden_chars(comment)])
        result_dir = os.path.join(self.data_dir,
                                  script_name, sub_dir)
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

    def _run_mumax(self, script_file, script):
        with open(script_file, 'w') as f:
            f.write(script)

        # Run
        command = ['mumax3', '-cache', self.cache_dir, script_file]
        ret = subprocess.run(command, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f'Mumax returned an error:\n{ret.stderr}')

    def set_max_angle(self, max_angle=None):
        self.max_angle = max_angle

    def set_callbacks(self, pre=None, post=None):
        self.callbacks = [pre, post]

    def set_process_func(self, f):
        self.process_func = f

    def add_table_plot(self, columns):
        self.table_columns_to_plot += [columns]

    def add_data_to_save(self, data, comp=ALL):
        self.data_to_save += [(data, comp.lower())]

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

    def run(self, script, variables=None, comment=''):
        output_dir = os.path.join(self.work_dir, f'{script.name}.out')
        if os.path.exists(output_dir):
            # Remove previous data from output directory
            shutil.rmtree(output_dir)
        script_file = os.path.join(self.work_dir, f'{script.name}.mx3')
        table_file = os.path.join(output_dir, 'table.txt')

        if variables is not None:
            # Dataframe with variables multiindex
            df = variables.get_df()
            var_names = df.index.names
            var_qty = len(variables)

            if var_qty > 1:
                # Create dict for colormapping
                v_list = variables.at(-2).values
                c_list = np.linspace(0, 0.9, len(v_list))
                color_dict = dict(zip(v_list, c_list))
        else:
            df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(0,)]))
            var_names = []
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
                v_list += [script.get_parameter_str(v)]
            var_str = ', '.join(v_list)
            if len(v_list) > 0:
                fvar_str = '_[' + ']_['.join(v_list).replace(' ', '') + ']'
            else:
                fvar_str = ''

            # %% Callback PRE
            if self.callbacks[PRE] is not None:
                script.parameters = self.callbacks[PRE](script.parameters)

            # Run
            print(f'#{iter_num:2} of {iter_qty} | {var_str}')
            self._run_mumax(script_file, script.text())

            time = datetime.now()
            duration_str = str(time - prev_time).split('.')[0]
            finish_time = (start_time
                           + ((time - start_time) / iter_num) * iter_qty)
            finish_time_str = datetime.strftime(finish_time, '%H:%M:%S %B %d')
            print(f'  {str(duration_str)} -> finish at ~{finish_time_str}')

            if iter_num == 1:
                self.result_dir = self._create_result_dir(script.name,
                                                          start_time, comment)
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
            for data, comp in self.data_to_save:
                ovf_to_mat(output_dir, data, comp,
                           output_dir=sub_dirs[MAT_FILES],
                           filename=f'{data}{comp}{fvar_str}')

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
            if self.process_func is not None:
                res_dict = self.process_func(table_df, table_units,
                                             script.parameters)

                if var_qty > 0:
                    if iter_num == 1:
                        # Create plot
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
        # %% Save figure
        if var_qty > 0:
            fig.savefig(os.path.join(self.result_dir, 'data.png'))

        if var_qty > 1:
            # Split data to columns by first index
            for col in df.columns:
                data_name, data_unit = get_name_and_unit_from_str(col)

                var = variables.at(0)
                v = var.name
                v_unit = str(script.get_parameter_unit(v))
                values = var.values

                idx = df.loc[values[0]].index
                new_df = pd.DataFrame(index=idx, columns=values)

                header = []
                for p in new_df.index.names:
                    unit = str(script.get_parameter_unit(p))
                    header += [(p, unit, '')]
                h = header[-1]
                # Add 'var_name (var_unit) = ' str
                header[-1] = [h[0], h[1], f'{v} ({v_unit}) =']

                for val in values:
                    new_df[val] = df.loc[val, col]
                    header += [(data_name, data_unit, val)]

                result_file = os.path.join(self.result_dir,
                                           f'results_{data_name}.dat')
                df_out = new_df.reset_index()
                df_out.columns = pd.MultiIndex.from_tuples(header)
                df_out.to_csv(result_file, sep='\t', index=False)

        # Create header with units:
        header = []
        # for variables
        for p in df.index.names:
            unit = str(script.get_parameter_unit(p))
            header += [(p, unit)]
        # for other output parameters
        for col in df.columns:
            header += [get_name_and_unit_from_str(col)]

        # %% Save result data
        result_file = os.path.join(self.result_dir, 'results.dat')
        df_out = df.reset_index()
        df_out.columns = pd.MultiIndex.from_tuples(header)
        df_out.to_csv(result_file, sep='\t', index=False)

        finish_time = datetime.now()
        print(f'<<<< Finished at {finish_time}')
        print(f'Elapsed time: {finish_time - start_time}')
        print('Results at', self.result_dir)

        # %% Plot and save all graphs
        if var_qty > 0:
            self._plot_results(df, self.result_dir)

        self.df = df
        return self.result_dir
