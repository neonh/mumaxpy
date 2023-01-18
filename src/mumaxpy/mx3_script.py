"""
Script configuration classes
"""
# %% Imports
import re
from astropy import units as u
from mumaxpy.material import Material
from mumaxpy.utilities import get_unit, get_filename, number_to_str

# Add Oersted unit
if not hasattr(u, 'Oe'):
    u.Oe = u.def_unit('Oe', 1e-4 * u.T)


# %% Constants
SPACE = ' '
COMMENTS_INDENT = 44
VECTOR_COMP = ['x', 'y', 'z']


# %% Classes
class Parameters:
    def __init__(self):
        pass

    def __setattr__(self, attr, val):
        if hasattr(self, attr):
            # Check units
            old_val = getattr(self, attr)
            if isinstance(old_val, u.Quantity):
                old_unit = old_val.unit
            else:
                old_unit = 1

            if isinstance(val, u.Quantity):
                unit = val.unit
                if unit.is_equivalent(old_unit):
                    super().__setattr__(attr, val)
                else:
                    error = f'Wrong unit: {attr} should be in {old_unit}'
                    raise AttributeError(error)
            else:
                super().__setattr__(attr, val * old_unit)
        else:
            # New attribute
            super().__setattr__(attr, val)

    def set_material(self, m: Material):
        self.fourPiMs = m.fourPiMs
        self.Ku_1 = m.Ku_1
        self.Ku_2 = m.Ku_2
        self.Kc_1 = m.Kc_1
        self.Kc_2 = m.Kc_2
        self.Kc_3 = m.Kc_3
        self.alpha_Gilbert = m.alpha_Gilbert
        self.A_ex = m.A_ex

    def set_grid(self, size, cells_num, pbc=[0, 0, 0]):
        if (len(size), len(cells_num), len(pbc)) != (3, 3, 3):
            err_msg = 'set_grid: Arguments should be lists of 3'
            raise ValueError(err_msg)

        for i in range(3):
            if ((isinstance(size[i], u.Quantity)
                 and size[i].unit.is_equivalent(u.m))):
                comp = VECTOR_COMP[i]
                setattr(self, f'gridSize_{comp}', size[i])
                setattr(self, f'gridN_{comp}', int(cells_num[i]))
                setattr(self, f'pbc_{comp}', int(pbc[i]))
            else:
                err_msg = 'set_grid: Grid size should be in units of length'
                raise ValueError(err_msg)

    def print_grid_parameters(self):
        s = f'Grid: {self.gridN_x} x {self.gridN_y} x {self.gridN_z} cells\n'
        s += ('Grid size: '
              f'{self.gridSize_x} x {self.gridSize_y} x {self.gridSize_z}\n')
        cx = self.gridSize_x / self.gridN_x
        cy = self.gridSize_y / self.gridN_y
        cz = self.gridSize_z / self.gridN_z
        s += f'Cell size: {cx} x {cy} x {cz}\n'
        if self.pbc_x > 0 or self.pbc_y > 0 or self.pbc_z > 0:
            s += f'PBC: {self.pbc_x} x {self.pbc_y} x {self.pbc_z} => '
            lx, ly, lz = (self.gridSize_x * (2*self.pbc_x + 1),
                          self.gridSize_y * (2*self.pbc_y + 1),
                          self.gridSize_z * (2*self.pbc_z + 1))
            s += ('Effective grid size: '
                  f'{lx} x {ly} x {lz}\n')
        print(s)

    def get_param_names(self):
        names_list = [name for name, val in vars(self).items()]
        return names_list

    def get_param_str(self):
        param_list = []
        for name, val in vars(self).items():
            if not name.startswith('_'):
                if isinstance(val, u.Quantity):
                    # Convert to SI
                    v = val.si
                    s = f'{name} := {v.value:g} // [{v.unit}]'
                else:
                    s = f'{name} := {val:g}'
                space_cnt = COMMENTS_INDENT - len(s)
                s += SPACE*space_cnt + f'// {name} = {val}'
                param_list += [s]
        s = '\n'.join(param_list) + '\n'
        return s

    def get_param_dict(self):
        return {name: str(val) for name, val in vars(self).items()}

    def __str__(self):
        s = ''
        for name, val in vars(self).items():
            if not name.startswith('_'):
                s += f'{name} = {val:g}\n'
        return s


class Script:
    def __init__(self, file):
        self.name = get_filename(file)
        with open(file, 'r') as f:
            self.script = f.read()
        self.parameters = None

    def set_parameters(self, parameters):
        self.parameters = parameters

    def flush_parameters(self):
        self.parameters = None

    def check(self):
        is_ok = True
        p_names_list = [name for name in self.parameters.get_param_names()
                        if not name.startswith('_')]
        for name in p_names_list:
            m = re.search(r'\b{}\b'.format(name), self.script,
                          flags=re.IGNORECASE)
            if m is None:
                print(f'WARNING: Parameter {name} is not used in template!')
                is_ok = False
        return is_ok

    def modify_parameter(self, parameter, new_value):
        if parameter is not None:
            if hasattr(self.parameters, parameter):
                setattr(self.parameters, parameter, new_value)
            else:
                raise AttributeError(f'{parameter} is not in Parameters')

    def get_parameter(self, parameter):
        if hasattr(self.parameters, parameter):
            p = getattr(self.parameters, parameter)
        else:
            p = None
        return p

    def get_parameter_name_str(self, parameter):
        if parameter is not None:
            p = getattr(self.parameters, parameter)
            s = ', '.join([parameter, str(get_unit(p))])
        else:
            s = parameter
        return s

    def get_parameter_unit(self, parameter):
        if parameter is not None:
            p = getattr(self.parameters, parameter)
            unit = get_unit(p)
        else:
            unit = ''
        return unit

    def get_parameter_str(self, parameter, fmt=None):
        if parameter is not None:
            value = getattr(self.parameters, parameter)
            s = f'{parameter}=' + number_to_str(value, fmt)
        else:
            s = ''
        return s

    def get_parameter_lines_qty(self):
        return self.parameters.get_param_str().count('\n')

    def text(self):
        if self.parameters is not None:
            param_str = self.parameters.get_param_str()
        else:
            param_str = ''
        text = '\n'.join([param_str, self.script])
        return text

    def save(self, file):
        pass

    def load(self, file):
        pass
