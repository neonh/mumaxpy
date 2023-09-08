"""
Utilities
"""
# %% Imports
import os
import glob
import re
import tkinter.filedialog
import tkinter as tk
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Iterable
import pandas as pd
import yaml
from astropy import units as u


# %% Types
# X, Y or Z
Ax = str
Path = str


# %% Constants
OS_FORBIDDEN_CHARS = r'<>:"/\|?*'
FORBIDDEN_CHARS = OS_FORBIDDEN_CHARS + ';$#\'"`'


# %% Classes
@dataclass
class NumberFormat:
    width: int
    precision: int
    exponent: int = 0
    sign: str = ''
    fill: str = '0'


# %% Functions
def init(file: Path) -> Tuple[Path, Path, Path]:
    # Open config file
    with open(file, encoding='utf8') as f:
        init_dict = yaml.safe_load(f)

    template_folder = init_dict['pathes']['mx3_templates']
    work_folder = init_dict['pathes']['mumax_output']
    data_folder = init_dict['pathes']['result_data']

    return template_folder, work_folder, data_folder


def get_filename(path: Path) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def get_files_list(filename_mask: str,
                   folder: Optional[Path] = None,
                   recursive: bool = True) -> List[Path]:
    if folder is None:
        folder = choose_folder()

    if recursive is True:
        files = glob.glob(os.path.join(folder, '**', filename_mask),
                          recursive=True)
    else:
        files = glob.glob(os.path.join(folder, filename_mask))

    return files


def get_unit(param: Union[u.Quantity, float]) -> str:
    if isinstance(param, u.Quantity):
        unit = param.unit
    else:
        unit = u.dimensionless_unscaled
    return unit


def remove_forbidden_chars(s: str) -> str:
    out = ''.join(c for c in s
                  if (ord(c) >= 32) and (c not in FORBIDDEN_CHARS))
    return out


def get_valid_dirname(s: str) -> str:
    """ Remove forbidden chars, strip spaces and replace dot at the end """
    out = remove_forbidden_chars(s).strip()
    if out[-1] == '.':
        out = out[:-1] + '!'
    return out


def get_name_and_unit_from_str(s: str) -> Tuple[str, str]:
    """
    'Amp, urad' -> ('Amp', 'urad')
    'Amp (urad)' -> ('Amp', 'urad')
    """
    COMMA = ','
    if COMMA in s:
        splits = s.split(COMMA)
    else:
        splits = s.replace(')', '').split('(')
    name = remove_forbidden_chars(splits[0]).strip()
    if len(splits) > 1:
        unit = splits[1].strip()
    else:
        unit = ''
    return (name, unit)


def extract_parameters(s: str) -> Dict[str, Tuple[float, str]]:
    param_dict = {}
    m = re.findall(r'(\w*)=([+-]?[\d.]*)(\w*)?', s)
    for param, value, unit in m:
        param_dict[param] = (float(value), unit)
    return param_dict


def read_dat_file(file: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """ Read dataframe and units from dat-file """
    with open(file) as f:
        # First row - variable names
        header = f.readline().split()
        # Second row - units
        units = f.readline().split()
        # Third row (optional) - comments, contains values for second index
        comments = f.readline().split('=')

    units_dict = {key: val for key, val in zip(header, units)}

    if len(comments) > 1:
        subheader_name_str = comments[0].strip()
        sh_name, sh_unit = get_name_and_unit_from_str(subheader_name_str)
        # Add to units dict
        units_dict[sh_name] = sh_unit
        # Subheader index values
        subheader = [float(x) for x in comments[1].split()]
        df = pd.read_csv(file, sep='\t', header=[0], skiprows=[1, 2],
                         index_col=0)
        df.columns = pd.MultiIndex.from_arrays([header[1:], subheader],
                                               names=('', sh_name))
    else:
        df = pd.read_csv(file, sep='\t', header=0, skiprows=[1])
    return df, units_dict


def create_hidden_window() -> tk.Tk:
    root = tk.Tk()
    root.attributes("-topmost", 1)
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.focus_force()
    return root


def choose_folder() -> Path:
    root = create_hidden_window()
    # Open dialog
    folder = tk.filedialog.askdirectory(parent=root)
    root.destroy()
    return folder


def choose_file(initialfile: Optional[str] = None,
                defaultextension: Optional[str] = None,
                filetypes: Iterable = [('All Files', '*.*')]) -> Optional[str]:
    root = create_hidden_window()
    # Open dialog
    file = tk.filedialog.askopenfilename(parent=root,
                                         initialfile=initialfile,
                                         defaultextension=defaultextension,
                                         filetypes=filetypes)
    root.destroy()
    if len(file) == 0:
        file = None
    return file


def msgbox(message: str,
           title: str = '',
           icon: str = 'info',
           cancel_button: bool = False) -> bool:
    root = create_hidden_window()
    if cancel_button:
        messagebox = tk.messagebox.askokcancel
    else:
        messagebox = tk.messagebox.showinfo
    ans = messagebox(parent=root, message=message, title=title, icon=icon)
    root.destroy()
    return ans


def find_common_number_format(numbers: Iterable[float]) -> NumberFormat:
    """
    Find suitable format to print all of the numbers to strings
    of equal lengths and precisions
    """
    # Strings in scientific format
    slist = [f'{x:e}' for x in numbers]

    # Sign
    sign = ''
    # Integer part
    d = []
    # Fractional part
    f = []
    # Exponent
    e = []
    for s in slist:
        m = re.search(r'(-)?(\d+)(?:\.(\d+))?(e[+|-]?\d+)?', s)
        if m.group(1) is not None:
            sign = '+'
        d += [len(m.group(2))]
        if m.group(3) is not None:
            f += [len(m.group(3).rstrip('0'))]
        else:
            f += [0]
        if m.group(4) is not None:
            e += [int(m.group(4)[1:])]
        else:
            e += [0]

    # Find common exponent if possible
    min_e = min(e)
    max_e = max(e)
    if max_e - min_e > 6:
        common_exp = None
    else:
        if max_e < 5 and min_e > -5:
            common_exp = 0
        else:
            # Choose closest 3*N value
            common_exp = (min_e // 3) * 3
        d = [d[i] + (e[i] - common_exp) for i in range(len(d))]
        f = [f[i] - (e[i] - common_exp) for i in range(len(f))]

    int_len = max(d)
    frac_len = max(f)
    if frac_len < 0:
        frac_len = 0

    fmt = NumberFormat(sign=sign,
                       width=len(sign) + int_len + frac_len + (frac_len > 0),
                       precision=frac_len,
                       exponent=common_exp)
    return fmt


def number_to_str(number: float, fmt: Optional[NumberFormat] = None) -> str:
    """Print number using specified format"""
    if fmt is None:
        out = f'{number:g}'
    else:
        s = '+' if fmt.sign else ''
        w = fmt.width
        p = fmt.precision
        e = fmt.exponent
        f = fmt.fill

        if e is None:
            # Scientific format
            out = f'{number:{f}={s}{w}.{p}e}'
        elif e == 0:
            # Float format
            out = f'{number:{f}={s}{w}.{p}f}'
        else:
            # Scientific format with fixed exponent value
            n = number / 10**fmt.exponent
            if isinstance(n, u.Quantity):
                out = f'{n.value:{f}={s}{w}.{p}f}e{e:+d} {n.unit}'
            else:
                out = f'{n:{f}={s}{w}.{p}f}e{e:+d}'

    return out
