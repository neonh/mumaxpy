"""
Utilities
"""
# %% Imports
import os
import glob
import re
import yaml
import tkinter.filedialog
import tkinter as tk
from astropy import units as u
from typing import Optional, List, Tuple, Dict, Union


# %% Types
# X, Y or Z
Ax = str
Path = str


# %% Constants
OS_FORBIDDEN_CHARS = r'<>:"/\|?*'
FORBIDDEN_CHARS = OS_FORBIDDEN_CHARS + ';$#\'"`'


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


def get_name_and_unit_from_str(s: str) -> Tuple[str, str]:
    """
    'Amp, urad' -> ('Amp', 'urad')
    'Amp (urad)' -> ('Amp', 'urad')
    """
    s = s.replace('(', ',').replace(')', '')
    splits = s.split(',')
    name = remove_forbidden_chars(splits[0]).strip()
    if len(splits) > 1:
        unit = splits[1].strip()
    else:
        unit = ''
    return (name, unit)


def extract_parameters(s: str) -> Dict[str, Tuple[float, str]]:
    param_dict = {}
    m = re.findall(r'(\w*)=([\d|.]*)(\w*)?', s)
    for param, value, unit in m:
        param_dict[param] = (float(value), unit)
    return param_dict


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
