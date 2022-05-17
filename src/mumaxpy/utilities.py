"""
Utilities
"""
import os
import re
import yaml
import tkinter as tk
from astropy import units as u


# %% Constants
OS_FORBIDDEN_CHARS = r'<>:"/\|?*'
FORBIDDEN_CHARS = OS_FORBIDDEN_CHARS + ';$#\'"`'


# %% Functions
def init(file):
    # Open config file
    with open(file, encoding='utf8') as f:
        init_dict = yaml.safe_load(f)

    template_folder = init_dict['pathes']['mx3_templates']
    work_folder = init_dict['pathes']['mumax_output']
    data_folder = init_dict['pathes']['result_data']

    return template_folder, work_folder, data_folder


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_unit(param):
    if isinstance(param, u.Quantity):
        unit = param.unit
    else:
        unit = u.dimensionless_unscaled
    return unit


def remove_forbidden_chars(s):
    out = ''.join(c for c in s
                  if (ord(c) >= 32) and (c not in FORBIDDEN_CHARS))
    return out


def get_name_and_unit_from_str(s):
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


def extract_parameters(s):
    param_dict = {}
    m = re.findall(r'(\w*)=([\d|.]*)(\w*)?', s)
    for param, value, unit in m:
        param_dict[param] = (float(value), unit)
    return param_dict


def create_hidden_window():
    root = tk.Tk()
    root.attributes("-topmost", 1)
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.focus_force()
    return root


def choose_folder():
    root = create_hidden_window()
    # Open dialog
    folder = tk.filedialog.askdirectory(parent=root)
    root.destroy()
    return folder


def msgbox(message, title='', icon='info', cancel_button=False):
    root = create_hidden_window()
    if cancel_button:
        messagebox = tk.messagebox.askokcancel
    else:
        messagebox = tk.messagebox.showinfo
    ans = messagebox(parent=root, message=message, title=title, icon=icon)
    root.destroy()
    return ans
