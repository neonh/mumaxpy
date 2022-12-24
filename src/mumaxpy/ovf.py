"""
Processing OVF files
"""
# %% Imports
import glob
import os
import re
import struct
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.io import savemat

from mumaxpy.utilities import get_filename


# %% Types
Path = str


# %% Constants
OVF_VER = 'OOMMF OVF 2.0'

X, Y, Z = 'x', 'y', 'z'
XYZ = {X: 0, Y: 1, Z: 2}

MIN_DESC = [f'{x}min' for x in XYZ]
MAX_DESC = [f'{x}max' for x in XYZ]
BASE_DESC = [f'{x}base' for x in XYZ]
NODES_DESC = [f'{x}nodes' for x in XYZ]
STEP_DESC = [f'{x}stepsize' for x in XYZ]


# %% Functions
def read_header(f: Path) -> Tuple[Dict, int]:
    version_str = next(f).decode('utf-8').strip()[2:]
    if version_str != OVF_VER:
        print(f'WARNING! Version of .ovf files is not {OVF_VER}')
    seg_cnt = int(next(f).decode('utf-8').split(':')[1].strip())
    if seg_cnt > 1:
        print('WARNING! Using only first segment from .ovf files')

    header = {}
    for line in f:
        s = line.decode('utf-8').split(':')
        # Descriptor (without # )
        desc = s[0][2:]
        # Value
        val = s[1].strip()
        if desc == 'Desc':
            header.update({s[1].strip(): s[2].strip()})
        elif desc == 'Begin':
            if 'Data' in val:
                s = val.split(' ')
                if s[1] == 'Binary':
                    # Size of data type
                    dtype_sz = int(s[2])
                else:
                    raise ValueError(f'OVF: {s[1]} format is not supported!')
                break
        elif desc == 'End':
            pass
        else:
            header.update({desc: val})

    return header, dtype_sz


def read(file: Path) -> Tuple[str, Dict, Dict, Dict, np.ndarray]:
    with open(file, 'rb') as f:
        header, dtype_sz = read_header(f)

        nodes = [int(header[s]) for s in NODES_DESC]
        dim = int(header['valuedim'])
        bytes_qty = np.prod(nodes) * dtype_sz * dim

        val = f.read(dtype_sz)
        if dtype_sz == 4:
            dtype = np.float32
            check_value = struct.pack('<f', 1234567.0)
        elif dtype_sz == 8:
            dtype = np.float64
            check_value = struct.pack('<f', 123456789012345.0)
        else:
            raise ValueError('OVF: Wrong data type size')

        if val != check_value:
            raise ValueError('OVF: Wrong byte order')

        if header['meshtype'] != 'rectangular':
            raise ValueError('OVF: Only rectangular mesh is supported')

        bytes_data = f.read(bytes_qty)

        shape = [dim] + nodes
        data = np.frombuffer(bytes_data, dtype=dtype).reshape(shape, order='F')

        title = header['Title']
        # Grid
        grid_size = [float(header[s[1]]) - float(header[s[0]])
                     for s in zip(MIN_DESC, MAX_DESC)]
        grid = {'nodes': nodes,
                'size': grid_size,
                'step': [float(header[s]) for s in STEP_DESC],
                'base': [float(header[s]) for s in BASE_DESC],
                'unit': header['meshunit']
                }
        # Time
        time_tuple = header.get('Total simulation time', '0 s').split(' ')
        time = {'values': [float(time_tuple[0])],
                'unit': time_tuple[-1]}

        # Data
        quantity = {'dimension': dim,
                    'labels': header['valuelabels'].split(' '),
                    'units': header['valueunits'].split(' '),
                    }
        return title, time, grid, quantity, data


def convert(file_list: List[Path]) -> Dict:
    time_values = []
    data_arr_list = []
    data_dict = {}
    for i, file in enumerate(file_list):
        title, time, grid, quantity, data_arr = read(file)
        time_values += time['values']
        data_arr_list += [data_arr]
        if i == 0:
            data_dict = {'title': title,
                         'time': time,
                         'grid': grid,
                         'quantity': quantity}
        else:
            # Check header data
            if ((title != data_dict['title'])
                    or (grid != data_dict['grid'])
                    or (quantity != data_dict['quantity'])):
                fname = os.path.basename(file)
                print(f'WARNING! {fname} file header differ from first one.'
                      'Loading stopped!')
                break

    data = np.stack(data_arr_list, axis=0)
    # Move dimension-axis to the end
    data = np.moveaxis(data, 1, -1)

    data_dict['time']['values'] = time_values
    data_dict['data'] = data

    return data_dict


def ovf_to_mat(input_dir: Path,
               output_dir: Optional[Path] = None,
               filename_suffix: str = '',
               extra_data: Optional[Dict] = None) -> List[Path]:
    ovf_files = glob.glob(os.path.join(input_dir, '*.ovf'))
    mat_files = []

    q_dict = {}
    for f in ovf_files:
        # Get quantity name (filename except digits at the end)
        match = re.match(r'(.*?)\d+$', get_filename(f))
        if match:
            q_name = match.group(1)
            f_list = q_dict.get(q_name, [])
            q_dict[q_name] = f_list + [f]

    for q_name, f_list in q_dict.items():
        out_dict = convert(sorted(f_list))
        if extra_data is not None:
            out_dict['extra'] = extra_data

        if output_dir is None:
            output_dir = input_dir
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        mat_file = os.path.join(output_dir, f'{q_name}{filename_suffix}.mat')
        mat_files += [mat_file]

        savemat(mat_file, out_dict, do_compression=False)

    return mat_files
