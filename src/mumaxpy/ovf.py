"""
Processing OVF files
"""
import glob
import numpy as np
import os
import struct
from scipy.io import savemat


# %% Constants
OVF_VER = 'OOMMF OVF 2.0'

X, Y, Z, ALL = 'x', 'y', 'z', ''
XYZ = {X: 0, Y: 1, Z: 2}

MIN_DESC = [f'{x}min' for x in XYZ]
MAX_DESC = [f'{x}max' for x in XYZ]
BASE_DESC = [f'{x}base' for x in XYZ]
NODES_DESC = [f'{x}nodes' for x in XYZ]
STEP_DESC = [f'{x}stepsize' for x in XYZ]


# %% Functions
def read_header(f):
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


def read(file):
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

        bytes_data = f.read(bytes_qty)

        shape = [dim] + nodes
        data = np.frombuffer(bytes_data, dtype=dtype).reshape(shape, order='F')

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
        q_unit = header['valueunits'].split(' ')[0]
        quantity = {'name': header['Title'],
                    'unit': q_unit,
                    'dimension': dim,
                    }
        return time, grid, quantity, data


def convert(file_list, component=ALL):
    time_values = []
    data_arr_list = []
    data_dict = {}
    for i, file in enumerate(file_list):
        time, grid, quantity, data_arr = read(file)
        time_values += time['values']
        data_arr_list += [data_arr]
        if i == 0:
            data_dict = {'time': time,
                         'grid': grid,
                         'quantity': quantity}
        else:
            # Check header data
            if ((grid != data_dict['grid'])
                    or (quantity != data_dict['quantity'])):
                fname = os.path.basename(file)
                print(f'WARNING! {fname} file header differ from first one.'
                      'Loading stopped!')
                break

    data = np.stack(data_arr_list, axis=0)

    if data_dict['quantity']['dimension'] > 1:
        if component != ALL:
            data_dict['quantity']['name'] += f'_{component}'
            data_dict['quantity']['dimension'] = 1
            # Get slice for this vector component
            data = data[:, XYZ[component], :, :, :]
        else:
            # Move dimension-axis to the end
            data = np.moveaxis(data, 1, -1)
    else:
        # Remove dimension-axis
        data = data[:, 0, :, :, :]

    data_dict['time']['values'] = time_values
    data_dict['data'] = data

    return data_dict


def ovf_to_mat(input_dir,
               quantity_name,
               component=ALL,
               output_dir=None,
               filename=None,
               extra_data=None):
    ovf_files = glob.glob(os.path.join(input_dir, f'{quantity_name}*.ovf'))

    if len(ovf_files) > 0:
        out_dict = convert(ovf_files, component=component)
        if extra_data is not None:
            out_dict['extra'] = extra_data

        if output_dir is None:
            output_dir = input_dir
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        if filename is None:
            filename = f'{quantity_name}{component}.mat'
        mat_file = os.path.join(output_dir, f'{filename}.mat')

        savemat(mat_file, out_dict, do_compression=False)
    else:
        mat_file = None

    return mat_file
