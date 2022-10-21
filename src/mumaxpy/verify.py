"""
Class with various functions for parameters verification
"""
# %% Imports
from astropy import units as u
from typing import Optional


# %% Constants
MUMAX_FLOAT_SIZE = 4 * u.B
SIZE_LIMIT = 4 * u.GB
VECTOR_COMP_QTY = 3


# %% Class
class Verify:
    def mat_files_size(iter_qty: int, cells_qty: int,
                       time: float, period: float,
                       vector_comp_qty: Optional[int] = None,
                       size_limit: float = SIZE_LIMIT) -> bool:
        ovf_qty = time / period
        ovf_size = cells_qty * MUMAX_FLOAT_SIZE
        single_mat_file_size = (ovf_qty * ovf_size).to(u.GB)
        all_mat_files_size = iter_qty * single_mat_file_size

        if vector_comp_qty is not None:
            all_mat_files_size *= vector_comp_qty
            opt_str = ''
        else:
            opt_str = f' (x{VECTOR_COMP_QTY} if saving all vector components)'

        print('MAT-files data size:', f'{all_mat_files_size:.1f}{opt_str}')

        if all_mat_files_size > size_limit:
            print('WARNING! Large output files')
            is_ok = False
        else:
            is_ok = True

        return is_ok
