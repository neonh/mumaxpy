"""
Variables class
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


# %% Classes
@dataclass
class Variable:
    name: str
    values: np.ndarray

    def __str__(self):
        s = f'{self.name}={self.values}'
        return s

    def __len__(self):
        return len(self.values)


class Variables:
    def __init__(self):
        self._variables = []

    def __str__(self):
        str_list = [str(v) for v in self._variables]
        return "\n".join(str_list)

    def __len__(self):
        return len(self._variables)

    def get_full_length(self):
        length = 1
        for v in self._variables:
            length *= len(v)
        return length

    def at(self, i):
        sz = len(self._variables)
        if (i < sz) and (i >= -sz):
            v = self._variables[i]
        else:
            v = None
        return v

    def add(self, name, values):
        self._variables += [Variable(name, np.array(values))]

    def get(self):
        return self._variables

    def flush(self):
        self._variables = []

    def get_df(self):
        var_list = self._variables
        if len(var_list) > 0:
            # Create multiindex from variables list
            var_names = [v.name for v in var_list]
            var_values = [v.values for v in var_list]
            var_index = pd.MultiIndex.from_product(var_values,
                                                   names=var_names)
        else:
            var_index = pd.MultiIndex.from_tuples([(0,)])
        df = pd.DataFrame(index=var_index)
        return df
