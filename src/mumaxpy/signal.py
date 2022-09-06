"""
Signal class
"""
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


# %% Classes
@dataclass
class Label:
    name: str
    unit: str

    def __repr__(self):
        s = str(self.name)
        if self.unit != '':
            s += f', {self.unit}'
        return s


class Signal:
    def __init__(self, x_data, x_name='', x_unit='',
                 y_name='', y_unit='',
                 legend_name='', legend_unit=''):

        self.df = pd.DataFrame(index=x_data)
        self.x = Label(x_name, x_unit)
        self.y = Label(y_name, y_unit)
        self.legend = Label(legend_name, legend_unit)

    def add(self, data, label=None):
        if label is not None:
            self.df[label] = data
        else:
            i = len(self.df.columns)
            self.df.iloc[:, i] = data

    def get(self):
        return self.df

    def plot(self, file=None):
        fig, ax = plt.subplots()
        self.df.plot.line(ax=ax, xlabel=str(self.x), ylabel=str(self.y))
        ax.legend(title=f'{self.legend}:')
        fig.tight_layout()

        if file is not None:
            fig.savefig(file)

        return ax

    def save(self, file):
        df = self.df.copy()
        col_qty = len(df.columns)
        df.columns = pd.MultiIndex.from_tuples(
                        zip([self.y.name]*col_qty,
                            [self.y.unit]*col_qty,
                            df.columns),
                        names=(self.x.name, self.x.unit, f'{self.legend}:'))

        df.to_csv(file, sep='\t')
