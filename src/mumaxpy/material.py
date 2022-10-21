"""
Material class
"""
import yaml
from astropy import units as u
from dataclasses import dataclass


# %% Classes
@dataclass
class Material:
    alpha_Gilbert: u.Quantity = 0 * u.dimensionless_unscaled
    A_ex: u.Quantity = 0 * u.J / u.m
    fourPiMs: u.Quantity = 0 * u.erg / (u.G * u.cm**3)
    Ku_1: u.Quantity = 0 * u.erg / u.cm**3
    Ku_2: u.Quantity = 0 * u.erg / u.cm**3
    Kc_1: u.Quantity = 0 * u.erg / u.cm**3
    Kc_2: u.Quantity = 0 * u.erg / u.cm**3
    Kc_3: u.Quantity = 0 * u.erg / u.cm**3


# %% Functions
def load_material(name, file):
    with open(file, encoding="utf8") as f:
        m = yaml.safe_load(f)[name]

    alpha_Gilbert = m.get('alpha', 0) * u.dimensionless_unscaled
    A_ex = m.get('Aex', 0) * u.J / u.m
    fourPiMs = m.get('4piMs', 0) * u.erg / (u.G * u.cm**3)
    Ku_1 = m.get('Ku1', 0) * u.erg / u.cm**3
    Ku_2 = m.get('Ku2', 0) * u.erg / u.cm**3
    Kc_1 = m.get('Kc1', 0) * u.erg / u.cm**3
    Kc_2 = m.get('Kc2', 0) * u.erg / u.cm**3
    Kc_3 = m.get('Kc3', 0) * u.erg / u.cm**3

    material = Material(alpha_Gilbert, A_ex, fourPiMs,
                        Ku_1, Ku_2, Kc_1, Kc_2, Kc_3)
    return material
