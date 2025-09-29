from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass(frozen=True)
class Forwards:
    F_power: Union[float, np.ndarray]
    F_gas: Union[float, np.ndarray]
    F_ghg: Union[float, np.ndarray] = 0.0

@dataclass(frozen=True)
class Vols:
    vol_power: Union[float, np.ndarray]
    vol_gas: Union[float, np.ndarray]


@dataclass(frozen=True)
class Corr:
    rho_pg: Union[float, np.ndarray]

@dataclass(frozen=True)
class Df:
    r: Union[float, np.ndarray]


