from dataclasses import dataclass

@dataclass(frozen=True)
class Forwards:
    F_power: float
    F_gas: float

@dataclass(frozen=True)
class Vols:
    vol_power: float
    vol_gas: float

@dataclass(frozen=True)
class Corr:
    rho_pg: float

@dataclass(frozen=True)
class Df:
    r: float #df
    """Curve to be implemented?"""

