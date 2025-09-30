from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GBMParams:
    vol_power:float
    vol_gas: float
