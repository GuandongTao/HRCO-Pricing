from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass(frozen=True)
class HeatRateCallSpec:
    h: Union[float, np.ndarray]                    # heat rate multiplier
    K: Union[float, np.ndarray]                    # strike
    settle_t: Union[float, np.ndarray]             # time to settlement in years
    vom: Union[float, np.ndarray]                  # VOM cost
    quantity: Union[float, np.ndarray] = 1.0       # MWh notional
    gas_adder: Union[float, np.ndarray] = 0.0      # gas adder for slippage on cost
    start_cost: Union[float, np.ndarray] = 0.0     # start cost to run the power plant
    c_allowance: Union[float, np.ndarray] = 0.0    # carbon allowance
    tp_cost: Union[float, np.ndarray] = 0.0        # gas transportation cost
    start_fuel: Union[float, np.ndarray] = 0.0     # fuel for startup


    # more to be defined (limitation)
    # max_run
    # min_stop
    # max_start
