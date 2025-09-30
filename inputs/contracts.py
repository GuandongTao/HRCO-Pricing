from dataclasses import dataclass
from custom_types.types import ArrayLike


@dataclass(frozen=True)
class HeatRateCallSpec:
    h: ArrayLike                    # heat rate multiplier
    K: ArrayLike                    # strike
    settle_t: ArrayLike             # time to settlement in years
    vom: ArrayLike                  # VOM cost
    quantity: ArrayLike = 1.0       # MWh notional
    gas_adder: ArrayLike = 0.0      # gas adder for slippage on cost
    start_cost: ArrayLike = 0.0     # start cost to run the power plant
    c_allowance: ArrayLike = 0.0    # carbon allowance
    tp_cost: ArrayLike = 0.0        # gas transportation cost
    start_fuel: ArrayLike = 0.0     # fuel for startup

    # more to be defined (limitation)
    # max_run
    # min_stop
    # max_start