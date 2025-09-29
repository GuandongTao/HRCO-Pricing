from dataclasses import dataclass

@dataclass(frozen=True)
class HeatRateCallSpec:
    h: float               # heat rate multiplier
    K: float                # strike
    settle_t: float         # time to settlement in years
    quantity: float = 1.0   # MWh notional
    # more to be defined:
    # gas_adder
    # start_cost
    # max_run
    # min_stop
    # max_start
