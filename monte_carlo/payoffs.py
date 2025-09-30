import numpy as np
from custom_types.types import FloatArray


def heat_rate_call_terminal(
        power_T: FloatArray,
        gas_T: FloatArray,
        h: float,
        K: float
) -> tuple[FloatArray, FloatArray]:
    """
    Compute heat rate call payoff and raw spread at terminal time.

    Note: This computes the simplified payoff. In practice, h and K should be
    the effective values that include all contract adjustments:

    h_effective = h + start_fuel / quantity
    K_effective = K + VOM + F_ghg × c_allowance
    G_effective = G_T + tp_cost + gas_adder

    These adjustments are handled in the pricer before calling this function.

    Payoff = max(P_T - h_eff × G_eff - K_eff, 0)
    Spread = P_T - h_eff × G_eff - K_eff (for control variate)

    Args:
        power_T: Terminal power prices, shape (n_paths,)
        gas_T: Terminal gas prices (effective, with adjustments), shape (n_paths,)
        h: Heat rate multiplier (effective)
        K: Strike price (effective)

    Returns:
        Tuple of (payoff, spread) both with shape (n_paths,)
    """
    spread = power_T - h * gas_T - K
    payoff = np.maximum(spread, 0.0)

    return payoff, spread