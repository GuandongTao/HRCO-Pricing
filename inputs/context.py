from dataclasses import dataclass
import numpy as np
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df
from types.types import ArrayLike, FloatArray, as_array


@dataclass(frozen=True)
class PricingContext:
    contract: HeatRateCallSpec
    forwards: Forwards
    vols: Vols
    corr: Corr
    df: Df

    def __post_init__(self):
        """Validate that all vector inputs have consistent lengths"""
        lengths = []

        # Collect lengths from contract
        for field_name in ['h', 'K', 'settle_t', 'vom', 'quantity', 'gas_adder',
                           'start_cost', 'c_allowance', 'tp_cost', 'start_fuel']:
            val = getattr(self.contract, field_name)
            arr = as_array(val)
            if arr.ndim > 0:
                lengths.append((f'contract.{field_name}', len(arr)))

        # Collect lengths from forwards
        for field_name in ['F_power', 'F_gas', 'F_ghg']:
            val = getattr(self.forwards, field_name)
            arr = as_array(val)
            if arr.ndim > 0:
                lengths.append((f'forwards.{field_name}', len(arr)))

        # Collect lengths from vols
        for field_name in ['vol_power', 'vol_gas']:
            val = getattr(self.vols, field_name)
            arr = as_array(val)
            if arr.ndim > 0:
                lengths.append((f'vols.{field_name}', len(arr)))

        # Collect lengths from corr
        val = as_array(self.corr.rho_pg)
        if val.ndim > 0:
            lengths.append(('corr.rho_pg', len(val)))

        # Collect lengths from df
        val = as_array(self.df.r)
        if val.ndim > 0:
            lengths.append(('df.r', len(val)))

        # Check if all lengths are the same
        if lengths:
            unique_lengths = set(length for _, length in lengths)
            if len(unique_lengths) > 1:
                length_details = ', '.join([f'{name}={length}' for name, length in lengths])
                raise ValueError(
                    f"Inconsistent vector lengths in PricingContext. "
                    f"All arrays must have the same length. Found: {length_details}"
                )

    @property
    def T(self) -> ArrayLike:
        return self.contract.settle_t