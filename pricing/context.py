from dataclasses import dataclass
from pricing.contracts import HeatRateCallSpec
from pricing.market import Forwards, Vols, Corr, Df
from typing import Union
import numpy as np


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
            if isinstance(val, np.ndarray):
                lengths.append((f'contract.{field_name}', len(val)))

        # Collect lengths from forwards
        for field_name in ['F_power', 'F_gas', 'F_ghg']:
            val = getattr(self.forwards, field_name)
            if isinstance(val, np.ndarray):
                lengths.append((f'forwards.{field_name}', len(val)))

        # Collect lengths from vols
        for field_name in ['vol_power', 'vol_gas']:
            val = getattr(self.vols, field_name)
            if isinstance(val, np.ndarray):
                lengths.append((f'vols.{field_name}', len(val)))

        # Collect lengths from corr
        if isinstance(self.corr.rho_pg, np.ndarray):
            lengths.append(('corr.rho_pg', len(self.corr.rho_pg)))

        # Collect lengths from df
        if isinstance(self.df.r, np.ndarray):
            lengths.append(('df.r', len(self.df.r)))

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
    def T(self) -> Union[float, np.ndarray]:
        return self.contract.settle_t
