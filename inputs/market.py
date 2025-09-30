from dataclasses import dataclass
import numpy as np
import warnings
from types.types import ArrayLike, as_array


@dataclass(frozen=True)
class Forwards:
    F_power: ArrayLike
    F_gas: ArrayLike
    F_ghg: ArrayLike = 0.0


@dataclass(frozen=True)
class Vols:
    vol_power: ArrayLike
    vol_gas: ArrayLike


@dataclass(frozen=True)
class Corr:
    rho_pg: ArrayLike

    def __post_init__(self):
        """Warn if correlation is outside [0, 1] range"""
        rho = as_array(self.rho_pg)

        if np.isscalar(self.rho_pg):
            if rho < 0.0 or rho > 1.0:
                warnings.warn(
                    f"Correlation {rho:.4f} is outside [0, 1] range. "
                    f"This may indicate unusual market conditions.",
                    UserWarning,
                    stacklevel=2
                )
        else:
            # Array case - warn if any values outside [0, 1]
            warning_mask = (rho < 0.0) | (rho > 1.0)
            if np.any(warning_mask):
                warning_indices = np.where(warning_mask)[0]
                warning_values = rho[warning_mask]
                warnings.warn(
                    f"Correlation values outside [0, 1] range at indices {warning_indices}: {warning_values}. "
                    f"This may indicate unusual market conditions.",
                    UserWarning,
                    stacklevel=2
                )


@dataclass(frozen=True)
class Df:
    r: ArrayLike