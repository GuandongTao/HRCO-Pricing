from dataclasses import dataclass
import numpy as np
from custom_types.types import FloatArray


@dataclass(frozen=True)
class GBMParams:
    vol_power: float
    vol_gas: float


class LognormalForwardProcess:
    """
    Forward dynamics under forward measure (risk-neutral): dF/F = sigma dW
    No drift term under forward measure.
    """

    def __init__(self, params: GBMParams):
        self.p = params

    def step(self, F: FloatArray, dW: FloatArray, dt: float) -> FloatArray:
        """
        Exact GBM step using log-Euler (closed form).

        Args:
            F: Current forward values, shape (..., 2) for [power, gas]
            dW: Correlated Brownian increments, shape (..., 2)
            dt: Time step size

        Returns:
            Next forward values with same shape as F
        """
        s1, s2 = self.p.vol_power, self.p.vol_gas
        sig = np.array([s1, s2], dtype=F.dtype)

        # Exact solution: F_{t+dt} = F_t * exp(-0.5*sigma^2*dt + sigma*sqrt(dt)*Z)
        drift = -0.5 * (sig ** 2) * dt
        diffusion = sig * np.sqrt(dt) * dW

        return F * np.exp(drift + diffusion)