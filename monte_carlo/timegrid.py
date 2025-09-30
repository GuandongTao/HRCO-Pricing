from dataclasses import dataclass
import numpy as np
from custom_types.types import FloatArray


@dataclass(frozen=True)
class TimeGrid:
    """
    Time discretization for simulation.
    Default: 365 steps per year (daily granularity).
    """
    T: float
    n_steps: int = 365

    @property
    def dt(self) -> float:
        """Time step size"""
        return self.T / self.n_steps

    def grid(self) -> FloatArray:
        """Return array of time points from 0 to T"""
        return np.linspace(0.0, self.T, self.n_steps + 1)