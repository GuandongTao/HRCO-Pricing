from dataclasses import dataclass
import numpy as np
from monte_carlo.corr import cholesky2
from monte_carlo.processes import LognormalForwardProcess
from custom_types.types import FloatArray


@dataclass(frozen=True)
class PathGenSettings:
    """Configuration for path generation"""
    n_paths: int
    antithetic: bool = True
    batch_size: int = 100_000  # Memory control for large simulations


class TwoFactorPathGenerator:
    """
    Generate correlated paths for power and gas forwards using 2-factor GBM.
    Implements antithetic variance reduction and batch processing.
    """

    def __init__(
            self,
            process: LognormalForwardProcess,
            rho: float,
            settings: PathGenSettings
    ):
        self.process = process
        self.L = cholesky2(rho)
        self.s = settings

    def simulate_terminal(
            self,
            F0: FloatArray,
            dt: float,
            n_steps: int,
            rng: np.random.Generator
    ) -> FloatArray:
        """
        Simulate terminal forward values for power and gas.

        Args:
            F0: Initial forwards [F_power, F_gas], shape (2,)
            dt: Time step size
            n_steps: Number of time steps
            rng: Random number generator

        Returns:
            Terminal values (P_T, G_T) as array of shape (n_paths, 2)
        """
        n = self.s.n_paths
        out = np.empty((n, 2), dtype=np.float64)
        use_pairs = self.s.antithetic

        i = 0
        while i < n:
            # Process in batches for memory efficiency
            m = min(self.s.batch_size, n - i)

            # Generate independent normal draws: shape (m, n_steps, 2)
            Z = rng.standard_normal(size=(m, n_steps, 2))

            # Antithetic pairs: double paths with -Z
            if use_pairs:
                Z_full = np.concatenate([Z, -Z], axis=0)
                m_eff = Z_full.shape[0]
            else:
                Z_full = Z
                m_eff = m

            # Apply correlation via Cholesky: shape (m_eff, n_steps, 2)
            Zc = Z_full @ self.L.T

            # Initialize paths at F0
            F = np.broadcast_to(F0, (m_eff, 2)).copy()

            # Evolve forward step by step
            for t in range(n_steps):
                F = self.process.step(F, Zc[:, t, :], dt)

            # Store results (handle last batch partial fill)
            n_to_store = min(m_eff, n - i)
            out[i:i + n_to_store] = F[:n_to_store]
            i += n_to_store

        return out