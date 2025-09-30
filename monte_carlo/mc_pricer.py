from dataclasses import dataclass
import numpy as np
from inputs.context import PricingContext
from monte_carlo.processes import GBMParams, LognormalForwardProcess
from monte_carlo.pathgen import TwoFactorPathGenerator, PathGenSettings
from monte_carlo.rng import make_rng
from monte_carlo.payoffs import heat_rate_call_terminal
from custom_types.types import FloatArray


@dataclass(frozen=True)
class MCParams:
    """Monte Carlo simulation parameters"""
    n_paths: int = 200_000
    n_steps: int = 365
    antithetic: bool = True
    seed: int | None = 42
    use_control_variate: bool = True


class MonteCarloPricer:
    """
    Monte Carlo pricer for heat rate call options.

    Features:
    - Lognormal forward dynamics under forward measure (zero drift)
    - Antithetic variance reduction
    - Control variate using terminal spread
    - Batch processing for memory efficiency
    """

    def __init__(self, params: MCParams = MCParams()):
        self.p = params

    def price(self, ctx: PricingContext) -> FloatArray:
        """
        Price a heat rate call option via Monte Carlo simulation.

        Args:
            ctx: Pricing context with contract specs and market data

        Returns:
            Option price (scalar or array depending on context)
        """
        # Extract initial forwards
        F0 = np.array([ctx.forwards.F_power, ctx.forwards.F_gas], dtype=np.float64)

        # Build process and path generator
        proc = LognormalForwardProcess(
            GBMParams(ctx.vols.vol_power, ctx.vols.vol_gas)
        )

        gen = TwoFactorPathGenerator(
            process=proc,
            rho=ctx.corr.rho_pg,
            settings=PathGenSettings(
                n_paths=self.p.n_paths,
                antithetic=self.p.antithetic
            )
        )

        # Random number generator
        rng = make_rng(self.p.seed)

        # Time step
        dt = ctx.T / self.p.n_steps

        # Simulate terminal values
        FT = gen.simulate_terminal(F0=F0, dt=dt, n_steps=self.p.n_steps, rng=rng)
        P_T, G_T = FT[:, 0], FT[:, 1]

        # Compute effective contract parameters (matching Kirk's logic)
        h_effective = ctx.contract.h + ctx.contract.start_fuel / ctx.contract.quantity
        G_T_effective = G_T + ctx.contract.tp_cost + ctx.contract.gas_adder
        K_effective = ctx.contract.K + ctx.contract.vom + ctx.forwards.F_ghg * ctx.contract.c_allowance

        # Compute payoff and spread
        payoff_T, spread_T = heat_rate_call_terminal(
            P_T, G_T_effective,
            h=h_effective,
            K=K_effective
        )

        # Discount factor
        disc = np.exp(-ctx.df.r * ctx.T)

        # Apply control variate if enabled
        if self.p.use_control_variate:
            est = self._apply_control_variate(
                payoff_T, spread_T, F0,
                h_effective, K_effective,
                ctx.contract.tp_cost, ctx.contract.gas_adder
            )
        else:
            est = payoff_T.mean()

        # Return price per unit (consistent with Kirk)
        return disc * est

    def _apply_control_variate(
            self,
            payoff: FloatArray,
            spread: FloatArray,
            F0: FloatArray,
            h_effective: float,
            K_effective: float,
            tp_cost: float,
            gas_adder: float
    ) -> float:
        """
        Apply control variate technique using terminal spread.

        Control: C = P_T - h_eff*G_T_eff - K_eff
        Known expectation: E[C] = F_power - h_eff*(F_gas + tp_cost + gas_adder) - K_eff
        """
        # Control variate and its expectation
        C = spread
        F_gas_effective = F0[1] + tp_cost + gas_adder
        C_bar = F0[0] - h_effective * F_gas_effective - K_effective

        # Estimate optimal coefficient: b* = Cov(X,C) / Var(C)
        cov_matrix = np.cov(payoff, C, ddof=1)
        var_C = cov_matrix[1, 1]

        if var_C > 0:
            cov_XC = cov_matrix[0, 1]
            b_star = cov_XC / var_C

            # Adjusted estimator: X - b*(C - E[C])
            adjusted = payoff - b_star * (C - C_bar)
            return adjusted.mean()
        else:
            # Degenerate case: fallback to standard estimator
            return payoff.mean()