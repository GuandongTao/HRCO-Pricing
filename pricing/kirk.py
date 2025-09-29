from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from pricing.context import PricingContext
from typing import Union


@dataclass(frozen=True)
class KirkSettings:
    clip_rho: bool = False  # default no clip of rho - testing for extreme cases
    eps_sigma: float = 1e-12  # floor sig to be this val for computational ease


class KirkPricer:
    def __init__(self, settings: KirkSettings = KirkSettings()):
        self.s = settings

    def price(self, ctx: PricingContext) -> Union[float, np.ndarray]:
        """
        Price heat rate call option(s).
        Returns float for single option, np.ndarray for strip.
        """
        return self._price_impl(
            F1=ctx.forwards.F_power,
            F2=ctx.forwards.F_gas,
            F_ghg=ctx.forwards.F_ghg,
            s1=ctx.vols.vol_power,
            s2=ctx.vols.vol_gas,
            rho=ctx.corr.rho_pg,
            h=ctx.contract.h,
            K=ctx.contract.K,
            vom=ctx.contract.vom,
            c_allowance=ctx.contract.c_allowance,
            T=ctx.T,
            r=ctx.df.r,
            tp_cost=ctx.contract.tp_cost,
            gas_adder=ctx.contract.gas_adder,
            quantity=ctx.contract.quantity,
            start_fuel=ctx.contract.start_fuel
        )

    def _price_impl(
            self,
            F1: Union[float, np.ndarray],
            F2: Union[float, np.ndarray],
            F_ghg: Union[float, np.ndarray],
            s1: Union[float, np.ndarray],
            s2: Union[float, np.ndarray],
            rho: Union[float, np.ndarray],
            h: Union[float, np.ndarray],
            K: Union[float, np.ndarray],
            c_allowance: Union[float, np.ndarray],
            vom: Union[float, np.ndarray],
            T: Union[float, np.ndarray],
            r: Union[float, np.ndarray],
            tp_cost: Union[float, np.ndarray],
            gas_adder: Union[float, np.ndarray],
            quantity: Union[float, np.ndarray],
            start_fuel: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Core Kirk approximation formula. Adapted for np array for vectorization.
        """
        is_scalar = np.isscalar(F1)

        # clip corr if needed
        if self.s.clip_rho:
            rho = np.clip(rho, -1.0, 1.0)

        h_effective = h + start_fuel / quantity             # don't really come into play unless dealing with HRCO strip
        F_gas_effective = F2 + tp_cost + gas_adder
        K_effective = K + vom + F_ghg * c_allowance

        denom = h_effective * F_gas_effective + K_effective

        # corr clipping when in production; in testing it's turned off
        if np.any(denom <= 0.0):
            if np.isscalar(denom):
                raise ValueError(f"Invalid denominator: h*F2+K = {denom:.6f} (must be > 0).")
            else:
                bad_indices = np.where(denom <= 0.0)[0]
                raise ValueError(f"Invalid denominator at indices {bad_indices} (must be > 0).")

        # Kirk approximation
        w = h_effective * F_gas_effective / denom
        sig2 = s1 ** 2 - 2 * w * rho * s1 * s2 + w ** 2 * s2 ** 2
        sig = np.sqrt(sig2)

        # handle near zero vol explosion
        intrinsic_fwd = F1 - denom

        result = np.where(
            sig < self.s.eps_sigma,
            np.exp(-r * T) * intrinsic_fwd,
            self._kirk_formula(F1, denom, sig, sig2, T, r)
        )

        return float(result) if is_scalar else result

    def _kirk_formula(
            self,
            F1: Union[float, np.ndarray],
            denom: Union[float, np.ndarray],
            sig: Union[float, np.ndarray],
            sig2: Union[float, np.ndarray],
            T: Union[float, np.ndarray],
            r: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        sqrt_T = np.sqrt(T)
        d1 = (np.log(F1 / denom) + 0.5 * sig2 * T) / (sig * sqrt_T)
        d2 = (np.log(F1 / denom) - 0.5 * sig2 * T) / (sig * sqrt_T)
        return np.exp(-r * T) * (F1 * norm.cdf(d1) - denom * norm.cdf(d2))
