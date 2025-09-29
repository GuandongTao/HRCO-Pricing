from dataclasses import dataclass
from math import log, sqrt, exp
from statistics import NormalDist
from pricing.context import PricingContext

N = NormalDist().cdf

@dataclass(frozen=True)
class KirkSettings:
    clip_rho: bool = False # default no clip of rho - testing for extreme cases
    eps_sigma: float = 1e-12 # floor sig to be this val for computational ease

class KirkPricer:
    def __init__(self, settings: KirkSettings = KirkSettings()):
        self.s = settings

    def price(self, ctx: PricingContext) -> float:
        F1 = ctx.forwards.F_power
        F2 = ctx.forwards.F_gas
        s1 = ctx.vols.vol_power
        s2 = ctx.vols.vol_gas
        rho = ctx.corr.rho_pg
        if self.s.clip_rho:
            rho = max(-1.0, min(1.0, rho))
        h = ctx.contract.h
        K = ctx.contract.K
        T = ctx.T
        r = ctx.df.r

        denom = h * F2 + K
        if denom <= 0.0:
            raise ValueError(f"Invalid denominator: h*F2+K = {denom:.6f} (must be > 0).")

        w = h * F2 / denom
        sig2 = s1 * s1 - 2 * w * rho * s1 * s2 + w * w * s2 * s2
        sig = sqrt(sig2)

        intrinsic_fwd = F1 - denom
        if sig < self.s.eps_sigma:                  # if vol = 0.0(...)01
            return exp(-r * T) * intrinsic_fwd      # permits negative intrinsic

        d1 = (log(F1/denom) + 0.5 * sig2 * T) / (sig * sqrt(T))
        d2 = (log(F1/denom) - 0.5 * sig2 * T) / (sig * sqrt(T))
        return exp(-r * T) * (F1 * N(d1) - denom * N(d2))


