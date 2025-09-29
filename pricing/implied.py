from __future__ import annotations
from dataclasses import replace
from pricing.context import PricingContext
from pricing.market import Vols
from pricing.kirk import KirkPricer
from pricing.solvers import Rootsettings, RootSolver

class KirkImplied:
    def __init__(self, pricer: KirkPricer | None = None, solver: RootSolver | None = None):
        self.pricer = pricer or KirkPricer()
        self.solver = solver or RootSolver(Rootsettings())

    def solve_vol_power(self, target_price: float, ctx: PricingContext, x0: float=0.3,
                        method: str="Newton") -> float:
        def f(sig1: float) -> float:
            new_vols = Vols(vol_power=max(sig1, 1e-12), vol_gas=ctx.vols.vol_gas)
            new_ctx = replace(ctx, vols=new_vols)
            return self.pricer.price(new_ctx) - target_price

        if method == "newton":
            return self.solver.newton(f, x0=x0)
        elif method == "brentq":
            return self.solver.brentq(f)
        else:
            raise ValueError("method must be 'newton' or 'brentq'")