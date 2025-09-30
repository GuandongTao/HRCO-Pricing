from typing import Union, Protocol
import numpy as np
from inputs.context import PricingContext
from dataclasses import replace

class Pricer(Protocol):
    def price(self, ctx: PricingContext) -> Union[float, np.ndarray]:
        ...

class ExerciseProbability:
    """
    Calculate exercise probability for heat rate call options using finite difference.

    Exercise probability ≈ -dPx/dK, approximated via central finite difference.
    """

    def __init__(self, pricer: Pricer, dK: float = 0.000005):
        self.pricer = pricer
        self.dK = dK

    def calculate(self, ctx: PricingContext, clamp: bool = False) -> Union[float, np.ndarray]:
        # Price at K + dK
        ctx_up = self._bump_strike(ctx, self.dK)
        price_up = self.pricer.price(ctx_up)

        # Price at K - dK
        ctx_down = self._bump_strike(ctx, -self.dK)
        price_down = self.pricer.price(ctx_down)

        # Central difference: -dV/dK
        exercise_prob = -(price_up - price_down) / (2 * self.dK)

        # Clamp to [0, 1] to handle numerical noise
        if clamp:
            exercise_prob = np.clip(exercise_prob, 0.0, 1.0)

        return exercise_prob

    def _bump_strike(self, ctx: PricingContext, bump: float) -> PricingContext:
        # Bump the strike
        K_bumped = ctx.contract.K + bump

        # Create new contract with bumped strike
        contract_bumped = replace(ctx.contract, K=K_bumped)

        # Create new context with bumped contract
        return replace(ctx, contract=contract_bumped)


# Convenience function
def calculate_exercise_probability(
        ctx: PricingContext,
        pricer: Pricer = None,
        dK: float = 0.000005
) -> Union[float, np.ndarray]:
    if pricer is None:
        from kirk.kirk import KirkPricer  # import future pricers here
        pricer = KirkPricer()

    calculator = ExerciseProbability(pricer, dK)
    return calculator.calculate(ctx)