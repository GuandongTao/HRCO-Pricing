from dataclasses import dataclass
from pricing.contracts import HeatRateCallSpec
from pricing.market import Forwards, Vols, Corr, Df

@dataclass(frozen=True)
class PricingContext:
    contract: HeatRateCallSpec
    forwards: Forwards
    vols: Vols
    corr: Corr
    df: Df

    @property
    def T(self) -> float:
        return self.contract.settle_t
