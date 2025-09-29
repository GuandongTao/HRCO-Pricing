from pricing.contracts import HeatRateCallSpec
from pricing.market import Forwards, Vols, Corr, Df
from pricing.context import PricingContext
from pricing.kirk import KirkPricer

def make_ctx():
    contract = HeatRateCallSpec(h=7.5, K=5.0, settle_t=1.0)
    fwds = Forwards(F_power=90.0, F_gas=3.0)
    vols = Vols(vol_power=0.35, vol_gas=0.45)
    corr = Corr(rho_pg=0.25)
    df = Df(r=0.02)
    return PricingContext(contract, fwds, vols, corr, df)

def test_kirk_price_non_negative():
    ctx = make_ctx()
    px = KirkPricer().price(ctx)
    print(f"Kirk price = {px:.6f}")
    assert px >= 0.0

