from pricing.contracts import HeatRateCallSpec
from pricing.market import Forwards, Vols, Corr, Df
from pricing.context import PricingContext
from pricing.kirk import KirkPricer
import numpy as np

def make_ctx():
    contract = HeatRateCallSpec(
        h=7.5,
        K=5.0,
        settle_t=1.0,
        quantity=15.0,
        vom=3.0,
        c_allowance=25.0,
        tp_cost=0.1,
        gas_adder=0.1,
        start_fuel=0.0
    )
    fwds = Forwards(F_power=90.0, F_gas=3.0, F_ghg=0.05)
    vols = Vols(vol_power=0.35, vol_gas=0.45)
    corr = Corr(rho_pg=0.25)
    df = Df(r=0.0)
    return PricingContext(contract, fwds, vols, corr, df)

def test_kirk_price():
    ctx = make_ctx()
    px = KirkPricer().price(ctx)
    total_px = px * ctx.contract.quantity
    print(f"Kirk price = {px:.6f}")
    print(f"Kirk price = {total_px:.6f}")
    assert px >= 0.0


def make_ctx_vector():
    """Multiple options with vectorized market data"""
    contract_vector = HeatRateCallSpec(
        h=7.5,
        K=5.0,
        settle_t=np.array([1.0, 1.2]),
        vom=np.array([3.0, 3.2]),
        quantity=np.array([15, 17]),
        c_allowance=25.0,
        tp_cost=0.1,
        gas_adder=0.1,
        start_fuel=0.0
    )

    # Vectorized market data - 2 scenarios
    fwds = Forwards(
        F_power=np.array([90.0, 95.0]),
        F_gas=np.array([3.0, 3.5]),
        F_ghg=np.array([0.05, 0.05])
    )
    vols = Vols(
        vol_power=np.array([0.35, 0.40]),
        vol_gas=np.array([0.45, 0.50])
    )
    corr = Corr(rho_pg=np.array([0.25, 0.30]))
    df = Df(r=np.array([0.0, 0.0]))

    return PricingContext(contract_vector, fwds, vols, corr, df)

def test_kirk_price_vector():
    """Test vectorized pricing with vectorized market data"""
    ctx = make_ctx_vector()
    pricer = KirkPricer()

    # Price using vectorized method
    prices = pricer.price(ctx)
    total_values = prices * ctx.contract.quantity

    print(f"\nVector prices per MWh: {prices}")
    print(f"Total notional values: {total_values}")

    # Test all prices are non-negative
    assert np.all(prices >= 0.0)

    # Test we got 2 prices back
    assert len(prices) == 2



if __name__ == "__main__":
    test_kirk_price()
    test_kirk_price_vector()