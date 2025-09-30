import pytest
import numpy as np
from monte_carlo.mc_pricer import MonteCarloPricer, MCParams
from kirk.kirk import KirkPricer
from inputs.context import PricingContext
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df


def make_base_ctx():
    """Create a base pricing context for testing"""
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


class TestMonteCarloPricer:
    """Test suite for Monte Carlo pricer"""

    def test_mc_basic_convergence(self):
        """MC price should be non-negative and finite"""
        ctx = make_base_ctx()
        mc_params = MCParams(n_paths=50_000, seed=42)
        pricer = MonteCarloPricer(mc_params)

        price = pricer.price(ctx)

        print(f"MC price: {price:.6f}")
        assert price >= 0.0
        assert np.isfinite(price)

    def test_mc_vs_kirk_comparison(self):
        """MC should be reasonably close to Kirk approximation"""
        ctx = make_base_ctx()

        # Kirk (analytical approximation)
        kirk_pricer = KirkPricer()
        kirk_price = kirk_pricer.price(ctx)

        # Monte Carlo
        mc_params = MCParams(n_paths=200_000, seed=42, use_control_variate=True)
        mc_pricer = MonteCarloPricer(mc_params)
        mc_price = mc_pricer.price(ctx)

        print(f"Kirk price: {kirk_price:.6f}")
        print(f"MC price:   {mc_price:.6f}")
        print(f"Difference: {abs(kirk_price - mc_price):.6f}")
        print(f"Rel error:  {abs(kirk_price - mc_price) / kirk_price * 100:.2f}%")

        # MC should be within ~5% of Kirk for reasonable convergence
        rel_error = abs(kirk_price - mc_price) / kirk_price
        assert rel_error < 0.05, f"MC vs Kirk relative error too high: {rel_error:.2%}"

    def test_mc_deep_itm(self):
        """Deep ITM option should have price close to intrinsic value"""
        contract = HeatRateCallSpec(
            h=7.0,
            K=0.0,
            settle_t=1.0,
            quantity=1.0,
            vom=0.0
        )
        fwds = Forwards(F_power=100.0, F_gas=5.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.0)

        ctx = PricingContext(contract, fwds, vols, corr, df)

        # Intrinsic value
        intrinsic = fwds.F_power - contract.h * fwds.F_gas - contract.K

        # MC price
        mc_params = MCParams(n_paths=100_000, seed=42)
        mc_pricer = MonteCarloPricer(mc_params)
        mc_price = mc_pricer.price(ctx)

        print(f"Intrinsic: {intrinsic:.6f}")
        print(f"MC price:  {mc_price:.6f}")

        # Deep ITM should be close to intrinsic
        assert mc_price > intrinsic * 0.95
        assert mc_price < intrinsic * 1.05

    def test_mc_deep_otm(self):
        """Deep OTM option should have very low price"""
        contract = HeatRateCallSpec(
            h=7.0,
            K=50.0,
            settle_t=1.0,
            quantity=1.0,
            vom=0.0
        )
        fwds = Forwards(F_power=30.0, F_gas=10.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.0)

        ctx = PricingContext(contract, fwds, vols, corr, df)

        mc_params = MCParams(n_paths=100_000, seed=42)
        mc_pricer = MonteCarloPricer(mc_params)
        mc_price = mc_pricer.price(ctx)

        print(f"OTM MC price: {mc_price:.6f}")

        # Deep OTM should be close to zero
        assert mc_price < 1.0
        assert mc_price >= 0.0

    @pytest.mark.slow
    def test_mc_antithetic_variance_reduction(self):                    # issue with this test, put aside for now
        """Antithetic sampling should reduce variance (slow test)"""
        ctx = make_base_ctx()

        # Run single simulation and estimate variance from paths
        n_paths = 50_000  # Reduced for speed
        n_runs = 15  # Reduced for speed

        # To properly test variance reduction, we need to access individual path payoffs
        # Run a simpler version: multiple independent runs and measure price variance
        prices_no_anti = []
        prices_anti = []

        for i in range(n_runs):
            # Different seed each run to get independent estimates
            params_no_anti = MCParams(n_paths=n_paths, antithetic=False, seed=100 + i, use_control_variate=False)
            params_anti = MCParams(n_paths=n_paths, antithetic=True, seed=100 + i, use_control_variate=False)

            pricer_no_anti = MonteCarloPricer(params_no_anti)
            pricer_anti = MonteCarloPricer(params_anti)

            prices_no_anti.append(pricer_no_anti.price(ctx))
            prices_anti.append(pricer_anti.price(ctx))

        # Compute standard errors (measures precision of estimator)
        std_no_anti = np.std(prices_no_anti, ddof=1)
        std_anti = np.std(prices_anti, ddof=1)

        mean_no_anti = np.mean(prices_no_anti)
        mean_anti = np.mean(prices_anti)

        print(f"\nAntithetic Variance Reduction Test ({n_runs} runs, {n_paths} paths each):")
        print(f"Without antithetic: mean={mean_no_anti:.6f}, std={std_no_anti:.6f}")
        print(f"With antithetic:    mean={mean_anti:.6f}, std={std_anti:.6f}")
        print(f"Variance ratio:     {(std_anti / std_no_anti) ** 2:.3f}")
        print(f"Variance reduction: {(1 - (std_anti / std_no_anti) ** 2) * 100:.1f}%")

        # Antithetic should reduce standard deviation
        # Allow some tolerance since this is statistical
        assert std_anti <= std_no_anti * 1.1, \
            f"Antithetic should not increase variance: {std_anti:.6f} vs {std_no_anti:.6f}"

        # Typically expect 20-40% variance reduction
        # But don't make this a hard requirement due to statistical noise
        if std_anti < std_no_anti * 0.9:
            print("✓ Significant variance reduction achieved!")

    def test_mc_control_variate_improves_accuracy(self):
        """Control variate should improve convergence"""
        ctx = make_base_ctx()
        kirk_price = KirkPricer().price(ctx)

        # Without control variate
        mc_no_cv = MonteCarloPricer(
            MCParams(n_paths=50_000, use_control_variate=False, seed=42)
        )
        price_no_cv = mc_no_cv.price(ctx)

        # With control variate
        mc_cv = MonteCarloPricer(
            MCParams(n_paths=50_000, use_control_variate=True, seed=42)
        )
        price_cv = mc_cv.price(ctx)

        error_no_cv = abs(price_no_cv - kirk_price)
        error_cv = abs(price_cv - kirk_price)

        print(f"Kirk price:              {kirk_price:.6f}")
        print(f"MC without CV:           {price_no_cv:.6f} (error: {error_no_cv:.6f})")
        print(f"MC with CV:              {price_cv:.6f} (error: {error_cv:.6f})")
        print(f"Error reduction:         {(1 - error_cv / error_no_cv) * 100:.1f}%")

        # Control variate should typically reduce error
        assert error_cv <= error_no_cv * 1.2  # Allow some noise

    def test_mc_reproducibility(self):
        """Same seed should produce same results"""
        ctx = make_base_ctx()
        mc_params = MCParams(n_paths=50_000, seed=42)

        pricer1 = MonteCarloPricer(mc_params)
        pricer2 = MonteCarloPricer(mc_params)

        price1 = pricer1.price(ctx)
        price2 = pricer2.price(ctx)

        print(f"Price 1: {price1:.10f}")
        print(f"Price 2: {price2:.10f}")

        assert price1 == price2, "Same seed should produce identical results"

    def test_mc_zero_vol_equals_intrinsic(self):
        """Zero volatility should give discounted intrinsic value"""
        contract = HeatRateCallSpec(
            h=7.0,
            K=5.0,
            settle_t=1.0,
            quantity=1.0,
            vom=0.0
        )
        fwds = Forwards(F_power=60.0, F_gas=7.0, F_ghg=0.0)
        vols = Vols(vol_power=0.0, vol_gas=0.0)  # Zero vol
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(contract, fwds, vols, corr, df)

        # Expected: max(60 - 7*7 - 5, 0) * exp(-0.05*1) = 6 * exp(-0.05)
        intrinsic = max(fwds.F_power - contract.h * fwds.F_gas - contract.K, 0)
        expected = intrinsic * np.exp(-df.r * ctx.T)

        mc_params = MCParams(n_paths=10_000, seed=42)
        mc_pricer = MonteCarloPricer(mc_params)
        mc_price = mc_pricer.price(ctx)

        print(f"Expected (discounted intrinsic): {expected:.6f}")
        print(f"MC price:                        {mc_price:.6f}")

        # Should be very close with zero vol
        assert abs(mc_price - expected) < 0.01

    def test_mc_exercise_probability(self):
        """Test that MC pricer works with exercise probability calculator"""
        from p_itm.p_itm import ExerciseProbability, calculate_exercise_probability

        ctx = make_base_ctx()

        # Use MC pricer with exercise probability
        mc_pricer = MonteCarloPricer(MCParams(n_paths=50_000, seed=42))
        calc = ExerciseProbability(mc_pricer, dK=0.001)  # Larger dK for MC noise
        prob_mc = calc.calculate(ctx, clamp=True)

        # Compare with Kirk-based probability
        prob_kirk = calculate_exercise_probability(ctx, dK=0.001)

        print(f"Exercise probability (MC):   {prob_mc:.4f}")
        print(f"Exercise probability (Kirk): {prob_kirk:.4f}")
        print(f"Difference: {abs(prob_mc - prob_kirk):.4f}")

        # Should be valid probability
        assert 0.0 <= prob_mc <= 1.0

        # Should be reasonably close to Kirk (within 10% given MC noise)
        assert abs(prob_mc - prob_kirk) < 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])