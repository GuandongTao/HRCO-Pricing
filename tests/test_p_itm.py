import pytest
import numpy as np
from kirk.kirk import KirkPricer
from p_itm.p_itm import ExerciseProbability, calculate_exercise_probability
from inputs.context import PricingContext
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df


class TestExerciseProbability:
    """Test suite for exercise probability calculations"""

    def test_deep_itm_high_probability(self):
        """Deep ITM option should have exercise probability close to 1"""
        # Setup: Deep ITM (power = 100, gas = 5, h = 7, K = 0)
        # Payoff = 100 - 7*5 - 0 = 65 (very deep ITM)
        contract = HeatRateCallSpec(
            h=7.0,
            K=0.0,
            settle_t=1.0,
            vom=0.0
        )

        forwards = Forwards(F_power=100.0, F_gas=5.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        pricer = KirkPricer()
        calc = ExerciseProbability(pricer, dK=0.0001)
        prob = calc.calculate(ctx, clamp=False)
        print(f"ITM P: {prob}")

        assert prob > 0.95, f"Deep ITM should have prob > 0.95, got {prob:.4f}"
        assert prob <= 1.0, f"Probability cannot exceed 1.0, got {prob:.4f}"

    def test_deep_otm_low_probability(self):
        """Deep OTM option should have exercise probability close to 0"""
        # Setup: Deep OTM (power = 30, gas = 10, h = 7, K = 10)
        # Payoff = 30 - 7*10 - 10 = -50 (very deep OTM)
        contract = HeatRateCallSpec(
            h=7.0,
            K=10.0,
            settle_t=1.0,
            vom=0.0
        )

        forwards = Forwards(F_power=30.0, F_gas=10.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        pricer = KirkPricer()
        calc = ExerciseProbability(pricer, dK=0.0001)
        prob = calc.calculate(ctx, clamp=False)
        print(f"OTM P: {prob}")

        assert prob < 0.05, f"Deep OTM should have prob < 0.05, got {prob:.4f}"
        assert prob >= 0.0, f"Probability cannot be negative, got {prob:.4f}"

    def test_atm_around_50_percent(self):
        """ATM option should have exercise probability around 50%"""
        # Setup: ATM (power = 50, gas = 7, h = 7, K = 1)
        # Payoff = 50 - 7*7 - 1 = 0 (ATM)
        contract = HeatRateCallSpec(
            h=7.0,
            K=1.0,
            settle_t=1.0,
            vom=0.0
        )

        forwards = Forwards(F_power=50.0, F_gas=7.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        pricer = KirkPricer()
        calc = ExerciseProbability(pricer, dK=0.0001)
        prob = calc.calculate(ctx, clamp=True)
        print(f"ATM P: {prob}")

        assert 0.35 < prob < 0.65, f"ATM should have prob around 0.5, got {prob:.4f}"

    def test_vectorized_strip(self):
        """Test exercise probability for a strip of options"""
        # Setup: 3 options - OTM, ATM, ITM
        contract = HeatRateCallSpec(
            h=np.array([7.0, 7.0, 7.0]),
            K=np.array([10.0, 1.0, 0.0]),
            settle_t=np.array([1.0, 1.0, 1.0]),
            vom=np.array([0.0, 0.0, 0.0])
        )

        forwards = Forwards(
            F_power=np.array([50.0, 50.0, 50.0]),
            F_gas=np.array([7.0, 7.0, 7.0]),
            F_ghg=np.array([0.0, 0.0, 0.0])
        )

        vols = Vols(
            vol_power=np.array([0.2, 0.2, 0.2]),
            vol_gas=np.array([0.3, 0.3, 0.3])
        )

        corr = Corr(rho_pg=np.array([0.5, 0.5, 0.5]))
        df = Df(r=np.array([0.05, 0.05, 0.05]))

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        pricer = KirkPricer()
        calc = ExerciseProbability(pricer, dK=0.0001)
        probs = calc.calculate(ctx, clamp=False)
        print(f"Vector P: {probs}")

        assert len(probs) == 3
        assert probs[0] < probs[1] < probs[2], "Prob should increase: OTM < ATM < ITM"
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_convenience_function(self):
        """Test the convenience function with default pricer"""
        contract = HeatRateCallSpec(
            h=7.0,
            K=0.0,
            settle_t=1.0,
            vom=0.0
        )

        forwards = Forwards(F_power=100.0, F_gas=5.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        # Using convenience function (creates default pricer)
        prob = calculate_exercise_probability(ctx)
        print(f"Convenience ITM P: {prob}")

        assert 0.0 <= prob <= 1.0
        assert prob > 0.9, "Deep ITM should have high probability"

    def test_clamp_flag(self):
        """Test that clamp flag works correctly"""
        contract = HeatRateCallSpec(
            h=7.0,
            K=0.0,
            settle_t=1.0,
            vom=0.0
        )

        forwards = Forwards(F_power=100.0, F_gas=5.0, F_ghg=0.0)
        vols = Vols(vol_power=0.2, vol_gas=0.3)
        corr = Corr(rho_pg=0.5)
        df = Df(r=0.05)

        ctx = PricingContext(
            contract=contract,
            forwards=forwards,
            vols=vols,
            corr=corr,
            df=df
        )

        pricer = KirkPricer()
        calc = ExerciseProbability(pricer, dK=0.0001)

        # With clamp
        prob_clamped = calc.calculate(ctx, clamp=True)
        assert 0.0 <= prob_clamped <= 1.0

        # Without clamp (might be slightly outside due to numerical noise)
        prob_unclamped = calc.calculate(ctx, clamp=False)
        # Should be close but might differ slightly
        assert abs(prob_clamped - prob_unclamped) < 0.01

        print(f"Clamped P: {prob_clamped}")
        print(f"Unclamped P: {prob_unclamped}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])