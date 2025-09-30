# Heat Rate Call Options Pricing

A Python library for pricing heat rate call options (spark spread options) using Kirk's approximation and Monte Carlo simulation.

## Project Structure

```
.
├── inputs/              # Market data and contract specifications
│   ├── contracts.py     # HeatRateCallSpec dataclass
│   ├── market.py        # Forwards, Vols, Corr, Df
│   └── context.py       # PricingContext wrapper
├── kirk/                # Kirk's approximation pricer
│   └── kirk.py          # KirkPricer implementation
├── monte_carlo/         # Monte Carlo simulation
│   ├── processes.py     # Lognormal forward dynamics (GBM)
│   ├── pathgen.py       # Correlated path generation
│   ├── payoffs.py       # Payoff calculations
│   ├── mc_pricer.py     # MonteCarloPricer main class
│   ├── rng.py           # Random number generation
│   ├── corr.py          # Cholesky decomposition for correlation
│   └── timegrid.py      # Time discretization
├── p_itm/               # Exercise probability calculator
│   └── p_itm.py         # Finite difference method
├── types/               # Shared type definitions
│   └── types.py         # ArrayLike, FloatArray helpers
└── tests/               # Test suite
    ├── test_kirk.py     # Kirk pricer tests
    ├── test_mc.py       # Monte Carlo tests
    └── test_p_itm.py    # Exercise probability tests
```

## Payoff Structure

Heat rate call options give the holder the right to convert gas to power:

```
Payoff = max(Power_Price - h × Gas_Price - Strike - Costs, 0) × quantity
```

## Pricing Methods

### 1. Kirk's Approximation (kirk/)

**Logic Flow:**
1. Transform heat rate option into equivalent spread option
2. Apply Kirk's approximation formula (Black-Scholes variant for spreads)
3. Handle degenerate cases (near-zero volatility)

**Key Features:**
- Fast analytical pricing (sub-millisecond)
- Vectorized for option strips
- Handles complex contract features (VOM, carbon, transport costs)

**Advantages:**
- Instant pricing for real-time applications
- Numerically stable
- Industry-standard approximation

---

### 2. Monte Carlo Simulation (monte_carlo/)

**Logic Flow:**
1. **Setup** → Initialize correlated 2-factor GBM (power, gas) under forward measure
2. **Simulate** → Generate terminal forward prices using:
   - Exact log-Euler stepping: `F(t+dt) = F(t) × exp(-0.5σ²dt + σ√dt×Z)`
   - Cholesky correlation: Independent normals → Correlated shocks
   - Antithetic pairs: Simulate both Z and -Z
3. **Payoff** → Calculate `max(P_T - h×G_T - K, 0)` for each path
4. **Control Variate** → Adjust using terminal spread (known expectation)
5. **Discount** → Apply discount factor and scale by notional

**Key Features:**
- Forward measure dynamics (zero drift for forwards)
- Antithetic variance reduction
- Control variate using terminal spread
- Batch processing for memory efficiency
- Reproducible seeding

**Advantages:**
- **Accuracy**: Converges to true price (within 1-2% with 200k paths)
- **Flexibility**: Handles any payoff structure, no approximation
- **Variance Reduction**: 
  - Antithetic: ~30-50% variance reduction
  - Control variate: Additional 20-40% improvement
- **Validation**: Can verify Kirk approximation quality

**Performance:**
- ~2-3 seconds for 200k paths (typical accuracy)
- Scales linearly with path count

---

### 3. Exercise Probability (p_itm/)

**Logic Flow:**
1. Bump strike up/down by small amount (dK)
2. Re-price using Kirk
3. Apply finite difference: `P(exercise) ≈ -dPrice/dK`

**Key Features:**
- Uses any pricer (Kirk, MC, etc.)
- Central difference for stability
- Optional clamping to [0, 1]

**Advantages:**
- Fast approximation of exercise probability
- Useful for risk management and hedging

---

## Key Design Choices

### Forward Measure
- Simulates forwards under their natural measure
- **Zero drift**: `dF/F = σ dW` (no μ term)
- Simplifies pricing and matches market conventions

### Exact GBM Steps
- Uses log-Euler (closed form) not naive Euler
- Eliminates discretization bias
- Formula: `F_new = F × exp(-0.5σ²Δt + σ√Δt Z)`

### Variance Reduction Stack
1. **Antithetic**: Generate Z and -Z pairs → Reduces variance
2. **Control Variate**: Use spread `C = P_T - h×G_T - K` with known `E[C] = F_power - h×F_gas - K`
   - Optimal coefficient: `b* = Cov(Payoff, C) / Var(C)`
   - Adjusted estimator: `Payoff - b*(C - E[C])`

### Vectorization
- All pricers support scalar or array inputs
- Price entire option strips in single call
- Consistent shapes validated at context creation

---

## When to Use Each Method

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **Kirk** | Real-time pricing, initial quotes, risk systems | <1ms | Good (~2-5% error) |
| **Monte Carlo** | Final pricing, exotic features, validation | ~2-3s | Excellent (<1% with 200k paths) |
| **Exercise Prob** | Greeks, risk metrics, delta hedging | <1ms | Approximation |

---

## Quick Example

```python
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df
from inputs.context import PricingContext
from kirk.kirk import KirkPricer
from monte_carlo.mc_pricer import MonteCarloPricer, MCParams

# Define contract and market
contract = HeatRateCallSpec(h=7.5, K=5.0, settle_t=1.0, quantity=15.0, vom=3.0)
forwards = Forwards(F_power=90.0, F_gas=3.0, F_ghg=0.05)
vols = Vols(vol_power=0.35, vol_gas=0.45)
ctx = PricingContext(contract, forwards, vols, Corr(rho_pg=0.25), Df(r=0.0))

# Kirk (fast)
kirk_px = KirkPricer().price(ctx)

# Monte Carlo (accurate)
mc_px = MonteCarloPricer(MCParams(n_paths=200_000, seed=42)).price(ctx)

print(f"Kirk: ${kirk_px:.2f}  |  MC: ${mc_px:.2f}")
```
