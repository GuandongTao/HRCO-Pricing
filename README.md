# Heat Rate Call Options Pricing System

A comprehensive Python-based pricing system for heat rate call options using the Kirk approximation and embedded SQL database for batch processing.

## Features

- **Kirk Approximation Pricing**: Fast analytical pricing for spread options
- **Vectorized Computation**: Efficient batch pricing of multiple contracts
- **Exercise Probability Calculation**: Finite difference method for ITM probability
- **Embedded SQLite Database**: Store contracts, market curves, and pricing results
- **Maturity-Based Curve Lookup**: Automatic matching of contracts to term structure points
- **Batch Processing**: Price multiple contracts against market scenarios with SQL filters
- **Multiple Export Formats**: CSV, Excel, and summary reports

## Project Structure

```
HRCO-Pricing/
├── inputs/
│   ├── contracts.py      # Contract specifications
│   ├── market.py         # Market data (forwards, vols, correlations)
│   └── context.py        # Pricing context wrapper
├── kirk/
│   └── kirk.py           # Kirk approximation pricer
├── p_itm/
│   └── p_itm.py          # Exercise probability calculator
├── database/
│   ├── schema.py         # Database structure and initialization
│   ├── loader.py         # Data loading utilities
│   ├── batch_pricer.py   # Batch pricing engine
│   └── output.py         # Results export utilities
├── types/
│   └── types.py          # Shared type definitions
├── tests/
│   ├── test_kirk.py      # Kirk pricer tests
│   ├── test_p_itm.py     # Exercise probability tests
│   └── test_db_workflow.py  # End-to-end workflow tests
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/GuandongTao/HRCO-Pricing.git
cd HRCO-Pricing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy pandas openpyxl pytest
```

## Quick Start

### 1. Basic Kirk Pricing

```python
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df
from inputs.context import PricingContext
from kirk.kirk import KirkPricer

# Define contract
contract = HeatRateCallSpec(
    h=7.5,           # Heat rate
    K=5.0,           # Strike
    settle_t=1.0,    # Maturity (years)
    vom=3.0,         # Variable O&M
    quantity=15.0    # MWh notional
)

# Define market data
forwards = Forwards(F_power=90.0, F_gas=3.0, F_ghg=0.05)
vols = Vols(vol_power=0.35, vol_gas=0.45)
corr = Corr(rho_pg=0.25)
df = Df(r=0.02)

# Create context and price
ctx = PricingContext(contract, forwards, vols, corr, df)
pricer = KirkPricer()
price = pricer.price(ctx)

print(f"Option price: ${price:.2f} per MWh")
```

### 2. Batch Pricing with Database

```python
from database.schema import PricingDatabase
from database.loader import DataLoader
from database.batch_pricer import BatchPricer
from kirk.kirk import KirkPricer

# Initialize database
db = PricingDatabase("pricing.db")
db.connect()
db.initialize_schema()

# Load contracts
loader = DataLoader(db.conn)
contract_ids = loader.load_contracts_batch([
    {"contract_name": "Contract_1Y", "h": 7.5, "K": 5.0, "settle_t": 1.0, "vom": 3.0, "quantity": 15.0},
    {"contract_name": "Contract_2Y", "h": 8.0, "K": 5.5, "settle_t": 2.0, "vom": 3.2, "quantity": 20.0}
])

# Load market term structure
loader.load_market_data("Base_Case", maturity=1.0, F_power=85.0, F_gas=2.8, 
                        vol_power=0.30, vol_gas=0.40, rho_pg=0.20, r=0.02)
loader.load_market_data("Base_Case", maturity=2.0, F_power=88.0, F_gas=2.9, 
                        vol_power=0.32, vol_gas=0.42, rho_pg=0.22, r=0.02)

# Map contracts to scenarios
loader.map_contracts_batch([
    {"contract_id": contract_ids[0], "scenario_name": "Base_Case"},
    {"contract_id": contract_ids[1], "scenario_name": "Base_Case"}
])

# Price all contracts
pricer = KirkPricer()
batch_pricer = BatchPricer(db.conn, pricer)
results = batch_pricer.price_mapped_contracts(include_exercise_prob=True)

print(results)
db.close()
```

### 3. Export Results

```python
from database.output import ResultsExporter

exporter = ResultsExporter(db.conn)

# Export to CSV
exporter.to_csv("output/results.csv")

# Export to Excel with summary
exporter.to_excel("output/results.xlsx")

# Generate text summary
exporter.to_summary_report("output/summary.txt")

# Get portfolio summary
summary = exporter.get_portfolio_summary()
print(summary)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kirk.py -v

# Run database workflow test
pytest tests/test_db_workflow.py -v

# Or run directly
python tests/test_db_workflow.py
```

## Database Schema

The system uses SQLite with the following tables:

- **contracts**: Heat rate call option specifications
- **market_data**: Forward curves, volatilities by scenario and maturity
- **contract_scenario_map**: Links contracts to pricing scenarios
- **pricing_results**: Computed option prices and exercise probabilities
- **pricing_summary** (view): Denormalized results for easy querying

## Key Features Explained

### Maturity-Based Curve Lookup

Contracts are automatically priced using market curves matching their maturity:

```python
# Contract with 2-year maturity uses 2-year forward curve
# No broadcasting needed - true term structure handling
```

### SQL Filtering

Filter contracts for batch pricing:

```python
results = batch_pricer.price_filtered_contracts(
    market_scenario_id=1,
    min_maturity=2.0,      # Only contracts >= 2 years
    max_heat_rate=8.0      # Only efficient plants
)
```

### Exercise Probability

Calculate probability of exercise using finite difference:

```python
from p_itm.p_itm import calculate_exercise_probability

prob = calculate_exercise_probability(ctx)
print(f"Probability of exercise: {prob:.2%}")
```

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest tests/ -v`
- Code follows existing style
- New features include tests

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
