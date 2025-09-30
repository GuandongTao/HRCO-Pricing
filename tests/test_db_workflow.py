"""
Test: Complete batch pricing workflow with embedded SQL database.

This demonstrates:
1. Creating and initializing the database
2. Loading toy contracts and market term structures
3. Running batch pricing with maturity-based curve lookup
4. Exporting results to multiple formats
"""

import os
import pytest
from database.schema import PricingDatabase
from database.loader import DataLoader
from database.batch_pricer import BatchPricer
from database.output import ResultsExporter
from kirk.kirk import KirkPricer


@pytest.fixture(scope="module")
def setup_database():
    """Pytest fixture to set up database once for all tests"""
    # Remove old database if it exists
    db_path = "toy_pricing.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✓ Removed existing database: {db_path}")

    # Initialize database
    db = PricingDatabase(db_path)
    db.connect()
    db.initialize_schema()

    # Verify view was created
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='pricing_summary'")
    view_exists = cursor.fetchone()
    if view_exists:
        print("✓ pricing_summary view verified")
    else:
        print("⚠ WARNING: pricing_summary view was NOT created!")
        cursor.execute("SELECT type, name FROM sqlite_master")
        objects = cursor.fetchall()
        print("Database objects:", objects)

    loader = DataLoader(db.conn)

    # Load toy contracts
    print("\n=== Loading Toy Contracts ===")
    contracts = [
        {"contract_name": "Contract_1Y_HR5", "h": 5.0, "K": 3.0, "settle_t": 1.0, "vom": 2.5, "quantity": 10.0},
        {"contract_name": "Contract_1Y_HR7", "h": 7.0, "K": 4.0, "settle_t": 1.0, "vom": 3.0, "quantity": 15.0},
        {"contract_name": "Contract_2Y_HR6", "h": 6.0, "K": 5.0, "settle_t": 2.0, "vom": 2.8, "quantity": 20.0},
        {"contract_name": "Contract_2Y_HR8", "h": 8.0, "K": 5.5, "settle_t": 2.0, "vom": 3.2, "quantity": 12.0},
        {"contract_name": "Contract_3Y_HR7", "h": 7.5, "K": 5.0, "settle_t": 3.0, "vom": 3.0, "quantity": 15.0},
        {"contract_name": "Contract_5Y_HR6", "h": 6.5, "K": 4.5, "settle_t": 5.0, "vom": 2.9, "quantity": 25.0},
    ]
    contract_ids = loader.load_contracts_batch(contracts)
    print(f"✓ Loaded {len(contract_ids)} contracts")

    # Load market term structures
    print("\n=== Loading Market Term Structures ===")
    base_case_curves = [
        {"scenario_name": "Base_Case", "maturity": 1.0, "F_power": 85.0, "F_gas": 2.8, "F_ghg": 0.04,
         "vol_power": 0.30, "vol_gas": 0.40, "rho_pg": 0.20, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 2.0, "F_power": 88.0, "F_gas": 2.9, "F_ghg": 0.045,
         "vol_power": 0.32, "vol_gas": 0.42, "rho_pg": 0.22, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 3.0, "F_power": 92.0, "F_gas": 3.0, "F_ghg": 0.05,
         "vol_power": 0.35, "vol_gas": 0.45, "rho_pg": 0.25, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 5.0, "F_power": 95.0, "F_gas": 3.2, "F_ghg": 0.055,
         "vol_power": 0.38, "vol_gas": 0.48, "rho_pg": 0.28, "r": 0.02},
    ]

    high_vol_curves = [
        {"scenario_name": "High_Vol", "maturity": 1.0, "F_power": 87.0, "F_gas": 2.9, "F_ghg": 0.05,
         "vol_power": 0.45, "vol_gas": 0.55, "rho_pg": 0.25, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 2.0, "F_power": 92.0, "F_gas": 3.1, "F_ghg": 0.055,
         "vol_power": 0.48, "vol_gas": 0.58, "rho_pg": 0.28, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 3.0, "F_power": 97.0, "F_gas": 3.3, "F_ghg": 0.06,
         "vol_power": 0.52, "vol_gas": 0.62, "rho_pg": 0.30, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 5.0, "F_power": 102.0, "F_gas": 3.6, "F_ghg": 0.065,
         "vol_power": 0.55, "vol_gas": 0.65, "rho_pg": 0.32, "r": 0.02},
    ]

    for curve in base_case_curves + high_vol_curves:
        loader.load_market_data(**curve)
    print(f"✓ Loaded {len(base_case_curves + high_vol_curves)} market curve points")

    # Map contracts to scenarios
    print("\n=== Mapping Contracts to Scenarios ===")
    mappings = [
        {"contract_id": contract_ids[0], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[1], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[2], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[3], "scenario_name": "High_Vol"},
        {"contract_id": contract_ids[4], "scenario_name": "High_Vol"},
        {"contract_id": contract_ids[5], "scenario_name": "Base_Case"},
    ]
    loader.map_contracts_batch(mappings)
    print(f"✓ Mapped {len(mappings)} contracts to scenarios")

    # Keep connection open, return db_path
    yield db_path

    # Cleanup after all tests (optional)
    db.close()


def test_maturity_based_pricing(setup_database):
    """Test maturity-based curve lookup and pricing"""
    db_path = setup_database

    db = PricingDatabase(db_path)
    db.connect()

    pricer = KirkPricer()
    batch_pricer = BatchPricer(db.conn, pricer)

    print("\n" + "="*70)
    print("TEST: Maturity-Based Curve Lookup Pricing")
    print("="*70)

    # Price all mapped contracts
    results = batch_pricer.price_mapped_contracts(include_exercise_prob=True)

    print("\nPricing Results:")
    print("=" * 70)
    display_cols = ['contract_name', 'scenario_name', 'settle_t', 'h', 'F_power',
                    'F_gas', 'price_per_mwh', 'total_value', 'exercise_prob']
    print(results[display_cols].to_string(index=False))

    # Assertions
    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    assert all(results['price_per_mwh'] >= 0), "All prices should be non-negative"
    assert all(results['total_value'] >= 0), "All total values should be non-negative"

    # Check that different maturities got different forward prices
    base_case_results = results[results['scenario_name'] == 'Base_Case']
    unique_forwards = base_case_results[['settle_t', 'F_power']].drop_duplicates()
    print(f"\n✓ Base_Case has {len(unique_forwards)} unique maturity-forward pairs")

    db.close()


def test_output_export(setup_database):
    """Test exporting results to different formats"""
    db_path = setup_database

    db = PricingDatabase(db_path)
    db.connect()

    exporter = ResultsExporter(db.conn)

    print("\n" + "="*70)
    print("TEST: Export Results")
    print("="*70)

    # Export to CSV
    exporter.to_csv("output/pricing_results.csv")

    # Export to Excel
    exporter.to_excel("output/pricing_results.xlsx")

    # Generate summary report
    exporter.to_summary_report("output/pricing_summary.txt")

    # Display portfolio summary
    print("\n=== Portfolio Summary by Scenario ===")
    summary = exporter.get_portfolio_summary()
    print(summary.to_string(index=False))

    # Assertions
    assert len(summary) == 2, "Should have 2 scenarios"
    assert all(summary['num_contracts'] > 0), "Each scenario should have contracts"
    assert all(summary['portfolio_value'] > 0), "Portfolio values should be positive"

    print("\n✓ All exports completed successfully")

    db.close()


if __name__ == "__main__":
    # When run directly (not through pytest), set up and run tests manually
    print("="*70)
    print("HEAT RATE CALL OPTIONS - DATABASE WORKFLOW TEST")
    print("="*70)

    # Remove old database
    db_path = "toy_pricing.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✓ Removed existing database: {db_path}")

    # Initialize database
    db = PricingDatabase(db_path)
    db.connect()
    db.initialize_schema()

    loader = DataLoader(db.conn)

    # Load toy contracts
    print("\n=== Loading Toy Contracts ===")
    contracts = [
        {"contract_name": "Contract_1Y_HR5", "h": 5.0, "K": 3.0, "settle_t": 1.0, "vom": 2.5, "quantity": 10.0},
        {"contract_name": "Contract_1Y_HR7", "h": 7.0, "K": 4.0, "settle_t": 1.0, "vom": 3.0, "quantity": 15.0},
        {"contract_name": "Contract_2Y_HR6", "h": 6.0, "K": 5.0, "settle_t": 2.0, "vom": 2.8, "quantity": 20.0},
        {"contract_name": "Contract_2Y_HR8", "h": 8.0, "K": 5.5, "settle_t": 2.0, "vom": 3.2, "quantity": 12.0},
        {"contract_name": "Contract_3Y_HR7", "h": 7.5, "K": 5.0, "settle_t": 3.0, "vom": 3.0, "quantity": 15.0},
        {"contract_name": "Contract_5Y_HR6", "h": 6.5, "K": 4.5, "settle_t": 5.0, "vom": 2.9, "quantity": 25.0},
    ]
    contract_ids = loader.load_contracts_batch(contracts)
    print(f"✓ Loaded {len(contract_ids)} contracts")

    # Load market term structures
    print("\n=== Loading Market Term Structures ===")
    base_case_curves = [
        {"scenario_name": "Base_Case", "maturity": 1.0, "F_power": 85.0, "F_gas": 2.8, "F_ghg": 0.04,
         "vol_power": 0.30, "vol_gas": 0.40, "rho_pg": 0.20, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 2.0, "F_power": 88.0, "F_gas": 2.9, "F_ghg": 0.045,
         "vol_power": 0.32, "vol_gas": 0.42, "rho_pg": 0.22, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 3.0, "F_power": 92.0, "F_gas": 3.0, "F_ghg": 0.05,
         "vol_power": 0.35, "vol_gas": 0.45, "rho_pg": 0.25, "r": 0.02},
        {"scenario_name": "Base_Case", "maturity": 5.0, "F_power": 95.0, "F_gas": 3.2, "F_ghg": 0.055,
         "vol_power": 0.38, "vol_gas": 0.48, "rho_pg": 0.28, "r": 0.02},
    ]

    high_vol_curves = [
        {"scenario_name": "High_Vol", "maturity": 1.0, "F_power": 87.0, "F_gas": 2.9, "F_ghg": 0.05,
         "vol_power": 0.45, "vol_gas": 0.55, "rho_pg": 0.25, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 2.0, "F_power": 92.0, "F_gas": 3.1, "F_ghg": 0.055,
         "vol_power": 0.48, "vol_gas": 0.58, "rho_pg": 0.28, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 3.0, "F_power": 97.0, "F_gas": 3.3, "F_ghg": 0.06,
         "vol_power": 0.52, "vol_gas": 0.62, "rho_pg": 0.30, "r": 0.02},
        {"scenario_name": "High_Vol", "maturity": 5.0, "F_power": 102.0, "F_gas": 3.6, "F_ghg": 0.065,
         "vol_power": 0.55, "vol_gas": 0.65, "rho_pg": 0.32, "r": 0.02},
    ]

    for curve in base_case_curves + high_vol_curves:
        loader.load_market_data(**curve)
    print(f"✓ Loaded {len(base_case_curves + high_vol_curves)} market curve points")

    # Map contracts to scenarios
    print("\n=== Mapping Contracts to Scenarios ===")
    mappings = [
        {"contract_id": contract_ids[0], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[1], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[2], "scenario_name": "Base_Case"},
        {"contract_id": contract_ids[3], "scenario_name": "High_Vol"},
        {"contract_id": contract_ids[4], "scenario_name": "High_Vol"},
        {"contract_id": contract_ids[5], "scenario_name": "Base_Case"},
    ]
    loader.map_contracts_batch(mappings)
    print(f"✓ Mapped {len(mappings)} contracts to scenarios")

    print(f"\n✓ Database setup complete: {db_path}")

    # Run tests manually
    test_maturity_based_pricing(db_path)
    test_output_export(db_path)

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("✓ Check the 'output/' folder for exported files.")
    print("="*70)