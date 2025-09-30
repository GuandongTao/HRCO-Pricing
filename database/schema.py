"""Database schema for heat rate options pricing system using SQLite."""

import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class PricingDatabase:
    """Embedded SQLite database for heat rate options"""

    def __init__(self, db_path: str = "pricing.db"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        conn = self.connect() if not self.conn else self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if not self.conn:  # Only close if we opened it
                conn.close()

    def initialize_schema(self):
        """Create all tables if they don't exist"""
        with self.transaction() as conn:
            cursor = conn.cursor()

            # Contracts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    contract_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_name TEXT,
                    h REAL NOT NULL,
                    K REAL NOT NULL,
                    settle_t REAL NOT NULL,
                    vom REAL NOT NULL,
                    quantity REAL DEFAULT 1.0,
                    gas_adder REAL DEFAULT 0.0,
                    start_cost REAL DEFAULT 0.0,
                    c_allowance REAL DEFAULT 0.0,
                    tp_cost REAL DEFAULT 0.0,
                    start_fuel REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT valid_settle_t CHECK (settle_t > 0),
                    CONSTRAINT valid_h CHECK (h > 0),
                    CONSTRAINT valid_quantity CHECK (quantity > 0)
                )
            """)

            # Market data table - now includes maturity as key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    market_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_name TEXT NOT NULL,
                    maturity REAL NOT NULL,             -- maturity in years (key for curve lookup)
                    F_power REAL NOT NULL,
                    F_gas REAL NOT NULL,
                    F_ghg REAL DEFAULT 0.0,
                    vol_power REAL NOT NULL,
                    vol_gas REAL NOT NULL,
                    rho_pg REAL NOT NULL,
                    r REAL NOT NULL,
                    as_of_date DATE DEFAULT CURRENT_DATE,
                    CONSTRAINT valid_vols CHECK (vol_power >= 0 AND vol_gas >= 0),
                    CONSTRAINT valid_forwards CHECK (F_power > 0 AND F_gas > 0),
                    UNIQUE(scenario_name, maturity)
                )
            """)

            # Contract-to-scenario mapping table (simplified - just scenario name)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contract_scenario_map (
                    map_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id INTEGER NOT NULL,
                    scenario_name TEXT NOT NULL,
                    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id),
                    UNIQUE(contract_id)
                )
            """)

            # Pricing results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pricing_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_id INTEGER NOT NULL,
                    market_id INTEGER NOT NULL,
                    pricer_type TEXT NOT NULL,
                    price_per_mwh REAL NOT NULL,
                    total_value REAL NOT NULL,
                    exercise_prob REAL,
                    pricing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (contract_id) REFERENCES contracts(contract_id),
                    FOREIGN KEY (market_id) REFERENCES market_data(market_id)
                )
            """)

            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_contracts_maturity 
                ON contracts(settle_t)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_contracts_heat_rate 
                ON contracts(h)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_scenario_maturity
                ON market_data(scenario_name, maturity)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_contract 
                ON pricing_results(contract_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_timestamp 
                ON pricing_results(pricing_timestamp)
            """)

            # View for joined pricing results
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS pricing_summary AS
                SELECT 
                    pr.result_id,
                    pr.pricing_timestamp,
                    c.contract_name,
                    c.h as heat_rate,
                    c.K as strike,
                    c.settle_t as maturity_years,
                    c.quantity,
                    m.scenario_name,
                    m.maturity,
                    m.F_power,
                    m.F_gas,
                    pr.pricer_type,
                    pr.price_per_mwh,
                    pr.total_value,
                    pr.exercise_prob,
                    c.contract_id,
                    m.market_id
                FROM pricing_results pr
                JOIN contracts c ON pr.contract_id = c.contract_id
                JOIN market_data m ON pr.market_id = m.market_id
            """)

            print(f"✓ Database schema initialized at {self.db_path}")

    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP VIEW IF EXISTS pricing_summary")
            cursor.execute("DROP TABLE IF EXISTS pricing_results")
            cursor.execute("DROP TABLE IF EXISTS market_data")
            cursor.execute("DROP TABLE IF EXISTS contracts")
            print("✓ All tables dropped")