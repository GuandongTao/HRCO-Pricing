"""Data loading utilities for populating the pricing database."""

import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class DataLoader:
    """Load contracts and market data into database"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def load_contract(
            self,
            h: float,
            K: float,
            settle_t: float,
            vom: float,
            contract_name: Optional[str] = None,
            **kwargs
    ) -> int:
        """Insert a single contract, return contract_id"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO contracts (
                contract_name, h, K, settle_t, vom, 
                quantity, gas_adder, start_cost, 
                c_allowance, tp_cost, start_fuel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contract_name,
            h, K, settle_t, vom,
            kwargs.get('quantity', 1.0),
            kwargs.get('gas_adder', 0.0),
            kwargs.get('start_cost', 0.0),
            kwargs.get('c_allowance', 0.0),
            kwargs.get('tp_cost', 0.0),
            kwargs.get('start_fuel', 0.0)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def load_contracts_batch(self, contracts: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple contracts, return list of contract_ids"""
        ids = []
        for contract in contracts:
            contract_id = self.load_contract(**contract)
            ids.append(contract_id)
        return ids

    def load_market_data(
            self,
            scenario_name: str,
            maturity: float,
            F_power: float,
            F_gas: float,
            vol_power: float,
            vol_gas: float,
            rho_pg: float,
            r: float,
            F_ghg: float = 0.0
    ) -> int:
        """Insert market data for a specific scenario and maturity"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO market_data (
                scenario_name, maturity, F_power, F_gas, F_ghg,
                vol_power, vol_gas, rho_pg, r
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scenario_name, maturity,
            F_power, F_gas, F_ghg,
            vol_power, vol_gas, rho_pg, r
        ))

        self.conn.commit()
        return cursor.lastrowid

    def load_market_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple market scenarios"""
        ids = []
        for scenario in scenarios:
            market_id = self.load_market_data(**scenario)
            ids.append(market_id)
        return ids

    def map_contract_to_scenario(
            self,
            contract_id: int,
            scenario_name: str
    ) -> int:
        """Map a contract to a scenario name (maturity used for curve lookup)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO contract_scenario_map (contract_id, scenario_name)
            VALUES (?, ?)
        """, (contract_id, scenario_name))

        self.conn.commit()
        return cursor.lastrowid

    def map_contracts_batch(
            self,
            mappings: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Map multiple contracts to scenarios.

        Args:
            mappings: List of dicts with 'contract_id' and 'scenario_name' keys

        Example:
            mappings = [
                {"contract_id": 1, "scenario_name": "Base_Case"},
                {"contract_id": 2, "scenario_name": "High_Vol"}
            ]
        """
        ids = []
        for mapping in mappings:
            map_id = self.map_contract_to_scenario(
                mapping['contract_id'],
                mapping['scenario_name']
            )
            ids.append(map_id)
        return ids

    def load_contracts_from_csv(self, csv_path: str) -> List[int]:
        """Load contracts from CSV file"""
        df = pd.read_csv(csv_path)
        contracts = df.to_dict('records')
        return self.load_contracts_batch(contracts)

    def load_market_from_csv(self, csv_path: str) -> List[int]:
        """Load market scenarios from CSV file"""
        df = pd.read_csv(csv_path)
        scenarios = df.to_dict('records')
        return self.load_market_scenarios(scenarios)


class DataQuery:
    """Query utilities for retrieving data from database"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_contracts(
            self,
            min_maturity: Optional[float] = None,
            max_maturity: Optional[float] = None,
            min_heat_rate: Optional[float] = None,
            max_heat_rate: Optional[float] = None,
            min_strike: Optional[float] = None,
            max_strike: Optional[float] = None
    ) -> pd.DataFrame:
        """Query contracts with filters"""
        query = "SELECT * FROM contracts WHERE 1=1"
        params = []

        if min_maturity is not None:
            query += " AND settle_t >= ?"
            params.append(min_maturity)

        if max_maturity is not None:
            query += " AND settle_t <= ?"
            params.append(max_maturity)

        if min_heat_rate is not None:
            query += " AND h >= ?"
            params.append(min_heat_rate)

        if max_heat_rate is not None:
            query += " AND h <= ?"
            params.append(max_heat_rate)

        if min_strike is not None:
            query += " AND K >= ?"
            params.append(min_strike)

        if max_strike is not None:
            query += " AND K <= ?"
            params.append(max_strike)

        query += " ORDER BY settle_t, h"

        return pd.read_sql_query(query, self.conn, params=params)

    def get_market_scenarios(
            self,
            scenario_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Query market scenarios"""
        if scenario_names:
            placeholders = ','.join('?' * len(scenario_names))
            query = f"""
                SELECT * FROM market_data 
                WHERE scenario_name IN ({placeholders})
            """
            return pd.read_sql_query(query, self.conn, params=scenario_names)
        else:
            return pd.read_sql_query("SELECT * FROM market_data", self.conn)

    def get_pricing_results(
            self,
            contract_ids: Optional[List[int]] = None,
            min_timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """Query pricing results"""
        query = "SELECT * FROM pricing_summary WHERE 1=1"
        params = []

        if contract_ids:
            placeholders = ','.join('?' * len(contract_ids))
            query += f" AND contract_id IN ({placeholders})"
            params.extend(contract_ids)

        if min_timestamp:
            query += " AND pricing_timestamp >= ?"
            params.append(min_timestamp)

        query += " ORDER BY pricing_timestamp DESC, contract_name"

        return pd.read_sql_query(query, self.conn, params=params)