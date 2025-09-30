"""Batch pricing engine that connects database to pricers."""

import sqlite3
import numpy as np
import pandas as pd
from typing import List, Optional, Protocol
from datetime import datetime

from inputs.context import PricingContext
from inputs.contracts import HeatRateCallSpec
from inputs.market import Forwards, Vols, Corr, Df


class Pricer(Protocol):
    """Protocol for any pricer implementation"""

    def price(self, ctx: PricingContext) -> np.ndarray:
        ...


class BatchPricer:
    """Batch pricing engine connecting database to pricing models"""

    def __init__(self, conn: sqlite3.Connection, pricer: Pricer):
        self.conn = conn
        self.pricer = pricer
        self.pricer_name = pricer.__class__.__name__

    # ========== Public Methods ==========

    def price_filtered_contracts(
            self,
            market_scenario_id: int,
            min_maturity: Optional[float] = None,
            max_maturity: Optional[float] = None,
            min_heat_rate: Optional[float] = None,
            max_heat_rate: Optional[float] = None,
            include_exercise_prob: bool = False
    ) -> pd.DataFrame:
        """
        Price contracts matching filters against a market scenario.
        Returns DataFrame with pricing results.
        """
        # Build contract filter query
        query = """
            SELECT 
                contract_id, h, K, settle_t, vom, quantity,
                gas_adder, start_cost, c_allowance, tp_cost, start_fuel
            FROM contracts
            WHERE 1=1
        """
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

        # Fetch contracts
        contracts_df = pd.read_sql_query(query, self.conn, params=params)

        if contracts_df.empty:
            print("No contracts match the filters")
            return pd.DataFrame()

        # Fetch market data
        market_query = """
            SELECT F_power, F_gas, F_ghg, vol_power, vol_gas, rho_pg, r
            FROM market_data
            WHERE market_id = ?
        """
        market_df = pd.read_sql_query(market_query, self.conn, params=[market_scenario_id])

        if market_df.empty:
            raise ValueError(f"Market scenario {market_scenario_id} not found")

        market = market_df.iloc[0]

        # Create vectorized pricing context
        ctx = self._build_vectorized_context(contracts_df, market)

        # Price all contracts at once
        prices = self.pricer.price(ctx)

        # Calculate total values
        total_values = prices * contracts_df['quantity'].values

        # Calculate exercise probabilities if requested
        exercise_probs = None
        if include_exercise_prob:
            exercise_probs = self._calculate_exercise_probs(ctx)

        # Store results in database
        result_ids = self._store_results(
            contracts_df['contract_id'].values,
            market_scenario_id,
            prices,
            total_values,
            exercise_probs
        )

        # Build results DataFrame
        results_df = contracts_df.copy()
        results_df['price_per_mwh'] = prices
        results_df['total_value'] = total_values
        if exercise_probs is not None:
            results_df['exercise_prob'] = exercise_probs
        results_df['result_id'] = result_ids

        return results_df

    def price_mapped_contracts(
            self,
            include_exercise_prob: bool = False
    ) -> pd.DataFrame:
        """
        Price contracts according to their mapped scenarios.
        Market curves are automatically looked up by contract maturity.

        For each contract:
        1. Get assigned scenario name from contract_scenario_map
        2. Look up market data using (scenario_name, contract.settle_t)
        3. Price the contract with matched market curve
        """
        # Get all contracts with their scenario assignments
        query = """
            SELECT 
                c.contract_id,
                c.contract_name,
                c.h, c.K, c.settle_t, c.vom, c.quantity,
                c.gas_adder, c.start_cost, c.c_allowance, c.tp_cost, c.start_fuel,
                csm.scenario_name
            FROM contracts c
            JOIN contract_scenario_map csm ON c.contract_id = csm.contract_id
            ORDER BY c.settle_t
        """

        contracts_df = pd.read_sql_query(query, self.conn)

        if contracts_df.empty:
            print("No contract-scenario mappings found")
            return pd.DataFrame()

        print(f"Found {len(contracts_df)} contracts to price")

        # Group by scenario to batch lookup market data
        all_results = []

        for scenario_name, group in contracts_df.groupby('scenario_name'):
            print(f"  Pricing {len(group)} contracts for scenario '{scenario_name}'")

            # Get maturities for this group
            maturities = group['settle_t'].unique()

            # Fetch market data for these maturities
            placeholders = ','.join('?' * len(maturities))
            market_query = f"""
                SELECT maturity, F_power, F_gas, F_ghg, vol_power, vol_gas, rho_pg, r
                FROM market_data
                WHERE scenario_name = ? AND maturity IN ({placeholders})
            """
            params = [scenario_name] + list(maturities)
            market_df = pd.read_sql_query(market_query, self.conn, params=params)

            # Create lookup dict: maturity -> market data
            market_lookup = {row['maturity']: row for _, row in market_df.iterrows()}

            # Match each contract to its market curve
            matched_contracts = []
            matched_markets = []

            for _, contract in group.iterrows():
                maturity = contract['settle_t']

                if maturity not in market_lookup:
                    print(f"    WARNING: No market data for {scenario_name} at maturity {maturity}")
                    continue

                matched_contracts.append(contract)
                matched_markets.append(market_lookup[maturity])

            if not matched_contracts:
                print(f"    No valid market curves found for scenario '{scenario_name}'")
                continue

            # Convert to DataFrames for vectorized pricing
            contracts_batch = pd.DataFrame(matched_contracts)
            markets_batch = pd.DataFrame(matched_markets)

            # Build vectorized context
            ctx = self._build_vectorized_context_from_curves(contracts_batch, markets_batch)

            # Price all at once
            prices = self.pricer.price(ctx)
            total_values = prices * contracts_batch['quantity'].values

            # Calculate exercise probs if requested
            exercise_probs = None
            if include_exercise_prob:
                exercise_probs = self._calculate_exercise_probs(ctx)

            # Store results
            result_ids = self._store_results_with_scenario(
                contracts_batch['contract_id'].values,
                scenario_name,
                contracts_batch['settle_t'].values,
                prices,
                total_values,
                exercise_probs
            )

            # Build result rows
            for i, contract in enumerate(matched_contracts):
                result = {
                    'result_id': result_ids[i],
                    'contract_id': contract['contract_id'],
                    'contract_name': contract['contract_name'],
                    'scenario_name': scenario_name,
                    'h': contract['h'],
                    'K': contract['K'],
                    'settle_t': contract['settle_t'],
                    'quantity': contract['quantity'],
                    'F_power': matched_markets[i]['F_power'],
                    'F_gas': matched_markets[i]['F_gas'],
                    'price_per_mwh': prices[i],
                    'total_value': total_values[i]
                }

                if exercise_probs is not None:
                    result['exercise_prob'] = exercise_probs[i]

                all_results.append(result)

        return pd.DataFrame(all_results)

    def price_all_scenarios(
            self,
            min_maturity: Optional[float] = None,
            max_heat_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """Price filtered contracts against all market scenarios"""
        # Get all market scenarios
        scenarios = pd.read_sql_query(
            "SELECT market_id, scenario_name FROM market_data",
            self.conn
        )

        all_results = []

        for _, scenario in scenarios.iterrows():
            print(f"Pricing scenario: {scenario['scenario_name']}")

            results = self.price_filtered_contracts(
                market_scenario_id=scenario['market_id'],
                min_maturity=min_maturity,
                max_heat_rate=max_heat_rate
            )

            if not results.empty:
                results['scenario_name'] = scenario['scenario_name']
                all_results.append(results)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    # ========== Private Helper Methods ==========

    def _build_vectorized_context(
            self,
            contracts_df: pd.DataFrame,
            market: pd.Series
    ) -> PricingContext:
        """Build vectorized pricing context from dataframes"""
        n = len(contracts_df)

        # Contract specs (vectorized)
        contract = HeatRateCallSpec(
            h=contracts_df['h'].values,
            K=contracts_df['K'].values,
            settle_t=contracts_df['settle_t'].values,
            vom=contracts_df['vom'].values,
            quantity=contracts_df['quantity'].values,
            gas_adder=contracts_df['gas_adder'].values,
            start_cost=contracts_df['start_cost'].values,
            c_allowance=contracts_df['c_allowance'].values,
            tp_cost=contracts_df['tp_cost'].values,
            start_fuel=contracts_df['start_fuel'].values
        )

        # Market data (broadcast to match contract length)
        forwards = Forwards(
            F_power=np.full(n, market['F_power']),
            F_gas=np.full(n, market['F_gas']),
            F_ghg=np.full(n, market['F_ghg'])
        )

        vols = Vols(
            vol_power=np.full(n, market['vol_power']),
            vol_gas=np.full(n, market['vol_gas'])
        )

        corr = Corr(rho_pg=np.full(n, market['rho_pg']))
        df = Df(r=np.full(n, market['r']))

        return PricingContext(contract, forwards, vols, corr, df)

    def _build_vectorized_context_from_curves(
            self,
            contracts_df: pd.DataFrame,
            markets_df: pd.DataFrame
    ) -> PricingContext:
        """
        Build vectorized pricing context where each contract has its own market curve.
        No broadcasting - lengths already match.
        """
        n = len(contracts_df)

        # Contract specs (already vectorized)
        contract = HeatRateCallSpec(
            h=contracts_df['h'].values,
            K=contracts_df['K'].values,
            settle_t=contracts_df['settle_t'].values,
            vom=contracts_df['vom'].values,
            quantity=contracts_df['quantity'].values,
            gas_adder=contracts_df['gas_adder'].values,
            start_cost=contracts_df['start_cost'].values,
            c_allowance=contracts_df['c_allowance'].values,
            tp_cost=contracts_df['tp_cost'].values,
            start_fuel=contracts_df['start_fuel'].values
        )

        # Market data (each row is a different curve - no broadcasting!)
        forwards = Forwards(
            F_power=markets_df['F_power'].values,
            F_gas=markets_df['F_gas'].values,
            F_ghg=markets_df['F_ghg'].values
        )

        vols = Vols(
            vol_power=markets_df['vol_power'].values,
            vol_gas=markets_df['vol_gas'].values
        )

        corr = Corr(rho_pg=markets_df['rho_pg'].values)
        df = Df(r=markets_df['r'].values)

        return PricingContext(contract, forwards, vols, corr, df)

    def _calculate_exercise_probs(self, ctx: PricingContext) -> np.ndarray:
        """Calculate exercise probabilities using finite difference"""
        from p_itm.p_itm import ExerciseProbability

        calc = ExerciseProbability(self.pricer, dK=0.0001)
        return calc.calculate(ctx, clamp=True)

    def _store_results(
            self,
            contract_ids: np.ndarray,
            market_id: int,
            prices: np.ndarray,
            total_values: np.ndarray,
            exercise_probs: Optional[np.ndarray]
    ) -> List[int]:
        """Store pricing results to database"""
        cursor = self.conn.cursor()
        result_ids = []

        for i, contract_id in enumerate(contract_ids):
            exercise_prob = exercise_probs[i] if exercise_probs is not None else None

            cursor.execute("""
                INSERT INTO pricing_results (
                    contract_id, market_id, pricer_type,
                    price_per_mwh, total_value, exercise_prob
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(contract_id),
                market_id,
                self.pricer_name,
                float(prices[i]),
                float(total_values[i]),
                float(exercise_prob) if exercise_prob is not None else None
            ))

            result_ids.append(cursor.lastrowid)

        self.conn.commit()
        return result_ids

    def _store_results_with_scenario(
            self,
            contract_ids: np.ndarray,
            scenario_name: str,
            maturities: np.ndarray,
            prices: np.ndarray,
            total_values: np.ndarray,
            exercise_probs: Optional[np.ndarray]
    ) -> List[int]:
        """Store pricing results, looking up market_id by scenario and maturity"""
        cursor = self.conn.cursor()
        result_ids = []

        for i, contract_id in enumerate(contract_ids):
            # Look up market_id for this scenario and maturity
            cursor.execute("""
                SELECT market_id FROM market_data
                WHERE scenario_name = ? AND maturity = ?
            """, (scenario_name, float(maturities[i])))

            row = cursor.fetchone()
            if not row:
                print(f"WARNING: Could not find market_id for {scenario_name}, maturity={maturities[i]}")
                continue

            market_id = row[0]
            exercise_prob = exercise_probs[i] if exercise_probs is not None else None

            cursor.execute("""
                INSERT INTO pricing_results (
                    contract_id, market_id, pricer_type,
                    price_per_mwh, total_value, exercise_prob
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(contract_id),
                market_id,
                self.pricer_name,
                float(prices[i]),
                float(total_values[i]),
                float(exercise_prob) if exercise_prob is not None else None
            ))

            result_ids.append(cursor.lastrowid)

        self.conn.commit()
        return result_ids