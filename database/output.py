"""Output utilities for exporting pricing results."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime


class ResultsExporter:
    """Export pricing results to various formats"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def to_csv(
            self,
            output_path: str,
            contract_ids: Optional[List[int]] = None,
            scenario_names: Optional[List[str]] = None,
            min_timestamp: Optional[str] = None
    ):
        """Export pricing results to CSV"""
        df = self._get_results(contract_ids, scenario_names, min_timestamp)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Exported {len(df)} results to {output_path}")

    def to_excel(
            self,
            output_path: str,
            contract_ids: Optional[List[int]] = None,
            scenario_names: Optional[List[str]] = None,
            min_timestamp: Optional[str] = None
    ):
        """Export pricing results to Excel with multiple sheets"""
        # Main results
        results_df = self._get_results(contract_ids, scenario_names, min_timestamp)

        # Summary statistics by scenario
        summary_df = results_df.groupby('scenario_name').agg({
            'price_per_mwh': ['mean', 'min', 'max', 'std'],
            'total_value': 'sum',
            'contract_name': 'count'
        }).round(4)
        summary_df.columns = ['avg_price', 'min_price', 'max_price', 'std_price',
                              'total_portfolio_value', 'num_contracts']
        summary_df = summary_df.reset_index()

        # Write to Excel
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Pricing Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Format worksheets
            self._format_excel_sheet(writer, 'Pricing Results')
            self._format_excel_sheet(writer, 'Summary')

        print(f"✓ Exported {len(results_df)} results to {output_path}")

    def to_summary_report(
            self,
            output_path: str,
            scenario_names: Optional[List[str]] = None
    ):
        """Generate a summary report text file"""
        results_df = self._get_results(scenario_names=scenario_names)

        if results_df.empty:
            print("No results to export")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HEAT RATE CALL OPTIONS - PRICING SUMMARY REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Contracts Priced: {len(results_df)}\n")
            f.write("\n")

            # Summary by scenario
            for scenario in results_df['scenario_name'].unique():
                scenario_df = results_df[results_df['scenario_name'] == scenario]

                f.write("-" * 70 + "\n")
                f.write(f"SCENARIO: {scenario}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of contracts: {len(scenario_df)}\n")
                f.write(f"Total portfolio value: ${scenario_df['total_value'].sum():,.2f}\n")
                f.write(f"Average price per MWh: ${scenario_df['price_per_mwh'].mean():.4f}\n")
                f.write(f"Min price per MWh: ${scenario_df['price_per_mwh'].min():.4f}\n")
                f.write(f"Max price per MWh: ${scenario_df['price_per_mwh'].max():.4f}\n")
                f.write("\n")

                # Top 5 most valuable contracts
                f.write("Top 5 Most Valuable Contracts:\n")
                top5 = scenario_df.nlargest(5, 'total_value')
                for idx, row in top5.iterrows():
                    f.write(f"  {row['contract_name']}: "
                            f"${row['total_value']:,.2f} "
                            f"(HR={row['heat_rate']:.1f}, T={row['maturity_years']:.1f}y)\n")
                f.write("\n")

        print(f"✓ Summary report written to {output_path}")

    def get_portfolio_summary(
            self,
            scenario_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get portfolio-level summary statistics"""
        query = """
            SELECT 
                scenario_name,
                COUNT(*) as num_contracts,
                SUM(total_value) as portfolio_value,
                AVG(price_per_mwh) as avg_price,
                MIN(price_per_mwh) as min_price,
                MAX(price_per_mwh) as max_price,
                AVG(exercise_prob) as avg_exercise_prob,
                SUM(quantity) as total_notional_mwh
            FROM pricing_summary
        """

        if scenario_name:
            query += " WHERE scenario_name = ?"
            params = [scenario_name]
        else:
            params = []

        query += " GROUP BY scenario_name ORDER BY scenario_name"

        return pd.read_sql_query(query, self.conn, params=params)

    def _get_results(
            self,
            contract_ids: Optional[List[int]] = None,
            scenario_names: Optional[List[str]] = None,
            min_timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """Internal method to fetch results with filters"""
        query = "SELECT * FROM pricing_summary WHERE 1=1"
        params = []

        if contract_ids:
            placeholders = ','.join('?' * len(contract_ids))
            query += f" AND contract_id IN ({placeholders})"
            params.extend(contract_ids)

        if scenario_names:
            placeholders = ','.join('?' * len(scenario_names))
            query += f" AND scenario_name IN ({placeholders})"
            params.extend(scenario_names)

        if min_timestamp:
            query += " AND pricing_timestamp >= ?"
            params.append(min_timestamp)

        query += " ORDER BY scenario_name, maturity_years, heat_rate"

        return pd.read_sql_query(query, self.conn, params=params)

    def _format_excel_sheet(self, writer, sheet_name: str):
        """Apply basic formatting to Excel worksheet"""
        worksheet = writer.sheets[sheet_name]

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width