import sys
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.time_series_diagnostics import (
    build_acf_table,
    build_adf_results,
    build_pacf_table,
    build_vif_table,
)


class TimeSeriesDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series([10, 12, 15, 14, 18, 17, 20, 22, 21, 25, 24, 27], dtype=float)
        self.feature_df = pd.DataFrame(
            {
                "trend": list(range(1, 13)),
                "lag_1": [9, 10, 12, 15, 14, 18, 17, 20, 22, 21, 25, 24],
                "lag_2": [8, 9, 10, 12, 15, 14, 18, 17, 20, 22, 21, 25],
            }
        )

    def test_adf_results_include_level_and_first_difference(self):
        adf_df = build_adf_results(self.series, series_name="total_patents")
        self.assertIn("level", adf_df["transform"].tolist())
        self.assertIn("first_difference", adf_df["transform"].tolist())

    def test_vif_table_has_one_row_per_feature(self):
        vif_df = build_vif_table(self.feature_df)
        self.assertEqual(len(vif_df), len(self.feature_df.columns))
        self.assertTrue((vif_df["VIF"] >= 0).all())

    def test_acf_and_pacf_tables_start_at_lag_zero(self):
        acf_df = build_acf_table(self.series)
        pacf_df = build_pacf_table(self.series)
        self.assertEqual(int(acf_df.iloc[0]["lag"]), 0)
        self.assertEqual(int(pacf_df.iloc[0]["lag"]), 0)


if __name__ == "__main__":
    unittest.main()
