import sys
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import (
    create_annual_modelling_dataset,
    create_annual_patent_counts,
    create_forecasting_leakage_check_table,
    create_keyword_shares,
    create_keyword_trends,
    get_forecasting_feature_columns,
)


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "year": [2000, 2000, 2001, 2001, 2001, 2002],
                "kw_5g": [0, 1, 1, 0, 1, 0],
                "kw_ai_ml": [1, 0, 1, 1, 0, 1],
            }
        )
        self.keyword_columns = ["kw_5g", "kw_ai_ml"]

    def test_annual_counts_sum_matches_rows(self):
        annual_counts = create_annual_patent_counts(self.df)
        self.assertEqual(int(annual_counts["total_patents"].sum()), len(self.df))

    def test_keyword_shares_stay_between_zero_and_one(self):
        annual_counts = create_annual_patent_counts(self.df)
        keyword_trends = create_keyword_trends(self.df, self.keyword_columns)
        keyword_shares = create_keyword_shares(annual_counts, keyword_trends, self.keyword_columns)
        share_columns = [column for column in keyword_shares.columns if column.endswith("_share")]
        for column in share_columns:
            self.assertTrue(keyword_shares[column].between(0, 1).all())

    def test_lag_1_is_shifted_correctly(self):
        annual_df = create_annual_modelling_dataset(self.df, keyword_columns=self.keyword_columns)
        self.assertTrue(pd.isna(annual_df.loc[0, "lag_1"]))
        self.assertEqual(annual_df.loc[1, "lag_1"], annual_df.loc[0, "total_patents"])

    def test_default_forecasting_features_are_leakage_safe(self):
        annual_df = create_annual_modelling_dataset(self.df, keyword_columns=self.keyword_columns)
        feature_columns = get_forecasting_feature_columns(annual_df)
        leakage_df = create_forecasting_leakage_check_table(feature_columns)
        self.assertTrue(leakage_df["is_leakage_safe"].all())


if __name__ == "__main__":
    unittest.main()
