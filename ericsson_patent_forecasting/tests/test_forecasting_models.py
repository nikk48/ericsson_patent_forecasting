import sys
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.forecasting_models import calculate_forecast_metrics, split_time_series_data, train_model_by_name


class ForecastingModelTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "year": [2000, 2001, 2002, 2003, 2004, 2005],
                "total_patents": [10, 12, 14, 16, 18, 20],
                "trend": [1, 2, 3, 4, 5, 6],
                "lag_1": [8, 10, 12, 14, 16, 18],
                "lag_2": [6, 8, 10, 12, 14, 16],
                "growth_rate_lag_1": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
            }
        )
        self.feature_columns = ["trend", "lag_1", "lag_2", "growth_rate_lag_1"]

    def test_train_validation_test_years_do_not_overlap(self):
        splits = split_time_series_data(
            df=self.df,
            train_end_year=2002,
            val_end_year=2004,
            target_column="total_patents",
            feature_columns=self.feature_columns,
        )
        train_years = set(splits["train_df"]["year"])
        val_years = set(splits["val_df"]["year"])
        test_years = set(splits["test_df"]["year"])
        self.assertTrue(train_years.isdisjoint(val_years))
        self.assertTrue(train_years.isdisjoint(test_years))
        self.assertTrue(val_years.isdisjoint(test_years))

    def test_naive_last_value_model_uses_lag_1(self):
        model = train_model_by_name(
            model_name="Naive Last Value",
            X_train=self.df[self.feature_columns],
            y_train=self.df["total_patents"],
        )
        predictions = model.predict(self.df[self.feature_columns])
        self.assertListEqual(predictions.tolist(), self.df["lag_1"].astype(float).tolist())

    def test_forecast_metrics_include_r_squared(self):
        y_true = pd.Series([10, 12, 14, 16])
        y_pred = pd.Series([10, 12, 14, 16], dtype=float)
        metrics = calculate_forecast_metrics(y_true, y_pred)
        self.assertIn("R2", metrics)
        self.assertEqual(metrics["R2"], 1.0)


if __name__ == "__main__":
    unittest.main()
