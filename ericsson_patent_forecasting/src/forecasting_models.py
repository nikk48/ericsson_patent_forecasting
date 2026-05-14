"""
forecasting_models.py

Modelling utilities for Task 2 and Task 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = str(exc)


@dataclass
class NaiveLastValueRegressor:
    """Forecast each year using the previous year's patent count."""

    lag_feature_name: str = "lag_1"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NaiveLastValueRegressor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_feature_name].to_numpy(dtype=float)


def is_xgboost_available() -> bool:
    """Return True when xgboost is importable and usable."""
    return XGBRegressor is not None


def get_xgboost_unavailable_reason() -> str:
    """Return a human-readable reason when xgboost cannot be used."""
    if is_xgboost_available():
        return ""
    if _XGBOOST_IMPORT_ERROR:
        return _XGBOOST_IMPORT_ERROR
    return "xgboost is not installed in this environment."


def prepare_model_data(
    annual_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "total_patents",
) -> pd.DataFrame:
    """Keep only modelling columns and drop rows with missing inputs."""
    modelling_df = annual_df[["year", target_column] + feature_columns].copy()
    modelling_df = modelling_df.dropna().reset_index(drop=True)
    return modelling_df


def split_time_series_data(
    df: pd.DataFrame,
    train_end_year: int,
    val_end_year: int,
    target_column: str,
    feature_columns: List[str],
    val_start_year: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Split data into chronological train, validation, and test sets."""
    validation_start = train_end_year + 1 if val_start_year is None else val_start_year
    train_df = df[df["year"] <= train_end_year].copy()
    val_df = df[(df["year"] >= validation_start) & (df["year"] <= val_end_year)].copy()
    test_df = df[df["year"] > val_end_year].copy()
    return {
        "X_train": train_df[feature_columns],
        "y_train": train_df[target_column],
        "X_val": val_df[feature_columns],
        "y_val": val_df[target_column],
        "X_test": test_df[feature_columns],
        "y_test": test_df[target_column],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train the interpretable linear regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_naive_last_value(X_train: pd.DataFrame, y_train: pd.Series) -> NaiveLastValueRegressor:
    """Create the naive last-value benchmark."""
    model = NaiveLastValueRegressor()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    hyperparameters: Optional[Dict] = None,
) -> RandomForestRegressor:
    """Train the complex ML Random Forest model."""
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": random_state,
    }
    if hyperparameters:
        params.update(hyperparameters)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    hyperparameters: Optional[Dict] = None,
):
    """Train the complex ML XGBoost model."""
    if XGBRegressor is None:
        raise ImportError(
            "XGBoost is unavailable in this environment. "
            f"Reason: {get_xgboost_unavailable_reason()}"
        )

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": random_state,
    }
    if hyperparameters:
        params.update(hyperparameters)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_model_by_name(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    hyperparameters: Optional[Dict] = None,
):
    """Train one supported model from its display name."""
    if model_name == "Naive Last Value":
        return train_naive_last_value(X_train=X_train, y_train=y_train)
    if model_name == "Linear Regression":
        return train_linear_regression(X_train=X_train, y_train=y_train)
    if model_name == "Random Forest":
        return train_random_forest(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            hyperparameters=hyperparameters,
        )
    if model_name == "XGBoost":
        return train_xgboost(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            hyperparameters=hyperparameters,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def clip_predictions_non_negative(predictions: np.ndarray) -> np.ndarray:
    """Keep forecasts consistent with non-negative patent counts."""
    return np.clip(np.asarray(predictions, dtype=float), 0.0, None)


def calculate_forecast_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, MAPE, and R-squared for one forecast set."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    r2 = np.nan
    if len(y_true) >= 2:
        try:
            r2 = r2_score(y_true, y_pred)
        except ValueError:
            r2 = np.nan
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4) if not pd.isna(r2) else np.nan,
    }


def evaluate_model_on_split(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    clip_non_negative: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Predict on one split and return predictions plus metrics."""
    predictions = model.predict(X)
    if clip_non_negative:
        predictions = clip_predictions_non_negative(predictions)
    metrics = calculate_forecast_metrics(y, predictions)
    return predictions, metrics


def build_metrics_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Convert nested evaluation results into a comparison table."""
    rows = []
    for model_name, split_results in results.items():
        for split_name, metrics in split_results.items():
            row = {"model": model_name, "split": split_name}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def get_linear_regression_coefficients(model: LinearRegression, feature_columns: List[str]) -> pd.DataFrame:
    """Extract linear regression coefficients."""
    coef_df = (
        pd.DataFrame({"feature": feature_columns, "coefficient": model.coef_})
        .sort_values(by="coefficient", ascending=False)
        .reset_index(drop=True)
    )
    intercept_df = pd.DataFrame({"feature": ["intercept"], "coefficient": [model.intercept_]})
    return pd.concat([intercept_df, coef_df], ignore_index=True)


def get_random_forest_feature_importance(model: RandomForestRegressor, feature_columns: List[str]) -> pd.DataFrame:
    """Extract Random Forest feature importances."""
    return (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def get_xgboost_feature_importance(model, feature_columns: List[str]) -> pd.DataFrame:
    """Extract XGBoost feature importances."""
    return (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model_with_rolling_origin(
    model_name: str,
    modelling_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    windows: List[Dict[str, int]],
    random_state: int = 42,
    hyperparameters: Optional[Dict] = None,
    clip_non_negative: bool = False,
) -> pd.DataFrame:
    """Evaluate one model over several rolling-origin validation windows."""
    rows = []
    for window_index, window in enumerate(windows, start=1):
        splits = split_time_series_data(
            df=modelling_df,
            train_end_year=window["train_end_year"],
            val_end_year=window["val_end_year"],
            val_start_year=window["val_start_year"],
            target_column=target_column,
            feature_columns=feature_columns,
        )
        if splits["X_train"].empty or splits["X_val"].empty:
            continue

        model = train_model_by_name(
            model_name=model_name,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            random_state=random_state,
            hyperparameters=hyperparameters,
        )
        _, metrics = evaluate_model_on_split(
            model=model,
            X=splits["X_val"],
            y=splits["y_val"],
            clip_non_negative=clip_non_negative,
        )
        rows.append(
            {
                "model": model_name,
                "window_id": window_index,
                "train_start_year": int(splits["train_df"]["year"].min()),
                "train_end_year": int(window["train_end_year"]),
                "validation_start_year": int(window["val_start_year"]),
                "validation_end_year": int(window["val_end_year"]),
                "n_train_years": int(len(splits["train_df"])),
                "n_validation_years": int(len(splits["val_df"])),
                "hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_rolling_validation(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Average rolling-origin validation metrics for each model/parameter set."""
    if rolling_df.empty:
        return pd.DataFrame()

    summary = (
        rolling_df.groupby(["model", "hyperparameters"], as_index=False)[["MAE", "RMSE", "MAPE", "R2"]]
        .mean()
        .rename(
            columns={
                "MAE": "average_MAE",
                "RMSE": "average_RMSE",
                "MAPE": "average_MAPE",
                "R2": "average_R2",
            }
        )
        .sort_values(by=["average_RMSE", "average_MAE", "model"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return summary


def tune_model_hyperparameters(
    model_name: str,
    modelling_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    windows: List[Dict[str, int]],
    parameter_grid: List[Dict],
    random_state: int = 42,
    clip_non_negative: bool = False,
) -> pd.DataFrame:
    """Evaluate a small hyperparameter grid using rolling-origin validation."""
    tuning_rows = []
    for params in parameter_grid:
        rolling_df = evaluate_model_with_rolling_origin(
            model_name=model_name,
            modelling_df=modelling_df,
            feature_columns=feature_columns,
            target_column=target_column,
            windows=windows,
            random_state=random_state,
            hyperparameters=params,
            clip_non_negative=clip_non_negative,
        )
        if rolling_df.empty:
            continue

        summary = summarize_rolling_validation(rolling_df)
        summary_row = summary.iloc[0].to_dict()
        summary_row["hyperparameters_dict"] = json.dumps(params, sort_keys=True)
        tuning_rows.append(summary_row)

    if not tuning_rows:
        return pd.DataFrame()

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        by=["average_RMSE", "average_MAE"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return tuning_df


def get_best_hyperparameters(tuning_df: pd.DataFrame) -> Dict:
    """Parse the best hyperparameter row from a tuning table."""
    if tuning_df.empty:
        return {}
    return json.loads(tuning_df.iloc[0]["hyperparameters_dict"])


def forecast_future_years_recursive(
    model,
    annual_df: pd.DataFrame,
    feature_columns: List[str],
    future_horizon: int,
    clip_non_negative: bool = False,
) -> pd.DataFrame:
    """Forecast future annual patent counts one year at a time."""
    working_df = annual_df.copy().sort_values("year").reset_index(drop=True)

    last_year = int(working_df["year"].max())
    last_trend = int(working_df["trend"].max())

    share_cols = [
        col for col in working_df.columns
        if col.endswith("_share") and not col.endswith("_share_lag_1")
    ]
    latest_share_values = {}
    for col in share_cols:
        latest_share_values[col] = working_df[col].dropna().iloc[-1]

    forecasts = []

    for step in range(1, future_horizon + 1):
        future_year = last_year + step
        future_trend = last_trend + step

        lag_1 = working_df["total_patents"].iloc[-1]
        lag_2 = working_df["total_patents"].iloc[-2]

        growth_rate = np.nan
        if lag_2 != 0:
            growth_rate = (lag_1 - lag_2) / lag_2

        previous_growth_rate = np.nan
        if "growth_rate" in working_df.columns:
            previous_growth_rate = working_df["growth_rate"].iloc[-1]

        row = {
            "year": future_year,
            "trend": future_trend,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "growth_rate": growth_rate,
            "growth_rate_lag_1": previous_growth_rate,
        }

        for col, value in latest_share_values.items():
            row[col] = value

        for feature_name in feature_columns:
            if feature_name.endswith("_share_lag_1"):
                base_share_col = feature_name.removesuffix("_lag_1")
                if base_share_col in working_df.columns:
                    row[feature_name] = working_df[base_share_col].iloc[-1]
                else:
                    row[feature_name] = latest_share_values.get(base_share_col, np.nan)

        X_future = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_columns}])
        predicted_total = float(model.predict(X_future)[0])
        if clip_non_negative:
            predicted_total = float(clip_predictions_non_negative(np.array([predicted_total]))[0])

        row["total_patents"] = predicted_total
        forecasts.append(row)
        working_df = pd.concat([working_df, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecasts)
