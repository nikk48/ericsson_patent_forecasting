"""
forecasting_models.py

This file contains the modelling logic for Task 2.

Task 2 asks us to:
- forecast total annual patent counts
- use one interpretable model
- use one complex machine learning model
- compare them with standard error metrics

In this file:
- Linear Regression is used as the interpretable model
- Random Forest is used as one complex machine learning model
- XGBoost is used as another complex machine learning model

Why include both Random Forest and XGBoost?
- Random Forest is robust, simple, and a strong nonlinear benchmark
- XGBoost often achieves higher accuracy because it learns sequentially by correcting earlier errors
- comparing both helps us justify model choice using evidence rather than assumption
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    _XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = str(exc)


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
    target_column: str = "total_patents"
) -> pd.DataFrame:
    """
    Keep only the modelling columns and remove rows with missing values.

    Why missing values appear:
    Lag variables and growth rate cannot be calculated for the first one or two years.
    For example:
    - lag_1 is missing for the very first year
    - lag_2 is missing for the first two years

    Why this step matters:
    Most machine learning models cannot train on rows with missing input values.
    """
    modelling_df = annual_df[["year", target_column] + feature_columns].copy()
    modelling_df = modelling_df.dropna().reset_index(drop=True)
    return modelling_df



def split_time_series_data(
    df: pd.DataFrame,
    train_end_year: int,
    val_end_year: int,
    target_column: str,
    feature_columns: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Split the data into train, validation, and test sets using time order.

    Why this matters:
    In forecasting, we must respect chronology.
    We should learn from the past and predict the future.
    We must never randomly mix future years into the training data.

    Structure:
    - training set: used to fit the model
    - validation set: used to compare models
    - test set: used for final unbiased evaluation
    """
    train_df = df[df["year"] <= train_end_year].copy()
    val_df = df[(df["year"] > train_end_year) & (df["year"] <= val_end_year)].copy()
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
    """
    Train the interpretable forecasting model: Linear Regression.

    Why this model is used:
    Linear regression is simple, transparent, and easy to explain.
    It helps us understand how each feature affects patent counts.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model



def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    """
    Train the complex ML forecasting model: Random Forest.

    Why this model is used:
    Random Forest can capture:
    - nonlinear relationships
    - interactions between variables
    - more complex patterns than linear regression

    Why it is suitable here:
    It is powerful but still relatively stable and easy to use on a medium-sized dataset.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model



def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Train the second complex ML forecasting model: XGBoost.

    Why this model is used:
    XGBoost builds trees sequentially, where each new tree focuses on correcting
    the mistakes made by earlier trees. This often leads to very strong predictive performance.

    Why it is useful in this coursework:
    Task 1 showed that patent activity is nonlinear and volatile.
    XGBoost is often strong in exactly this kind of setting.
    """
    if XGBRegressor is None:
        raise ImportError(
            "XGBoost is unavailable in this environment. "
            f"Reason: {get_xgboost_unavailable_reason()}"
        )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model



def calculate_forecast_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecasting error metrics.

    Metrics used:
    - MAE: average absolute error
    - RMSE: gives more penalty to large mistakes
    - MAPE: percentage error
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "MAPE": round(float(mape), 4)}



def evaluate_model_on_split(model, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Use a trained model to make predictions and evaluate them.
    """
    predictions = model.predict(X)
    metrics = calculate_forecast_metrics(y, predictions)
    return predictions, metrics



def build_metrics_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Convert model evaluation results into a neat comparison table.

    Why this matters:
    This allows direct comparison of Linear Regression, Random Forest, and XGBoost
    across validation and test sets.
    """
    rows = []
    for model_name, split_results in results.items():
        for split_name, metrics in split_results.items():
            row = {"model": model_name, "split": split_name}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)



def get_linear_regression_coefficients(model: LinearRegression, feature_columns: List[str]) -> pd.DataFrame:
    """
    Extract coefficients from the linear regression model.
    """
    coef_df = pd.DataFrame({"feature": feature_columns, "coefficient": model.coef_}).sort_values(by="coefficient", ascending=False).reset_index(drop=True)
    intercept_df = pd.DataFrame({"feature": ["intercept"], "coefficient": [model.intercept_]})
    return pd.concat([intercept_df, coef_df], ignore_index=True)



def get_random_forest_feature_importance(model: RandomForestRegressor, feature_columns: List[str]) -> pd.DataFrame:
    """
    Extract feature importance values from the Random Forest model.
    """
    importance_df = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return importance_df



def get_xgboost_feature_importance(model, feature_columns: List[str]) -> pd.DataFrame:
    """
    Extract feature importance values from the XGBoost model.

    Why this matters:
    XGBoost is complex, so feature importance helps us understand
    which variables contributed most to the predictions.
    """
    importance_df = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_}).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return importance_df



def forecast_future_years_recursive(
    model,
    annual_df: pd.DataFrame,
    feature_columns: List[str],
    future_horizon: int,
) -> pd.DataFrame:
    """
    Forecast future annual patent counts one year at a time.

    Why this is called recursive forecasting:
    Once we move into the future, actual patent counts are unknown.
    So after predicting the next year, we use that predicted value
    as an input for forecasting the following year.

    Simple assumption used here:
    Technology shares are carried forward from the latest known year.
    Because the model uses lagged shares, this keeps future forecasting
    aligned with the information pattern used during training.
    """
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

        row["total_patents"] = predicted_total
        forecasts.append(row)
        working_df = pd.concat([working_df, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecasts)
