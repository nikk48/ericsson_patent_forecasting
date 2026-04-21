"""
run_forecasting.py

This is the driver script for Task 2.

Task 2 asks us to:
- forecast total annual patent counts
- build one interpretable model
- build one complex machine learning model
- compare them using forecast accuracy metrics
- generate future forecasts

This script trains three models:
- Linear Regression (interpretable baseline)
- Random Forest (complex ML benchmark)
- XGBoost (advanced boosting model)

Important modelling rule:
All forecasting inputs must be available before the year being predicted.
That means this script uses lagged features only for model fitting.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import ensure_directory, get_keyword_columns
from src.data_loader import load_dataset, save_dataframe
from src.feature_engineering import (
    create_annual_modelling_dataset,
    get_forecasting_feature_columns,
)
from src.forecasting_models import (
    prepare_model_data,
    split_time_series_data,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model_on_split,
    build_metrics_table,
    get_linear_regression_coefficients,
    get_random_forest_feature_importance,
    get_xgboost_feature_importance,
    forecast_future_years_recursive,
    is_xgboost_available,
    get_xgboost_unavailable_reason,
)

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_patents.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "total_patents"
TRAIN_END_YEAR = 2010
VAL_END_YEAR = 2016
FUTURE_HORIZON = 5


def _train_model_by_name(model_name: str, X_train, y_train):
    """Train a model based on its display name."""
    if model_name == "Linear Regression":
        return train_linear_regression(X_train=X_train, y_train=y_train)
    if model_name == "Random Forest":
        return train_random_forest(X_train=X_train, y_train=y_train, random_state=42)
    if model_name == "XGBoost":
        return train_xgboost(X_train=X_train, y_train=y_train, random_state=42)
    raise ValueError(f"Unsupported model name: {model_name}")


def _save_model_interpretation_outputs(model_name: str, model, feature_columns, tables_dir: Path) -> None:
    """Save coefficients or feature importances for a trained model."""
    if model_name == "Linear Regression":
        linear_coef_df = get_linear_regression_coefficients(model, feature_columns=feature_columns)
        save_dataframe(linear_coef_df, str(tables_dir / "linear_regression_coefficients.csv"))
    elif model_name == "Random Forest":
        rf_importance_df = get_random_forest_feature_importance(model, feature_columns=feature_columns)
        save_dataframe(rf_importance_df, str(tables_dir / "random_forest_feature_importance.csv"))
    elif model_name == "XGBoost":
        xgb_importance_df = get_xgboost_feature_importance(model, feature_columns=feature_columns)
        save_dataframe(xgb_importance_df, str(tables_dir / "xgboost_feature_importance.csv"))


def _prediction_output_filename(model_name: str) -> str:
    """Return the output filename for test predictions."""
    if model_name == "Linear Regression":
        return "linear_regression_test_predictions.csv"
    if model_name == "Random Forest":
        return "random_forest_test_predictions.csv"
    if model_name == "XGBoost":
        return "xgboost_test_predictions.csv"
    safe_name = model_name.lower().replace(" ", "_")
    return f"{safe_name}_test_predictions.csv"


def main() -> None:
    """Run the full Task 2 forecasting pipeline."""
    ensure_directory(str(TABLES_DIR))
    ensure_directory(str(PROCESSED_DIR))

    print("Loading cleaned patent data...")
    df_clean = load_dataset(str(CLEAN_DATA_PATH))

    print("Building annual modelling dataset...")
    keyword_columns = get_keyword_columns()
    annual_df = create_annual_modelling_dataset(df_clean, keyword_columns=keyword_columns)
    save_dataframe(annual_df, str(PROCESSED_DIR / "annual_modelling_dataset.csv"))

    print("Selecting forecasting features...")
    feature_columns = get_forecasting_feature_columns(annual_df)
    modelling_df = prepare_model_data(
        annual_df=annual_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    save_dataframe(modelling_df, str(PROCESSED_DIR / "task2_modelling_dataset.csv"))

    print("Splitting data into train / validation / test...")
    splits = split_time_series_data(
        df=modelling_df,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
    )
    train_val_df = modelling_df[modelling_df["year"] <= VAL_END_YEAR].copy()
    X_train_val = train_val_df[feature_columns]
    y_train_val = train_val_df[TARGET_COLUMN]

    model_names = ["Linear Regression", "Random Forest"]
    if is_xgboost_available():
        model_names.append("XGBoost")
    else:
        print("XGBoost is unavailable; continuing with Linear Regression and Random Forest.")
        print(f"XGBoost unavailable reason: {get_xgboost_unavailable_reason()}")
    validation_metrics_by_model = {}
    test_metrics_by_model = {}
    test_predictions_by_model = {}

    print("Evaluating models on validation split...")
    for model_name in model_names:
        print(f"Training {model_name} on train split...")
        validation_model = _train_model_by_name(model_name, splits["X_train"], splits["y_train"])
        _, validation_metrics_by_model[model_name] = evaluate_model_on_split(
            validation_model,
            splits["X_val"],
            splits["y_val"],
        )

    print("Retraining models on train + validation split for test evaluation...")
    for model_name in model_names:
        print(f"Training {model_name} on train + validation split...")
        test_model = _train_model_by_name(model_name, X_train_val, y_train_val)
        test_pred, test_metrics_by_model[model_name] = evaluate_model_on_split(
            test_model,
            splits["X_test"],
            splits["y_test"],
        )
        test_predictions_by_model[model_name] = test_pred
        _save_model_interpretation_outputs(model_name, test_model, feature_columns, TABLES_DIR)

    # -----------------------------
    # Comparison table for all models
    # -----------------------------
    results = {
        model_name: {
            "validation": validation_metrics_by_model[model_name],
            "test": test_metrics_by_model[model_name],
        }
        for model_name in model_names
    }
    metrics_table = build_metrics_table(results)
    save_dataframe(metrics_table, str(TABLES_DIR / "task2_model_metrics.csv"))

    # -----------------------------
    # Save actual vs predicted for each model
    # -----------------------------
    for model_name in model_names:
        test_results = splits["test_df"][["year", TARGET_COLUMN]].copy()
        test_results["predicted_total_patents"] = test_predictions_by_model[model_name]
        test_results["model"] = model_name
        save_dataframe(test_results, str(TABLES_DIR / _prediction_output_filename(model_name)))

    # -----------------------------
    # Compare Random Forest vs XGBoost directly
    # -----------------------------
    comparison_models = ["Random Forest"]
    if "XGBoost" in model_names:
        comparison_models.append("XGBoost")
    rf_vs_xgb = metrics_table[metrics_table["model"].isin(comparison_models)].copy()
    save_dataframe(rf_vs_xgb, str(TABLES_DIR / "random_forest_vs_xgboost_metrics.csv"))

    # -----------------------------
    # Select best model using validation RMSE
    # -----------------------------
    validation_rows = metrics_table[metrics_table["split"] == "validation"].copy()
    best_row = validation_rows.sort_values(by="RMSE", ascending=True).iloc[0]
    best_model_name = best_row["model"]
    print(f"Best validation model selected: {best_model_name}")

    # -----------------------------
    # Refit selected model on all observed data, then forecast future years
    # -----------------------------
    print("Refitting selected model on all observed years for final future forecasting...")
    best_model = _train_model_by_name(
        best_model_name,
        modelling_df[feature_columns],
        modelling_df[TARGET_COLUMN],
    )
    future_forecasts = forecast_future_years_recursive(
        model=best_model,
        annual_df=annual_df,
        feature_columns=feature_columns,
        future_horizon=FUTURE_HORIZON,
    )
    future_forecasts["selected_model"] = best_model_name
    save_dataframe(future_forecasts, str(TABLES_DIR / "task2_future_forecasts.csv"))

    # Also save separate future forecasts for Random Forest and XGBoost so they can be compared directly
    rf_future_model = _train_model_by_name(
        "Random Forest",
        modelling_df[feature_columns],
        modelling_df[TARGET_COLUMN],
    )
    rf_future_forecasts = forecast_future_years_recursive(
        model=rf_future_model,
        annual_df=annual_df,
        feature_columns=feature_columns,
        future_horizon=FUTURE_HORIZON,
    )
    rf_future_forecasts["model"] = "Random Forest"

    save_dataframe(rf_future_forecasts, str(TABLES_DIR / "random_forest_future_forecasts.csv"))
    if "XGBoost" in model_names:
        xgb_future_model = _train_model_by_name(
            "XGBoost",
            modelling_df[feature_columns],
            modelling_df[TARGET_COLUMN],
        )
        xgb_future_forecasts = forecast_future_years_recursive(
            model=xgb_future_model,
            annual_df=annual_df,
            feature_columns=feature_columns,
            future_horizon=FUTURE_HORIZON,
        )
        xgb_future_forecasts["model"] = "XGBoost"
        save_dataframe(xgb_future_forecasts, str(TABLES_DIR / "xgboost_future_forecasts.csv"))

    # -----------------------------
    # Console summary
    # -----------------------------
    print("----- TASK 2 MODEL PERFORMANCE -----")
    print(metrics_table.to_string(index=False))

    print("----- RANDOM FOREST VS XGBOOST -----")
    print(rf_vs_xgb.to_string(index=False))

    print("Forecasting feature columns used:")
    for col in feature_columns:
        print(f" - {col}")

    print("Future forecasts from selected best model:")
    print(future_forecasts[["year", "total_patents", "selected_model"]].to_string(index=False))

    print("Task 2 forecasting pipeline completed successfully.")
    print(f"Outputs saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
