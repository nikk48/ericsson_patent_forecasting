"""
run_forecasting.py

Driver script for Task 2.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_dataset, save_dataframe
from src.feature_engineering import (
    build_feature_ablation_sets,
    create_annual_modelling_dataset,
    create_forecasting_leakage_check_table,
    get_forecasting_feature_columns,
)
from src.forecasting_models import (
    build_metrics_table,
    evaluate_model_on_split,
    evaluate_model_with_rolling_origin,
    forecast_future_years_recursive,
    get_best_hyperparameters,
    get_linear_regression_coefficients,
    get_random_forest_feature_importance,
    get_xgboost_feature_importance,
    is_xgboost_available,
    prepare_model_data,
    split_time_series_data,
    summarize_rolling_validation,
    train_model_by_name,
    tune_model_hyperparameters,
    get_xgboost_unavailable_reason,
)
from src.project_config import (
    build_parameter_grid,
    get_core_settings,
    get_rolling_validation_windows,
)
from src.time_series_diagnostics import build_vif_table
from src.utils import ensure_directory, get_keyword_columns


CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_patents.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "total_patents"
CORE_SETTINGS = get_core_settings()
TRAIN_END_YEAR = CORE_SETTINGS["train_end_year"]
VAL_END_YEAR = CORE_SETTINGS["val_end_year"]
FUTURE_HORIZON = CORE_SETTINGS["future_horizon"]
RANDOM_STATE = CORE_SETTINGS["random_state"]
CLIP_NON_NEGATIVE = CORE_SETTINGS["enforce_non_negative_forecasts"]


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
    safe_name = model_name.lower().replace(" ", "_")
    return f"{safe_name}_test_predictions.csv"


def _future_output_filename(model_name: str) -> str:
    """Return the output filename for per-model future forecasts."""
    safe_name = model_name.lower().replace(" ", "_")
    return f"{safe_name}_future_forecasts.csv"


def _tune_complex_models(
    modelling_df: pd.DataFrame,
    feature_columns,
    rolling_windows,
) -> dict:
    """Tune Random Forest and XGBoost with small rolling-origin grids."""
    selected_hyperparameters = {}

    rf_tuning_df = tune_model_hyperparameters(
        model_name="Random Forest",
        modelling_df=modelling_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        windows=rolling_windows,
        parameter_grid=build_parameter_grid("random_forest_grid"),
        random_state=RANDOM_STATE,
        clip_non_negative=CLIP_NON_NEGATIVE,
    )
    if not rf_tuning_df.empty:
        save_dataframe(rf_tuning_df, str(TABLES_DIR / "random_forest_tuning_validation_results.csv"))
        selected_hyperparameters["Random Forest"] = get_best_hyperparameters(rf_tuning_df)

    if is_xgboost_available():
        xgb_tuning_df = tune_model_hyperparameters(
            model_name="XGBoost",
            modelling_df=modelling_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            windows=rolling_windows,
            parameter_grid=build_parameter_grid("xgboost_grid"),
            random_state=RANDOM_STATE,
            clip_non_negative=CLIP_NON_NEGATIVE,
        )
        if not xgb_tuning_df.empty:
            save_dataframe(xgb_tuning_df, str(TABLES_DIR / "xgboost_tuning_validation_results.csv"))
            selected_hyperparameters["XGBoost"] = get_best_hyperparameters(xgb_tuning_df)

    return selected_hyperparameters


def _build_feature_ablation_table(
    annual_df: pd.DataFrame,
    feature_sets,
    rolling_windows,
    model_hyperparameters_by_name,
) -> pd.DataFrame:
    """Compare progressive feature sets for the main forecasting models."""
    ablation_rows = []
    for feature_set in feature_sets:
        feature_columns = feature_set["features"]
        modelling_df = prepare_model_data(
            annual_df=annual_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
        )
        if modelling_df.empty:
            continue

        fixed_splits = split_time_series_data(
            df=modelling_df,
            train_end_year=TRAIN_END_YEAR,
            val_end_year=VAL_END_YEAR,
            target_column=TARGET_COLUMN,
            feature_columns=feature_columns,
        )
        train_val_df = modelling_df[modelling_df["year"] <= VAL_END_YEAR].copy()

        for model_name in ["Linear Regression", "Random Forest"]:
            rolling_df = evaluate_model_with_rolling_origin(
                model_name=model_name,
                modelling_df=modelling_df,
                feature_columns=feature_columns,
                target_column=TARGET_COLUMN,
                windows=rolling_windows,
                random_state=RANDOM_STATE,
                hyperparameters=model_hyperparameters_by_name.get(model_name),
                clip_non_negative=CLIP_NON_NEGATIVE,
            )
            if rolling_df.empty:
                continue
            rolling_summary = summarize_rolling_validation(rolling_df).iloc[0]

            test_model = train_model_by_name(
                model_name=model_name,
                X_train=train_val_df[feature_columns],
                y_train=train_val_df[TARGET_COLUMN],
                random_state=RANDOM_STATE,
                hyperparameters=model_hyperparameters_by_name.get(model_name),
            )
            _, test_metrics = evaluate_model_on_split(
                model=test_model,
                X=fixed_splits["X_test"],
                y=fixed_splits["y_test"],
                clip_non_negative=CLIP_NON_NEGATIVE,
            )
            ablation_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set["feature_set"],
                    "description": feature_set["description"],
                    "n_features": len(feature_columns),
                    "average_rolling_MAE": rolling_summary["average_MAE"],
                    "average_rolling_RMSE": rolling_summary["average_RMSE"],
                    "average_rolling_MAPE": rolling_summary["average_MAPE"],
                    "average_rolling_R2": rolling_summary["average_R2"],
                    "test_MAE": test_metrics["MAE"],
                    "test_RMSE": test_metrics["RMSE"],
                    "test_MAPE": test_metrics["MAPE"],
                    "test_R2": test_metrics["R2"],
                }
            )

    return pd.DataFrame(ablation_rows).sort_values(
        by=["model", "average_rolling_RMSE", "test_RMSE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


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

    print("Selecting forecasting features and checking leakage...")
    feature_columns = get_forecasting_feature_columns(annual_df)
    leakage_check_df = create_forecasting_leakage_check_table(feature_columns)
    save_dataframe(leakage_check_df, str(TABLES_DIR / "task2_leakage_feature_check.csv"))
    modelling_df = prepare_model_data(
        annual_df=annual_df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    save_dataframe(modelling_df, str(PROCESSED_DIR / "task2_modelling_dataset.csv"))

    rolling_windows = get_rolling_validation_windows()

    model_names = ["Naive Last Value", "Linear Regression", "Random Forest"]
    if is_xgboost_available():
        model_names.append("XGBoost")
    else:
        print("XGBoost is unavailable; continuing without it.")
        print(f"XGBoost unavailable reason: {get_xgboost_unavailable_reason()}")

    print("Tuning complex models with rolling-origin validation...")
    model_hyperparameters_by_name = _tune_complex_models(
        modelling_df=modelling_df,
        feature_columns=feature_columns,
        rolling_windows=rolling_windows,
    )
    selected_hyperparameters_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "selected_hyperparameters": str(model_hyperparameters_by_name.get(model_name, {})),
            }
            for model_name in model_names
        ]
    )
    save_dataframe(selected_hyperparameters_df, str(TABLES_DIR / "task2_selected_hyperparameters.csv"))

    print("Running rolling-origin validation summaries...")
    rolling_frames = []
    for model_name in model_names:
        rolling_df = evaluate_model_with_rolling_origin(
            model_name=model_name,
            modelling_df=modelling_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            windows=rolling_windows,
            random_state=RANDOM_STATE,
            hyperparameters=model_hyperparameters_by_name.get(model_name),
            clip_non_negative=CLIP_NON_NEGATIVE,
        )
        if not rolling_df.empty:
            rolling_frames.append(rolling_df)

    if not rolling_frames:
        raise ValueError("Rolling-origin validation did not produce any evaluation windows.")

    rolling_detailed_df = pd.concat(rolling_frames, ignore_index=True)
    rolling_summary_df = summarize_rolling_validation(rolling_detailed_df)
    save_dataframe(rolling_detailed_df, str(TABLES_DIR / "task2_rolling_validation_detailed.csv"))
    save_dataframe(rolling_summary_df, str(TABLES_DIR / "task2_rolling_validation_metrics.csv"))

    print("Running feature ablation study...")
    feature_ablation_df = _build_feature_ablation_table(
        annual_df=annual_df,
        feature_sets=build_feature_ablation_sets(annual_df),
        rolling_windows=rolling_windows,
        model_hyperparameters_by_name=model_hyperparameters_by_name,
    )
    save_dataframe(feature_ablation_df, str(TABLES_DIR / "task2_feature_ablation.csv"))

    print("Splitting data into train / validation / test...")
    splits = split_time_series_data(
        df=modelling_df,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
    )
    vif_df = build_vif_table(splits["X_train"])
    save_dataframe(vif_df, str(TABLES_DIR / "task2_vif_results.csv"))
    train_val_df = modelling_df[modelling_df["year"] <= VAL_END_YEAR].copy()
    X_train_val = train_val_df[feature_columns]
    y_train_val = train_val_df[TARGET_COLUMN]

    training_metrics_by_model = {}
    validation_metrics_by_model = {}
    test_metrics_by_model = {}
    test_predictions_by_model = {}

    print("Evaluating models on fixed validation and test windows...")
    for model_name in model_names:
        validation_model = train_model_by_name(
            model_name=model_name,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            random_state=RANDOM_STATE,
            hyperparameters=model_hyperparameters_by_name.get(model_name),
        )
        _, training_metrics_by_model[model_name] = evaluate_model_on_split(
            validation_model,
            splits["X_train"],
            splits["y_train"],
            clip_non_negative=CLIP_NON_NEGATIVE,
        )
        _, validation_metrics_by_model[model_name] = evaluate_model_on_split(
            validation_model,
            splits["X_val"],
            splits["y_val"],
            clip_non_negative=CLIP_NON_NEGATIVE,
        )

        test_model = train_model_by_name(
            model_name=model_name,
            X_train=X_train_val,
            y_train=y_train_val,
            random_state=RANDOM_STATE,
            hyperparameters=model_hyperparameters_by_name.get(model_name),
        )
        test_pred, test_metrics_by_model[model_name] = evaluate_model_on_split(
            test_model,
            splits["X_test"],
            splits["y_test"],
            clip_non_negative=CLIP_NON_NEGATIVE,
        )
        test_predictions_by_model[model_name] = test_pred
        _save_model_interpretation_outputs(model_name, test_model, feature_columns, TABLES_DIR)

    results = {
        model_name: {
            "training": training_metrics_by_model[model_name],
            "validation": validation_metrics_by_model[model_name],
            "test": test_metrics_by_model[model_name],
        }
        for model_name in model_names
    }
    metrics_table = build_metrics_table(results)
    save_dataframe(metrics_table, str(TABLES_DIR / "task2_model_metrics.csv"))

    for model_name in model_names:
        test_results = splits["test_df"][["year", TARGET_COLUMN]].copy()
        test_results["predicted_total_patents"] = test_predictions_by_model[model_name]
        test_results["model"] = model_name
        save_dataframe(test_results, str(TABLES_DIR / _prediction_output_filename(model_name)))

    comparison_models = ["Random Forest"]
    if "XGBoost" in model_names:
        comparison_models.append("XGBoost")
    rf_vs_xgb = metrics_table[metrics_table["model"].isin(comparison_models)].copy()
    save_dataframe(rf_vs_xgb, str(TABLES_DIR / "random_forest_vs_xgboost_metrics.csv"))

    best_row = rolling_summary_df.sort_values(
        by=["average_RMSE", "average_MAE", "model"],
        ascending=[True, True, True],
    ).iloc[0]
    best_model_name = best_row["model"]
    print(f"Best rolling-validation model selected: {best_model_name}")

    print("Refitting models on all observed years for future forecasting...")
    for model_name in model_names:
        future_model = train_model_by_name(
            model_name=model_name,
            X_train=modelling_df[feature_columns],
            y_train=modelling_df[TARGET_COLUMN],
            random_state=RANDOM_STATE,
            hyperparameters=model_hyperparameters_by_name.get(model_name),
        )
        future_forecasts = forecast_future_years_recursive(
            model=future_model,
            annual_df=annual_df,
            feature_columns=feature_columns,
            future_horizon=FUTURE_HORIZON,
            clip_non_negative=CLIP_NON_NEGATIVE,
        )
        future_forecasts["model"] = model_name
        save_dataframe(future_forecasts, str(TABLES_DIR / _future_output_filename(model_name)))
        if model_name == best_model_name:
            selected_future = future_forecasts.copy()
            selected_future["selected_model"] = model_name
            save_dataframe(selected_future, str(TABLES_DIR / "task2_future_forecasts.csv"))

    holdout_best_row = metrics_table[metrics_table["split"] == "test"].sort_values(
        by=["RMSE", "MAE", "model"],
        ascending=[True, True, True],
    ).iloc[0]
    recommended_model_summary = pd.DataFrame(
        [
            {
                "summary_type": "selected_for_future_forecast",
                "model": best_model_name,
                "selection_basis": "lowest average rolling-origin validation RMSE",
                "selection_metrics": (
                    f"average_RMSE={best_row['average_RMSE']}, "
                    f"average_MAE={best_row['average_MAE']}, "
                    f"average_MAPE={best_row['average_MAPE']}"
                ),
                "selected_hyperparameters": str(model_hyperparameters_by_name.get(best_model_name, {})),
                "non_negative_constraint_applied": CLIP_NON_NEGATIVE,
            },
            {
                "summary_type": "best_holdout_test_model",
                "model": holdout_best_row["model"],
                "selection_basis": "lowest fixed-window holdout test RMSE",
                "selection_metrics": (
                    f"RMSE={holdout_best_row['RMSE']}, "
                    f"MAE={holdout_best_row['MAE']}, "
                    f"MAPE={holdout_best_row['MAPE']}"
                ),
                "selected_hyperparameters": str(model_hyperparameters_by_name.get(holdout_best_row["model"], {})),
                "non_negative_constraint_applied": CLIP_NON_NEGATIVE,
            },
        ]
    )
    save_dataframe(recommended_model_summary, str(TABLES_DIR / "recommended_model_summary.csv"))

    print("----- TASK 2 MODEL PERFORMANCE -----")
    print(metrics_table.to_string(index=False))
    print("----- TASK 2 ROLLING-ORIGIN VALIDATION -----")
    print(rolling_summary_df.to_string(index=False))
    print("Forecasting feature columns used:")
    for col in feature_columns:
        print(f" - {col}")

    print("Task 2 forecasting pipeline completed successfully.")
    print(f"Outputs saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
