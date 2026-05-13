"""
run_task3.py

Driver script for Task 3 segmentation-based forecasting.
"""

import ast
import os
from pathlib import Path
import sys
import warnings

# Avoid noisy loky core-detection warnings on some macOS Python builds.
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(min(8, os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_dataset, save_dataframe
from src.feature_engineering import create_annual_patent_counts
from src.project_config import (
    get_core_settings,
    get_task3_candidate_k_values,
    get_text_segmentation_grid,
)
from src.segmentation_forecasting import (
    build_non_text_segmentation,
    build_task3_baseline_comparison,
    build_text_segmentation,
    create_cluster_balance_table,
    create_yearly_cluster_counts,
    run_segmented_forecast_for_scheme,
)
from src.utils import ensure_directory


CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_patents.csv"
TASK2_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "task2_model_metrics.csv"
TASK2_LINEAR_FUTURE_PATH = PROJECT_ROOT / "outputs" / "tables" / "linear_regression_future_forecasts.csv"
TASK2_RF_FUTURE_PATH = PROJECT_ROOT / "outputs" / "tables" / "random_forest_future_forecasts.csv"
TASK2_HYPERPARAMETERS_PATH = PROJECT_ROOT / "outputs" / "tables" / "task2_selected_hyperparameters.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CORE_SETTINGS = get_core_settings()
TRAIN_END_YEAR = CORE_SETTINGS["train_end_year"]
VAL_END_YEAR = CORE_SETTINGS["val_end_year"]
FUTURE_HORIZON = CORE_SETTINGS["future_horizon"]
RANDOM_STATE = CORE_SETTINGS["random_state"]
CLIP_NON_NEGATIVE = CORE_SETTINGS["enforce_non_negative_forecasts"]
TASK3_MODELS = ["Linear Regression", "Random Forest"]


def _load_task2_hyperparameters() -> dict:
    """Load tuned Task 2 hyperparameters for reuse in Task 3."""
    if not TASK2_HYPERPARAMETERS_PATH.exists():
        return {}

    hyperparameter_df = load_dataset(str(TASK2_HYPERPARAMETERS_PATH))
    result = {}
    for _, row in hyperparameter_df.iterrows():
        try:
            result[row["model"]] = ast.literal_eval(row["selected_hyperparameters"])
        except (ValueError, SyntaxError):
            result[row["model"]] = {}
    return result


def _build_limitations_summary(
    annual_totals: pd.DataFrame,
    text_k_selection: pd.DataFrame,
    non_text_k_selection: pd.DataFrame,
    cluster_balance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a small limitations table to support the report narrative."""
    smallest_clusters = (
        cluster_balance_df.sort_values(by=["portfolio_share", "scheme"], ascending=[True, True])
        .groupby("scheme", as_index=False)
        .first()[["scheme", "cluster", "portfolio_share"]]
    )
    imbalance_summary = "; ".join(
        f"{row['scheme']} smallest cluster {row['cluster']} share={row['portfolio_share']}"
        for _, row in smallest_clusters.iterrows()
    )

    text_best_silhouette = float(text_k_selection["silhouette_score"].max())
    non_text_best_silhouette = float(non_text_k_selection["silhouette_score"].max())

    rows = [
        {
            "theme": "sample_size",
            "evidence": f"{len(annual_totals)} annual observations available for forecasting.",
            "implication": "Limited annual history constrains the reliability of flexible models and multi-step forecasts.",
        },
        {
            "theme": "recursive_forecasting",
            "evidence": f"Future horizon fixed at {FUTURE_HORIZON} years with recursive lag updates.",
            "implication": "Forecast errors can compound as each future step depends on earlier predictions.",
        },
        {
            "theme": "future_covariate_assumption",
            "evidence": "Lagged keyword-share features are carried forward from the most recent observed year.",
            "implication": "Future forecasts may miss shifts in Ericsson's technology mix if keyword shares change materially.",
        },
        {
            "theme": "text_cluster_separation",
            "evidence": f"Best text-cluster silhouette score = {text_best_silhouette:.4f}.",
            "implication": "Weak separation suggests title-only text clusters may be informative for interpretation but noisy for forecasting.",
        },
        {
            "theme": "cluster_imbalance",
            "evidence": imbalance_summary,
            "implication": "Small clusters create sparse annual series, which can reduce segmented forecast stability.",
        },
        {
            "theme": "omitted_business_drivers",
            "evidence": "No external R&D, market, or regulatory variables are included.",
            "implication": "Forecasts rely entirely on internal patent history and title-derived signals.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full Task 3 segmentation-based forecasting pipeline."""
    ensure_directory(str(TABLES_DIR))
    ensure_directory(str(PROCESSED_DIR))

    print("Loading cleaned patent data...")
    df_clean = load_dataset(str(CLEAN_DATA_PATH))
    annual_totals = create_annual_patent_counts(df_clean)
    save_dataframe(annual_totals, str(TABLES_DIR / "task3_annual_total_patents.csv"))

    train_segmentation_df = df_clean[df_clean["year"] <= TRAIN_END_YEAR].copy()
    model_hyperparameters_by_name = _load_task2_hyperparameters()

    print("Building leakage-safe text segmentation...")
    text_df, text_k_selection, text_profiles, text_sensitivity = build_text_segmentation(
        train_df=train_segmentation_df,
        full_df=df_clean,
        random_state=RANDOM_STATE,
        k_values=get_task3_candidate_k_values(),
        text_grid=get_text_segmentation_grid(),
    )
    text_annual_counts = create_yearly_cluster_counts(
        text_df,
        cluster_column="text_cluster",
        start_year=int(annual_totals["year"].min()),
        end_year=int(annual_totals["year"].max()),
    )
    text_balance = create_cluster_balance_table(
        text_df,
        scheme_name="text_embedding_kmeans",
        cluster_column="text_cluster",
    )
    save_dataframe(text_k_selection, str(TABLES_DIR / "task3_text_k_selection.csv"))
    save_dataframe(text_profiles, str(TABLES_DIR / "task3_text_cluster_profiles.csv"))
    save_dataframe(text_annual_counts, str(TABLES_DIR / "task3_text_cluster_annual_counts.csv"))
    save_dataframe(text_sensitivity, str(TABLES_DIR / "task3_text_segmentation_sensitivity.csv"))

    print("Building leakage-safe non-text segmentation...")
    non_text_df, non_text_k_selection, non_text_profiles = build_non_text_segmentation(
        train_df=train_segmentation_df,
        full_df=df_clean,
        random_state=RANDOM_STATE,
        k_values=get_task3_candidate_k_values(),
    )
    non_text_annual_counts = create_yearly_cluster_counts(
        non_text_df,
        cluster_column="non_text_cluster",
        start_year=int(annual_totals["year"].min()),
        end_year=int(annual_totals["year"].max()),
    )
    non_text_balance = create_cluster_balance_table(
        non_text_df,
        scheme_name="non_text_kmeans",
        cluster_column="non_text_cluster",
    )
    save_dataframe(non_text_k_selection, str(TABLES_DIR / "task3_non_text_k_selection.csv"))
    save_dataframe(non_text_profiles, str(TABLES_DIR / "task3_non_text_cluster_profiles.csv"))
    save_dataframe(non_text_annual_counts, str(TABLES_DIR / "task3_non_text_cluster_annual_counts.csv"))

    cluster_balance_df = pd.concat([text_balance, non_text_balance], ignore_index=True)
    save_dataframe(cluster_balance_df, str(TABLES_DIR / "task3_cluster_balance.csv"))

    print("Running segmented cluster-level forecasting for both schemes...")
    text_metrics, text_predictions, text_future = run_segmented_forecast_for_scheme(
        annual_cluster_counts=text_annual_counts,
        annual_total_counts=annual_totals,
        scheme_name="text_embedding_kmeans",
        model_names=TASK3_MODELS,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        future_horizon=FUTURE_HORIZON,
        clip_non_negative=CLIP_NON_NEGATIVE,
        random_state=RANDOM_STATE,
        model_hyperparameters_by_name=model_hyperparameters_by_name,
    )
    non_text_metrics, non_text_predictions, non_text_future = run_segmented_forecast_for_scheme(
        annual_cluster_counts=non_text_annual_counts,
        annual_total_counts=annual_totals,
        scheme_name="non_text_kmeans",
        model_names=TASK3_MODELS,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        future_horizon=FUTURE_HORIZON,
        clip_non_negative=CLIP_NON_NEGATIVE,
        random_state=RANDOM_STATE,
        model_hyperparameters_by_name=model_hyperparameters_by_name,
    )

    task3_metrics_detailed = pd.concat([text_metrics, non_text_metrics], ignore_index=True)
    task3_predictions = pd.concat([text_predictions, non_text_predictions], ignore_index=True)
    task3_future = pd.concat([text_future, non_text_future], ignore_index=True)

    save_dataframe(task3_metrics_detailed, str(TABLES_DIR / "task3_segmented_metrics_detailed.csv"))
    save_dataframe(task3_predictions, str(TABLES_DIR / "task3_segmented_aggregate_predictions.csv"))
    save_dataframe(task3_future, str(TABLES_DIR / "task3_segmented_future_forecasts.csv"))

    print("Building segmented vs baseline comparison tables...")
    task2_metrics = load_dataset(str(TASK2_METRICS_PATH))
    baseline_metrics = build_task3_baseline_comparison(task2_metrics)

    aggregate_segmented_metrics = task3_metrics_detailed.copy()
    if "cluster" in aggregate_segmented_metrics.columns:
        aggregate_segmented_metrics = aggregate_segmented_metrics[aggregate_segmented_metrics["cluster"].isna()].copy()
    aggregate_segmented_metrics = aggregate_segmented_metrics[
        ["scheme", "model", "split", "MAE", "RMSE", "MAPE", "n_clusters"]
    ].copy()

    baseline_vs_segmented = pd.concat([baseline_metrics, aggregate_segmented_metrics], ignore_index=True)
    baseline_vs_segmented = baseline_vs_segmented.sort_values(
        by=["split", "RMSE", "scheme", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    save_dataframe(baseline_vs_segmented, str(TABLES_DIR / "task3_baseline_vs_segmented_metrics.csv"))

    print("Building future forecast comparison table...")
    baseline_future_rows = []
    linear_future = load_dataset(str(TASK2_LINEAR_FUTURE_PATH))
    for _, row in linear_future.iterrows():
        baseline_future_rows.append(
            {
                "scheme": "no_segmentation",
                "model": "Linear Regression",
                "year": int(row["year"]),
                "predicted_total_patents": float(row["total_patents"]),
            }
        )

    rf_future = load_dataset(str(TASK2_RF_FUTURE_PATH))
    for _, row in rf_future.iterrows():
        baseline_future_rows.append(
            {
                "scheme": "no_segmentation",
                "model": "Random Forest",
                "year": int(row["year"]),
                "predicted_total_patents": float(row["total_patents"]),
            }
        )

    segmented_aggregate_future = task3_future[task3_future["cluster"] == "aggregated_total"].copy()
    segmented_aggregate_future = segmented_aggregate_future.rename(
        columns={"predicted_cluster_patents": "predicted_total_patents"}
    )[["scheme", "model", "year", "predicted_total_patents"]]

    future_comparison = pd.concat(
        [pd.DataFrame(baseline_future_rows), segmented_aggregate_future],
        ignore_index=True,
    ).sort_values(by=["year", "scheme", "model"]).reset_index(drop=True)
    save_dataframe(future_comparison, str(TABLES_DIR / "task3_future_baseline_vs_segmented.csv"))

    limitations_df = _build_limitations_summary(
        annual_totals=annual_totals,
        text_k_selection=text_k_selection,
        non_text_k_selection=non_text_k_selection,
        cluster_balance_df=cluster_balance_df,
    )
    save_dataframe(limitations_df, str(TABLES_DIR / "limitations_summary.csv"))

    print("----- TASK 3 BASELINE VS SEGMENTED (AGGREGATED) -----")
    print(baseline_vs_segmented.to_string(index=False))
    print("----- TASK 3 CLUSTER BALANCE -----")
    print(cluster_balance_df.to_string(index=False))
    print("Task 3 segmentation-based forecasting pipeline completed successfully.")
    print(f"Outputs saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
