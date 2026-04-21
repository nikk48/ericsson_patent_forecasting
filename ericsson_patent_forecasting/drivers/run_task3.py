"""
run_task3.py

Driver script for Task 3 segmentation-based forecasting.

Task 3 requirements covered here:
- two segmentation schemes
- text embeddings from patent titles
- cluster count selection with silhouette scores
- cluster interpretation tables
- cluster-level yearly forecasts
- aggregation back to total yearly forecasts
- comparison against non-segmented baseline metrics
"""

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

from src.utils import ensure_directory
from src.data_loader import load_dataset, save_dataframe
from src.feature_engineering import create_annual_patent_counts
from src.segmentation_forecasting import (
    build_non_text_segmentation,
    build_task3_baseline_comparison,
    build_text_segmentation,
    create_yearly_cluster_counts,
    run_segmented_forecast_for_scheme,
)


CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_patents.csv"
TASK2_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "task2_model_metrics.csv"
TASK2_SELECTED_FUTURE_PATH = PROJECT_ROOT / "outputs" / "tables" / "task2_future_forecasts.csv"
TASK2_RF_FUTURE_PATH = PROJECT_ROOT / "outputs" / "tables" / "random_forest_future_forecasts.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_END_YEAR = 2010
VAL_END_YEAR = 2016
FUTURE_HORIZON = 5
TASK3_MODELS = ["Linear Regression", "Random Forest"]


def main() -> None:
    """Run the full Task 3 segmentation-based forecasting pipeline."""
    ensure_directory(str(TABLES_DIR))
    ensure_directory(str(PROCESSED_DIR))

    print("Loading cleaned patent data...")
    df_clean = load_dataset(str(CLEAN_DATA_PATH))
    annual_totals = create_annual_patent_counts(df_clean)
    save_dataframe(annual_totals, str(TABLES_DIR / "task3_annual_total_patents.csv"))

    # -----------------------------
    # Segmentation scheme 1: text embedding clusters
    # -----------------------------
    print("Building text-embedding segmentation (TF-IDF + SVD + KMeans)...")
    text_df, text_k_selection, text_profiles = build_text_segmentation(df_clean)
    text_annual_counts = create_yearly_cluster_counts(
        text_df,
        cluster_column="text_cluster",
        start_year=int(annual_totals["year"].min()),
        end_year=int(annual_totals["year"].max()),
    )
    save_dataframe(text_k_selection, str(TABLES_DIR / "task3_text_k_selection.csv"))
    save_dataframe(text_profiles, str(TABLES_DIR / "task3_text_cluster_profiles.csv"))
    save_dataframe(text_annual_counts, str(TABLES_DIR / "task3_text_cluster_annual_counts.csv"))

    # -----------------------------
    # Segmentation scheme 2: non-text feature clusters
    # -----------------------------
    print("Building non-text segmentation (numeric/categorical feature KMeans)...")
    non_text_df, non_text_k_selection, non_text_profiles = build_non_text_segmentation(df_clean)
    non_text_annual_counts = create_yearly_cluster_counts(
        non_text_df,
        cluster_column="non_text_cluster",
        start_year=int(annual_totals["year"].min()),
        end_year=int(annual_totals["year"].max()),
    )
    save_dataframe(non_text_k_selection, str(TABLES_DIR / "task3_non_text_k_selection.csv"))
    save_dataframe(non_text_profiles, str(TABLES_DIR / "task3_non_text_cluster_profiles.csv"))
    save_dataframe(non_text_annual_counts, str(TABLES_DIR / "task3_non_text_cluster_annual_counts.csv"))

    # -----------------------------
    # Cluster-level forecasting and aggregation
    # -----------------------------
    print("Running segmented cluster-level forecasting for both schemes...")
    text_metrics, text_predictions, text_future = run_segmented_forecast_for_scheme(
        annual_cluster_counts=text_annual_counts,
        annual_total_counts=annual_totals,
        scheme_name="text_embedding_kmeans",
        model_names=TASK3_MODELS,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        future_horizon=FUTURE_HORIZON,
    )
    non_text_metrics, non_text_predictions, non_text_future = run_segmented_forecast_for_scheme(
        annual_cluster_counts=non_text_annual_counts,
        annual_total_counts=annual_totals,
        scheme_name="non_text_kmeans",
        model_names=TASK3_MODELS,
        train_end_year=TRAIN_END_YEAR,
        val_end_year=VAL_END_YEAR,
        future_horizon=FUTURE_HORIZON,
    )

    task3_metrics_detailed = pd.concat([text_metrics, non_text_metrics], ignore_index=True)
    task3_predictions = pd.concat([text_predictions, non_text_predictions], ignore_index=True)
    task3_future = pd.concat([text_future, non_text_future], ignore_index=True)

    save_dataframe(task3_metrics_detailed, str(TABLES_DIR / "task3_segmented_metrics_detailed.csv"))
    save_dataframe(task3_predictions, str(TABLES_DIR / "task3_segmented_aggregate_predictions.csv"))
    save_dataframe(task3_future, str(TABLES_DIR / "task3_segmented_future_forecasts.csv"))

    # -----------------------------
    # Compare against non-segmented baseline metrics
    # -----------------------------
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

    # -----------------------------
    # Future forecast comparison table
    # -----------------------------
    baseline_future_rows = []
    selected_future = load_dataset(str(TASK2_SELECTED_FUTURE_PATH))
    for _, row in selected_future.iterrows():
        baseline_future_rows.append(
            {
                "scheme": "no_segmentation",
                "model": row["selected_model"],
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

    # -----------------------------
    # Console summary
    # -----------------------------
    print("----- TASK 3 BASELINE VS SEGMENTED (AGGREGATED) -----")
    print(baseline_vs_segmented.to_string(index=False))

    print("----- TASK 3 K SELECTION (TEXT) -----")
    print(text_k_selection.to_string(index=False))

    print("----- TASK 3 K SELECTION (NON-TEXT) -----")
    print(non_text_k_selection.to_string(index=False))

    print("Task 3 segmentation-based forecasting pipeline completed successfully.")
    print(f"Outputs saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
