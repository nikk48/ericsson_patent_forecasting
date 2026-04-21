"""
segmentation_forecasting.py

Utilities for Task 3 segmentation-based forecasting.

This module supports two segmentation schemes:
- text embeddings from patent titles
- non-text patent-level feature clustering

It also provides helpers to:
- justify cluster count using silhouette scores
- interpret clusters
- convert patent-level clusters into yearly cluster counts
- forecast yearly cluster counts and aggregate them back to total patents
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.forecasting_models import (
    calculate_forecast_metrics,
    evaluate_model_on_split,
    prepare_model_data,
    split_time_series_data,
    train_linear_regression,
    train_random_forest,
)


RANDOM_STATE = 42
MAX_ABS_NUMERIC_VALUE = 1e6


@contextmanager
def _suppress_known_sklearn_runtime_warnings():
    """
    Suppress noisy numeric RuntimeWarnings from sklearn linear algebra kernels.

    These warnings can appear on some Mac/Python/NumPy builds even when inputs
    are finite and outputs are valid. We contain suppression to clustering/SVD
    calls so the rest of the pipeline keeps normal warning behavior.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*encountered in matmul.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*overflow encountered.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*divide by zero encountered.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )
        yield


def _sanitize_numeric_matrix(X: np.ndarray) -> np.ndarray:
    """
    Ensure matrices used in clustering are finite and bounded.
    """
    matrix = np.asarray(X, dtype=np.float64)
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=MAX_ABS_NUMERIC_VALUE,
        neginf=-MAX_ABS_NUMERIC_VALUE,
    )
    matrix = np.clip(matrix, -MAX_ABS_NUMERIC_VALUE, MAX_ABS_NUMERIC_VALUE)
    return matrix


def select_optimal_kmeans(
    X: np.ndarray,
    k_values: Iterable[int] = range(2, 7),
    random_state: int = RANDOM_STATE,
    sample_size: int = 5000,
) -> Tuple[int, pd.DataFrame]:
    """
    Select the number of clusters using average silhouette score.

    For efficiency, silhouette scoring is computed on a sample when the dataset
    is large, while the final KMeans model is always fit on the full matrix.
    """
    X = _sanitize_numeric_matrix(X)

    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(X), size=sample_size, replace=False)
        X_eval = X[sample_indices]
    else:
        X_eval = X

    X_eval = _sanitize_numeric_matrix(X_eval)

    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        with _suppress_known_sklearn_runtime_warnings():
            labels = model.fit_predict(X_eval)
            score = silhouette_score(X_eval, labels)
        rows.append({"k": int(k), "silhouette_score": round(float(score), 4)})

    selection_df = pd.DataFrame(rows).sort_values(
        by=["silhouette_score", "k"],
        ascending=[False, True],
    ).reset_index(drop=True)
    best_k = int(selection_df.iloc[0]["k"])
    return best_k, selection_df


def build_text_segmentation(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the text-based segmentation scheme using TF-IDF + SVD + KMeans.
    """
    working_df = df.copy()
    text_series = working_df["patent_title"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=3000,
    )
    tfidf_matrix = vectorizer.fit_transform(text_series)

    n_components = max(2, min(50, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    with _suppress_known_sklearn_runtime_warnings():
        reduced_embeddings = svd.fit_transform(tfidf_matrix)
    reduced_embeddings = _sanitize_numeric_matrix(reduced_embeddings)

    best_k, selection_df = select_optimal_kmeans(
        reduced_embeddings,
        random_state=random_state,
    )
    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    with _suppress_known_sklearn_runtime_warnings():
        working_df["text_cluster"] = final_model.fit_predict(reduced_embeddings)

    profile_df = _create_text_cluster_profiles(
        df=working_df,
        cluster_column="text_cluster",
        tfidf_matrix=tfidf_matrix,
        feature_names=vectorizer.get_feature_names_out().tolist(),
    )
    selection_df["selected_k"] = best_k
    selection_df["svd_components"] = n_components
    selection_df["svd_explained_variance"] = round(float(svd.explained_variance_ratio_.sum()), 4)
    return working_df, selection_df, profile_df


def build_non_text_segmentation(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the non-text segmentation scheme using title stats, keyword flags,
    and patent-type indicators.
    """
    working_df = df.copy()
    feature_columns = [
        "title_len_chars",
        "title_len_words",
        "title_has_number",
        "title_has_acronym",
        "keyword_score",
        "kw_5g",
        "kw_ai_ml",
        "kw_cloud_edge",
        "kw_security",
        "kw_iot",
        "kw_network",
        "kw_energy",
        "kw_antenna",
        "kw_data",
        "is_utility",
        "is_design",
        "is_other_type",
    ]
    X = working_df[feature_columns].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = _sanitize_numeric_matrix(X_scaled)

    best_k, selection_df = select_optimal_kmeans(
        X_scaled,
        random_state=random_state,
    )
    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    with _suppress_known_sklearn_runtime_warnings():
        working_df["non_text_cluster"] = final_model.fit_predict(X_scaled)

    profile_df = _create_non_text_cluster_profiles(
        df=working_df,
        cluster_column="non_text_cluster",
    )
    selection_df["selected_k"] = best_k
    return working_df, selection_df, profile_df


def create_yearly_cluster_counts(
    df: pd.DataFrame,
    cluster_column: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Create a complete year-by-cluster count matrix."""
    if start_year is None:
        start_year = int(df["year"].min())
    if end_year is None:
        end_year = int(df["year"].max())

    years = list(range(start_year, end_year + 1))
    clusters = sorted(df[cluster_column].dropna().unique().tolist())

    counts = (
        df.groupby(["year", cluster_column])
        .size()
        .unstack(fill_value=0)
        .reindex(index=years, columns=clusters, fill_value=0)
        .reset_index()
        .rename(columns={"index": "year"})
    )

    renamed_columns = {
        cluster_label: f"cluster_{int(cluster_label)}"
        for cluster_label in clusters
    }
    counts = counts.rename(columns=renamed_columns)
    return counts


def create_cluster_modelling_dataset(
    annual_cluster_counts: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Build a leakage-free annual forecasting dataset for one cluster."""
    annual_df = annual_cluster_counts[["year", target_column]].copy()
    annual_df["trend"] = range(1, len(annual_df) + 1)
    annual_df["lag_1"] = annual_df[target_column].shift(1)
    annual_df["lag_2"] = annual_df[target_column].shift(2)
    growth_rate = annual_df[target_column].pct_change()
    growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    annual_df["growth_rate"] = growth_rate
    annual_df["growth_rate_lag_1"] = annual_df["growth_rate"].shift(1)
    return annual_df


def get_cluster_forecasting_feature_columns() -> List[str]:
    """Return the feature set used for cluster-level forecasting."""
    return ["trend", "lag_1", "lag_2", "growth_rate_lag_1"]


def train_cluster_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Train one of the baseline models used in segmented forecasting."""
    if model_name == "Linear Regression":
        return train_linear_regression(X_train=X_train, y_train=y_train)
    if model_name == "Random Forest":
        return train_random_forest(X_train=X_train, y_train=y_train, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported cluster forecasting model: {model_name}")


def forecast_cluster_future_years_recursive(
    model,
    annual_df: pd.DataFrame,
    target_column: str,
    future_horizon: int,
) -> pd.DataFrame:
    """Forecast future cluster counts using recursive lag updates."""
    feature_columns = get_cluster_forecasting_feature_columns()
    working_df = annual_df.copy().sort_values("year").reset_index(drop=True)

    last_year = int(working_df["year"].max())
    last_trend = int(working_df["trend"].max())
    forecasts = []

    for step in range(1, future_horizon + 1):
        lag_1 = float(working_df[target_column].iloc[-1])
        lag_2 = float(working_df[target_column].iloc[-2])
        growth_rate = np.nan if lag_2 == 0 else (lag_1 - lag_2) / lag_2
        previous_growth_rate = working_df["growth_rate"].iloc[-1]
        if pd.isna(previous_growth_rate):
            previous_growth_rate = 0.0

        row = {
            "year": last_year + step,
            "trend": last_trend + step,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "growth_rate": growth_rate,
            "growth_rate_lag_1": previous_growth_rate,
        }
        X_future = pd.DataFrame([{col: row.get(col, np.nan) for col in feature_columns}])
        predicted_value = float(model.predict(X_future)[0])
        predicted_value = max(0.0, predicted_value)

        row[target_column] = predicted_value
        forecasts.append(row)
        working_df = pd.concat([working_df, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecasts)


def run_segmented_forecast_for_scheme(
    annual_cluster_counts: pd.DataFrame,
    annual_total_counts: pd.DataFrame,
    scheme_name: str,
    model_names: List[str],
    train_end_year: int,
    val_end_year: int,
    future_horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forecast cluster-level yearly counts, aggregate them, and return metrics,
    predictions, and future forecasts for a given segmentation scheme.
    """
    cluster_columns = [col for col in annual_cluster_counts.columns if col != "year"]
    feature_columns = get_cluster_forecasting_feature_columns()

    cluster_metric_rows = []
    aggregate_prediction_rows = []
    future_rows = []

    split_years = {
        "validation": annual_cluster_counts.loc[
            (annual_cluster_counts["year"] > train_end_year) & (annual_cluster_counts["year"] <= val_end_year),
            "year",
        ].tolist(),
        "test": annual_cluster_counts.loc[annual_cluster_counts["year"] > val_end_year, "year"].tolist(),
    }

    aggregate_predictions = {
        model_name: {
            "validation": pd.DataFrame(
                {"year": split_years["validation"], "predicted_total_patents": np.zeros(len(split_years["validation"]))}
            ),
            "test": pd.DataFrame(
                {"year": split_years["test"], "predicted_total_patents": np.zeros(len(split_years["test"]))}
            ),
            "future": pd.DataFrame(
                {
                    "year": list(range(int(annual_cluster_counts["year"].max()) + 1, int(annual_cluster_counts["year"].max()) + future_horizon + 1)),
                    "predicted_total_patents": np.zeros(future_horizon),
                }
            ),
        }
        for model_name in model_names
    }

    for cluster_column in cluster_columns:
        cluster_annual_df = create_cluster_modelling_dataset(annual_cluster_counts, cluster_column)
        modelling_df = prepare_model_data(
            annual_df=cluster_annual_df,
            feature_columns=feature_columns,
            target_column=cluster_column,
        )
        splits = split_time_series_data(
            df=modelling_df,
            train_end_year=train_end_year,
            val_end_year=val_end_year,
            target_column=cluster_column,
            feature_columns=feature_columns,
        )
        train_val_df = modelling_df[modelling_df["year"] <= val_end_year].copy()

        for model_name in model_names:
            validation_model = train_cluster_model(model_name, splits["X_train"], splits["y_train"])
            val_pred, val_metrics = evaluate_model_on_split(validation_model, splits["X_val"], splits["y_val"])
            val_pred = np.clip(val_pred, 0, None)
            val_metrics = calculate_forecast_metrics(splits["y_val"], val_pred)
            cluster_metric_rows.append(
                {
                    "scheme": scheme_name,
                    "cluster": cluster_column,
                    "model": model_name,
                    "split": "validation",
                    **val_metrics,
                }
            )
            aggregate_predictions[model_name]["validation"]["predicted_total_patents"] += val_pred

            test_model = train_cluster_model(
                model_name,
                train_val_df[feature_columns],
                train_val_df[cluster_column],
            )
            test_pred, test_metrics = evaluate_model_on_split(test_model, splits["X_test"], splits["y_test"])
            test_pred = np.clip(test_pred, 0, None)
            test_metrics = calculate_forecast_metrics(splits["y_test"], test_pred)
            cluster_metric_rows.append(
                {
                    "scheme": scheme_name,
                    "cluster": cluster_column,
                    "model": model_name,
                    "split": "test",
                    **test_metrics,
                }
            )
            aggregate_predictions[model_name]["test"]["predicted_total_patents"] += test_pred

            future_model = train_cluster_model(
                model_name,
                modelling_df[feature_columns],
                modelling_df[cluster_column],
            )
            future_forecasts = forecast_cluster_future_years_recursive(
                model=future_model,
                annual_df=cluster_annual_df,
                target_column=cluster_column,
                future_horizon=future_horizon,
            )
            aggregate_predictions[model_name]["future"]["predicted_total_patents"] += future_forecasts[cluster_column].values

            for _, row in future_forecasts.iterrows():
                future_rows.append(
                    {
                        "scheme": scheme_name,
                        "model": model_name,
                        "cluster": cluster_column,
                        "year": int(row["year"]),
                        "predicted_cluster_patents": round(float(row[cluster_column]), 4),
                    }
                )

    total_lookup = annual_total_counts.set_index("year")["total_patents"]
    aggregate_metric_rows = []
    for model_name in model_names:
        for split_name in ["validation", "test"]:
            prediction_df = aggregate_predictions[model_name][split_name].copy()
            prediction_df["actual_total_patents"] = prediction_df["year"].map(total_lookup)
            metrics = calculate_forecast_metrics(
                prediction_df["actual_total_patents"],
                prediction_df["predicted_total_patents"],
            )
            aggregate_metric_rows.append(
                {
                    "scheme": scheme_name,
                    "model": model_name,
                    "split": split_name,
                    **metrics,
                    "n_clusters": int(len(cluster_columns)),
                }
            )
            for _, row in prediction_df.iterrows():
                aggregate_prediction_rows.append(
                    {
                        "scheme": scheme_name,
                        "model": model_name,
                        "split": split_name,
                        "year": int(row["year"]),
                        "actual_total_patents": float(row["actual_total_patents"]),
                        "predicted_total_patents": round(float(row["predicted_total_patents"]), 4),
                    }
                )

        for _, row in aggregate_predictions[model_name]["future"].iterrows():
            future_rows.append(
                {
                    "scheme": scheme_name,
                    "model": model_name,
                    "cluster": "aggregated_total",
                    "year": int(row["year"]),
                    "predicted_cluster_patents": round(float(row["predicted_total_patents"]), 4),
                }
            )

    metrics_df = pd.DataFrame(aggregate_metric_rows)
    cluster_metrics_df = pd.DataFrame(cluster_metric_rows)
    if not cluster_metrics_df.empty:
        metrics_df = pd.concat([metrics_df, cluster_metrics_df], ignore_index=True)

    predictions_df = pd.DataFrame(aggregate_prediction_rows)
    future_df = pd.DataFrame(future_rows)
    return metrics_df, predictions_df, future_df


def build_task3_baseline_comparison(task2_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the two baseline models used for Task 3 comparison."""
    baseline_df = task2_metrics_df[
        task2_metrics_df["model"].isin(["Linear Regression", "Random Forest"])
    ].copy()
    baseline_df["scheme"] = "no_segmentation"
    baseline_df["n_clusters"] = 1
    ordered_columns = ["scheme", "model", "split", "MAE", "RMSE", "MAPE", "n_clusters"]
    return baseline_df[ordered_columns]


def _create_text_cluster_profiles(
    df: pd.DataFrame,
    cluster_column: str,
    tfidf_matrix,
    feature_names: List[str],
) -> pd.DataFrame:
    """Create an interpretable profile table for text clusters."""
    rows = []
    keyword_columns = [
        "kw_5g",
        "kw_ai_ml",
        "kw_cloud_edge",
        "kw_security",
        "kw_iot",
        "kw_network",
        "kw_energy",
        "kw_antenna",
        "kw_data",
    ]
    patent_type_columns = ["is_utility", "is_design", "is_other_type"]

    for cluster_label in sorted(df[cluster_column].unique()):
        mask = (df[cluster_column] == cluster_label).to_numpy()
        cluster_df = df.loc[mask]
        mean_tfidf = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_term_indices = np.argsort(mean_tfidf)[::-1][:8]
        top_terms = ", ".join(feature_names[idx] for idx in top_term_indices if mean_tfidf[idx] > 0)

        keyword_means = cluster_df[keyword_columns].mean()
        patent_type_means = cluster_df[patent_type_columns].mean()

        rows.append(
            {
                "cluster": int(cluster_label),
                "n_patents": int(len(cluster_df)),
                "portfolio_share": round(float(len(cluster_df) / len(df)), 4),
                "avg_year": round(float(cluster_df["year"].mean()), 2),
                "avg_title_len_words": round(float(cluster_df["title_len_words"].mean()), 2),
                "dominant_keyword": keyword_means.idxmax(),
                "dominant_patent_type": patent_type_means.idxmax(),
                "top_terms": top_terms,
            }
        )

    return pd.DataFrame(rows).sort_values(by="cluster").reset_index(drop=True)


def _create_non_text_cluster_profiles(
    df: pd.DataFrame,
    cluster_column: str,
) -> pd.DataFrame:
    """Create an interpretable profile table for non-text clusters."""
    rows = []
    keyword_columns = [
        "kw_5g",
        "kw_ai_ml",
        "kw_cloud_edge",
        "kw_security",
        "kw_iot",
        "kw_network",
        "kw_energy",
        "kw_antenna",
        "kw_data",
    ]
    patent_type_columns = ["is_utility", "is_design", "is_other_type"]

    for cluster_label in sorted(df[cluster_column].unique()):
        cluster_df = df[df[cluster_column] == cluster_label]
        keyword_means = cluster_df[keyword_columns].mean()
        patent_type_means = cluster_df[patent_type_columns].mean()

        rows.append(
            {
                "cluster": int(cluster_label),
                "n_patents": int(len(cluster_df)),
                "portfolio_share": round(float(len(cluster_df) / len(df)), 4),
                "avg_year": round(float(cluster_df["year"].mean()), 2),
                "avg_title_len_words": round(float(cluster_df["title_len_words"].mean()), 2),
                "avg_keyword_score": round(float(cluster_df["keyword_score"].mean()), 4),
                "dominant_keyword": keyword_means.idxmax(),
                "dominant_patent_type": patent_type_means.idxmax(),
            }
        )

    return pd.DataFrame(rows).sort_values(by="cluster").reset_index(drop=True)
