"""
segmentation_forecasting.py

Utilities for Task 3 segmentation-based forecasting.
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
    train_model_by_name,
)


RANDOM_STATE = 42
MAX_ABS_NUMERIC_VALUE = 1e6


@contextmanager
def _suppress_known_sklearn_runtime_warnings():
    """Suppress noisy numerical warnings around linear algebra kernels."""
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
    """Ensure matrices used in clustering are finite and bounded."""
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
    """Select the number of clusters using average silhouette score."""
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
    train_df: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
    random_state: int = RANDOM_STATE,
    k_values: Optional[Iterable[int]] = None,
    text_grid: Optional[List[Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the text segmentation on the training window only, then assign all rows.
    """
    training_df = train_df.copy()
    working_df = train_df.copy() if full_df is None else full_df.copy()
    k_values = list(range(2, 7)) if k_values is None else list(k_values)
    text_grid = text_grid or [
        {"min_df": 5, "max_features": 3000, "svd_components": 50},
    ]

    best_artifacts = None
    sensitivity_rows = []

    for option in text_grid:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=option["min_df"],
            max_features=option["max_features"],
        )
        train_tfidf = vectorizer.fit_transform(training_df["patent_title"].fillna("").astype(str))
        if train_tfidf.shape[1] < 2:
            continue

        n_components = max(2, min(option["svd_components"], train_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        with _suppress_known_sklearn_runtime_warnings():
            train_reduced = svd.fit_transform(train_tfidf)
        train_reduced = _sanitize_numeric_matrix(train_reduced)

        best_k, selection_df = select_optimal_kmeans(
            train_reduced,
            k_values=k_values,
            random_state=random_state,
        )
        best_score = float(selection_df.iloc[0]["silhouette_score"])
        explained_variance = round(float(svd.explained_variance_ratio_.sum()), 4)
        sensitivity_rows.append(
            {
                "min_df": int(option["min_df"]),
                "max_features": int(option["max_features"]),
                "svd_components": int(n_components),
                "selected_k": int(best_k),
                "best_silhouette_score": round(best_score, 4),
                "svd_explained_variance": explained_variance,
            }
        )

        if best_artifacts is None or best_score > best_artifacts["best_score"]:
            best_artifacts = {
                "vectorizer": vectorizer,
                "svd": svd,
                "train_tfidf": train_tfidf,
                "train_reduced": train_reduced,
                "selection_df": selection_df,
                "best_k": best_k,
                "best_score": best_score,
                "selected_option": {
                    "min_df": int(option["min_df"]),
                    "max_features": int(option["max_features"]),
                    "svd_components": int(n_components),
                    "svd_explained_variance": explained_variance,
                },
            }

    if best_artifacts is None:
        raise ValueError("Unable to fit text segmentation because TF-IDF features were empty.")

    final_model = KMeans(
        n_clusters=best_artifacts["best_k"],
        random_state=random_state,
        n_init=20,
    )
    with _suppress_known_sklearn_runtime_warnings():
        final_model.fit(best_artifacts["train_reduced"])

    full_tfidf = best_artifacts["vectorizer"].transform(working_df["patent_title"].fillna("").astype(str))
    with _suppress_known_sklearn_runtime_warnings():
        full_reduced = best_artifacts["svd"].transform(full_tfidf)
    full_reduced = _sanitize_numeric_matrix(full_reduced)
    working_df["text_cluster"] = final_model.predict(full_reduced)

    profile_df = _create_text_cluster_profiles(
        df=working_df,
        cluster_column="text_cluster",
        tfidf_matrix=full_tfidf,
        feature_names=best_artifacts["vectorizer"].get_feature_names_out().tolist(),
    )

    selection_df = best_artifacts["selection_df"].copy()
    selection_df["selected_k"] = best_artifacts["best_k"]
    selection_df["min_df"] = best_artifacts["selected_option"]["min_df"]
    selection_df["max_features"] = best_artifacts["selected_option"]["max_features"]
    selection_df["svd_components"] = best_artifacts["selected_option"]["svd_components"]
    selection_df["svd_explained_variance"] = best_artifacts["selected_option"]["svd_explained_variance"]
    selection_df["fit_end_year"] = int(training_df["year"].max())

    sensitivity_df = (
        pd.DataFrame(sensitivity_rows)
        .sort_values(
            by=["best_silhouette_score", "svd_explained_variance"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return working_df, selection_df, profile_df, sensitivity_df


def build_non_text_segmentation(
    train_df: pd.DataFrame,
    full_df: Optional[pd.DataFrame] = None,
    random_state: int = RANDOM_STATE,
    k_values: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the non-text segmentation on the training window only, then assign all rows.
    """
    training_df = train_df.copy()
    working_df = train_df.copy() if full_df is None else full_df.copy()
    k_values = list(range(2, 7)) if k_values is None else list(k_values)

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
    scaler = StandardScaler()
    train_X = training_df[feature_columns].copy().fillna(0)
    full_X = working_df[feature_columns].copy().fillna(0)

    train_X_scaled = scaler.fit_transform(train_X)
    full_X_scaled = scaler.transform(full_X)
    train_X_scaled = _sanitize_numeric_matrix(train_X_scaled)
    full_X_scaled = _sanitize_numeric_matrix(full_X_scaled)

    best_k, selection_df = select_optimal_kmeans(
        train_X_scaled,
        k_values=k_values,
        random_state=random_state,
    )
    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    with _suppress_known_sklearn_runtime_warnings():
        final_model.fit(train_X_scaled)
    working_df["non_text_cluster"] = final_model.predict(full_X_scaled)

    profile_df = _create_non_text_cluster_profiles(
        df=working_df,
        cluster_column="non_text_cluster",
    )
    selection_df["selected_k"] = best_k
    selection_df["fit_end_year"] = int(training_df["year"].max())
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


def create_cluster_balance_table(df: pd.DataFrame, scheme_name: str, cluster_column: str) -> pd.DataFrame:
    """Summarize cluster sizes and shares for report-ready diagnostics."""
    cluster_counts = (
        df[cluster_column]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_patents")
    )
    cluster_counts["scheme"] = scheme_name
    cluster_counts["cluster"] = cluster_counts["cluster"].apply(lambda value: f"cluster_{int(value)}")
    cluster_counts["portfolio_share"] = (cluster_counts["n_patents"] / len(df)).round(4)
    return cluster_counts[["scheme", "cluster", "n_patents", "portfolio_share"]]


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


def forecast_cluster_future_years_recursive(
    model,
    annual_df: pd.DataFrame,
    target_column: str,
    future_horizon: int,
    clip_non_negative: bool = False,
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
        if clip_non_negative:
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
    clip_non_negative: bool = False,
    random_state: int = RANDOM_STATE,
    model_hyperparameters_by_name: Optional[Dict[str, Dict]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forecast cluster-level yearly counts, aggregate them, and return metrics,
    predictions, and future forecasts for a given segmentation scheme.
    """
    cluster_columns = [col for col in annual_cluster_counts.columns if col != "year"]
    feature_columns = get_cluster_forecasting_feature_columns()
    model_hyperparameters_by_name = model_hyperparameters_by_name or {}

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
                    "year": list(
                        range(
                            int(annual_cluster_counts["year"].max()) + 1,
                            int(annual_cluster_counts["year"].max()) + future_horizon + 1,
                        )
                    ),
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
            hyperparameters = model_hyperparameters_by_name.get(model_name)
            validation_model = train_model_by_name(
                model_name=model_name,
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                random_state=random_state,
                hyperparameters=hyperparameters,
            )
            val_pred, val_metrics = evaluate_model_on_split(
                validation_model,
                splits["X_val"],
                splits["y_val"],
                clip_non_negative=clip_non_negative,
            )
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

            test_model = train_model_by_name(
                model_name=model_name,
                X_train=train_val_df[feature_columns],
                y_train=train_val_df[cluster_column],
                random_state=random_state,
                hyperparameters=hyperparameters,
            )
            test_pred, test_metrics = evaluate_model_on_split(
                test_model,
                splits["X_test"],
                splits["y_test"],
                clip_non_negative=clip_non_negative,
            )
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

            future_model = train_model_by_name(
                model_name=model_name,
                X_train=modelling_df[feature_columns],
                y_train=modelling_df[cluster_column],
                random_state=random_state,
                hyperparameters=hyperparameters,
            )
            future_forecasts = forecast_cluster_future_years_recursive(
                model=future_model,
                annual_df=cluster_annual_df,
                target_column=cluster_column,
                future_horizon=future_horizon,
                clip_non_negative=clip_non_negative,
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
    baseline_df = baseline_df[baseline_df["split"].isin(["validation", "test"])].copy()
    baseline_df["scheme"] = "no_segmentation"
    baseline_df["n_clusters"] = 1
    ordered_columns = ["scheme", "model", "split", "MAE", "RMSE", "MAPE", "R2", "n_clusters"]
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
