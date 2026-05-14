"""
time_series_diagnostics.py

Diagnostics for annual patent-count time series and forecasting inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, adfuller, pacf


def _prepare_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values and drop missing rows."""
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def _default_nlags(series_length: int) -> int:
    """Pick a conservative number of lags for a short annual time series."""
    return max(1, min(10, (series_length // 2) - 1))


def build_adf_results(series: pd.Series, series_name: str) -> pd.DataFrame:
    """Run ADF on the level series and its first difference."""
    rows = []
    candidate_series = {
        "level": _prepare_numeric_series(series),
        "first_difference": _prepare_numeric_series(series.diff()),
    }

    for transform_name, values in candidate_series.items():
        if len(values) < 4:
            continue
        test_statistic, p_value, used_lag, n_obs, critical_values, _ = adfuller(
            values,
            autolag="AIC",
        )
        rows.append(
            {
                "series": series_name,
                "transform": transform_name,
                "n_observations": int(n_obs),
                "used_lag": int(used_lag),
                "test_statistic": round(float(test_statistic), 4),
                "p_value": round(float(p_value), 6),
                "critical_value_1pct": round(float(critical_values["1%"]), 4),
                "critical_value_5pct": round(float(critical_values["5%"]), 4),
                "critical_value_10pct": round(float(critical_values["10%"]), 4),
                "stationary_at_5pct": bool(p_value < 0.05),
            }
        )

    return pd.DataFrame(rows)


def build_vif_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for each numeric predictor."""
    numeric_df = feature_df.apply(pd.to_numeric, errors="coerce").dropna().copy()
    rows = []
    for column in numeric_df.columns:
        if numeric_df[column].nunique(dropna=True) <= 1:
            vif_value = np.inf
        else:
            try:
                vif_value = variance_inflation_factor(
                    numeric_df.to_numpy(dtype=float),
                    numeric_df.columns.get_loc(column),
                )
            except Exception:
                vif_value = np.inf
        rows.append(
            {
                "feature": column,
                "VIF": round(float(vif_value), 4) if np.isfinite(vif_value) else np.inf,
                "tolerance": round(float(1.0 / vif_value), 6) if np.isfinite(vif_value) and vif_value != 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(by="VIF", ascending=False).reset_index(drop=True)


def build_acf_table(series: pd.Series, nlags: Optional[int] = None) -> pd.DataFrame:
    """Return ACF values with confidence intervals."""
    values = _prepare_numeric_series(series)
    lag_count = _default_nlags(len(values)) if nlags is None else nlags
    acf_values, confint = acf(values, nlags=lag_count, alpha=0.05)
    rows = []
    for lag, corr_value in enumerate(acf_values):
        significant = lag > 0 and (confint[lag][0] > 0 or confint[lag][1] < 0)
        rows.append(
            {
                "lag": lag,
                "correlation": round(float(corr_value), 6),
                "ci_lower": round(float(confint[lag][0]), 6),
                "ci_upper": round(float(confint[lag][1]), 6),
                "significant_at_95pct": bool(significant),
            }
        )
    return pd.DataFrame(rows)


def build_pacf_table(series: pd.Series, nlags: Optional[int] = None) -> pd.DataFrame:
    """Return PACF values with confidence intervals."""
    values = _prepare_numeric_series(series)
    lag_count = _default_nlags(len(values)) if nlags is None else nlags
    pacf_values, confint = pacf(values, nlags=lag_count, alpha=0.05, method="ywm")
    rows = []
    for lag, corr_value in enumerate(pacf_values):
        significant = lag > 0 and (confint[lag][0] > 0 or confint[lag][1] < 0)
        rows.append(
            {
                "lag": lag,
                "correlation": round(float(corr_value), 6),
                "ci_lower": round(float(confint[lag][0]), 6),
                "ci_upper": round(float(confint[lag][1]), 6),
                "significant_at_95pct": bool(significant),
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_lags(
    correlation_df: pd.DataFrame,
    title: str,
    y_label: str,
    output_path: Optional[str] = None,
) -> None:
    """Plot lag correlations with 95% confidence intervals."""
    plt.figure(figsize=(10, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.vlines(
        correlation_df["lag"],
        0,
        correlation_df["correlation"],
        color="#1f77b4",
        linewidth=2,
    )
    plt.scatter(
        correlation_df["lag"],
        correlation_df["correlation"],
        color="#1f77b4",
        s=35,
        zorder=3,
    )
    plt.fill_between(
        correlation_df["lag"],
        correlation_df["ci_lower"],
        correlation_df["ci_upper"],
        color="#1f77b4",
        alpha=0.15,
    )
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel(y_label)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
