"""
feature_engineering.py

This file creates new variables and summary tables from the cleaned patent data.

Why this file exists:
Raw patent-level data is useful, but many analyses need transformed versions of it.

Examples:
- Task 1 needs yearly totals and yearly keyword trends
- Task 2 needs lag variables, trend variables, and keyword shares

In simple words:
This file turns raw information into analysis-ready features.
"""

from typing import List, Optional
import pandas as pd


def create_annual_patent_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create yearly total patent counts."""
    annual_counts = (
        df.groupby("year")
        .size()
        .reset_index(name="total_patents")
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual_counts


def create_keyword_trends(df: pd.DataFrame, keyword_columns: List[str]) -> pd.DataFrame:
    """
    Create yearly counts for each technology keyword.

    Because each keyword column is 0/1, the yearly sum tells us how many patents
    in that year belonged to that technology area.
    """
    existing_columns = [col for col in keyword_columns if col in df.columns]

    keyword_trends = (
        df.groupby("year")[existing_columns]
        .sum()
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return keyword_trends


def create_keyword_shares(
    annual_counts: pd.DataFrame,
    keyword_trends: pd.DataFrame,
    keyword_columns: List[str],
) -> pd.DataFrame:
    """
    Convert yearly keyword counts into proportions of total patents.

    Example:
    If Ericsson filed 100 patents in a year and 20 were AI-related,
    then AI share = 20 / 100 = 0.20.
    """
    merged = annual_counts.merge(keyword_trends, on="year", how="left")

    for col in keyword_columns:
        if col in merged.columns:
            merged[f"{col}_share"] = merged[col] / merged["total_patents"]

    share_columns = ["year"] + [f"{col}_share" for col in keyword_columns if f"{col}_share" in merged.columns]
    return merged[share_columns]


def create_patent_type_trends(df: pd.DataFrame, patent_type_columns: List[str]) -> pd.DataFrame:
    """Create yearly counts for patent types."""
    existing_columns = [col for col in patent_type_columns if col in df.columns]

    patent_type_trends = (
        df.groupby("year")[existing_columns]
        .sum()
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return patent_type_trends


def create_segment_trends(
    df: pd.DataFrame,
    segment_column: str,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create yearly counts for a categorical segmentation variable.

    Examples:
    - patent_type
    - tech_era
    """
    if segment_column not in df.columns:
        return pd.DataFrame()

    segment_df = df[["year", segment_column]].copy()
    segment_df[segment_column] = segment_df[segment_column].fillna("missing").astype(str)

    if top_n is not None:
        top_segments = (
            segment_df[segment_column]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
        segment_df[segment_column] = segment_df[segment_column].where(
            segment_df[segment_column].isin(top_segments),
            "other",
        )

    trends = (
        segment_df.groupby(["year", segment_column])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return trends


def create_segment_shares(segment_trends: pd.DataFrame) -> pd.DataFrame:
    """Convert segment counts into yearly shares."""
    if segment_trends.empty:
        return pd.DataFrame()

    share_df = segment_trends.copy()
    segment_columns = [col for col in share_df.columns if col != "year"]
    totals = share_df[segment_columns].sum(axis=1)

    for col in segment_columns:
        share_df[col] = share_df[col] / totals

    return share_df


def create_growth_rate(annual_counts: pd.DataFrame) -> pd.DataFrame:
    """Add annual patent growth rate."""
    growth_df = annual_counts.copy()
    growth_df["growth_rate"] = growth_df["total_patents"].pct_change()
    return growth_df


def create_title_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create yearly averages of available title-related features."""
    possible_columns = [
        "title_len_chars",
        "title_len_words",
        "title_has_number",
        "title_has_acronym",
        "keyword_score",
    ]
    existing_columns = [col for col in possible_columns if col in df.columns]

    if not existing_columns:
        return pd.DataFrame()

    title_summary = (
        df.groupby("year")[existing_columns]
        .mean()
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return title_summary


def create_feature_justification_table(
    annual_counts: pd.DataFrame,
    keyword_shares: pd.DataFrame,
    patent_type_trends: pd.DataFrame,
    title_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an annual feature evidence table for Task 1.

    The goal is to help justify later forecasting inputs using descriptive
    relationships observed during EDA.
    """
    annual_features = annual_counts.copy()

    if not keyword_shares.empty:
        annual_features = annual_features.merge(keyword_shares, on="year", how="left")

    if not patent_type_trends.empty:
        patent_type_share_df = patent_type_trends.copy()
        type_columns = [col for col in patent_type_share_df.columns if col != "year"]
        totals = patent_type_share_df[type_columns].sum(axis=1)
        for col in type_columns:
            patent_type_share_df[f"{col}_share"] = patent_type_share_df[col] / totals
        patent_type_share_df = patent_type_share_df[
            ["year"] + [f"{col}_share" for col in type_columns]
        ]
        annual_features = annual_features.merge(patent_type_share_df, on="year", how="left")

    if not title_summary.empty:
        annual_features = annual_features.merge(title_summary, on="year", how="left")

    candidate_columns = [col for col in annual_features.columns if col not in {"year", "total_patents"}]
    rows = []

    for col in candidate_columns:
        series = annual_features[col]
        rows.append(
            {
                "feature": col,
                "feature_group": _infer_feature_group(col),
                "missing_count": int(series.isna().sum()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "correlation_with_total_patents": round(float(series.corr(annual_features["total_patents"])), 4),
                "abs_correlation_with_total_patents": round(
                    float(abs(series.corr(annual_features["total_patents"]))), 4
                ),
            }
        )

    feature_table = (
        pd.DataFrame(rows)
        .sort_values(
            by=["abs_correlation_with_total_patents", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    return feature_table


def _infer_feature_group(feature_name: str) -> str:
    """Group feature names into broad categories for easier interpretation."""
    if feature_name.startswith("kw_"):
        return "keyword_share"
    if feature_name.startswith("is_"):
        return "patent_type_share"
    if feature_name.startswith("title_") or feature_name == "keyword_score":
        return "title_feature"
    return "other"


def create_annual_modelling_dataset(
    df: pd.DataFrame,
    keyword_columns: List[str]
) -> pd.DataFrame:
    """
    Build the annual modelling dataset used in Task 2 forecasting.

    This creates:
    - total patents per year
    - keyword shares per year
    - trend variable
    - lag_1 and lag_2
    - growth rate
    - lagged predictors for leakage-free modelling

    Important principle:
    When predicting year t, the model should only use information available
    up to year t-1. Therefore, forecasting features are stored in lagged form.
    """
    annual_counts = (
        df.groupby("year")
        .size()
        .reset_index(name="total_patents")
        .sort_values("year")
        .reset_index(drop=True)
    )

    existing_keywords = [col for col in keyword_columns if col in df.columns]
    keyword_counts = (
        df.groupby("year")[existing_keywords]
        .sum()
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )

    annual_df = annual_counts.merge(keyword_counts, on="year", how="left")

    for col in existing_keywords:
        annual_df[f"{col}_share"] = annual_df[col] / annual_df["total_patents"]

    annual_df["trend"] = range(1, len(annual_df) + 1)
    annual_df["lag_1"] = annual_df["total_patents"].shift(1)
    annual_df["lag_2"] = annual_df["total_patents"].shift(2)
    annual_df["growth_rate"] = annual_df["total_patents"].pct_change()
    annual_df["growth_rate_lag_1"] = annual_df["growth_rate"].shift(1)

    for col in existing_keywords:
        annual_df[f"{col}_share_lag_1"] = annual_df[f"{col}_share"].shift(1)

    return annual_df


def get_forecasting_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the default leakage-free feature columns for Task 2 forecasting."""
    preferred_features = [
        "trend",
        "lag_1",
        "lag_2",
        "growth_rate_lag_1",
        "kw_5g_share_lag_1",
        "kw_ai_ml_share_lag_1",
        "kw_cloud_edge_share_lag_1",
        "kw_security_share_lag_1",
        "kw_iot_share_lag_1",
        "kw_network_share_lag_1",
        "kw_energy_share_lag_1",
        "kw_antenna_share_lag_1",
        "kw_data_share_lag_1",
    ]

    return [col for col in preferred_features if col in df.columns]


def build_feature_ablation_sets(df: pd.DataFrame) -> List[dict]:
    """Create progressively richer feature sets for Task 2 ablation tests."""
    candidate_sets = [
        {
            "feature_set": "trend_only",
            "description": "Trend only",
            "features": ["trend"],
        },
        {
            "feature_set": "trend_plus_lags",
            "description": "Trend plus lag_1 and lag_2",
            "features": ["trend", "lag_1", "lag_2"],
        },
        {
            "feature_set": "trend_lags_growth",
            "description": "Trend plus lag features and lagged growth",
            "features": ["trend", "lag_1", "lag_2", "growth_rate_lag_1"],
        },
        {
            "feature_set": "full_feature_set",
            "description": "Trend, lags, lagged growth, and lagged keyword shares",
            "features": get_forecasting_feature_columns(df),
        },
    ]

    cleaned_sets = []
    for feature_set in candidate_sets:
        existing_features = [feature for feature in feature_set["features"] if feature in df.columns]
        cleaned_sets.append(
            {
                "feature_set": feature_set["feature_set"],
                "description": feature_set["description"],
                "features": existing_features,
            }
        )
    return cleaned_sets


def create_forecasting_leakage_check_table(feature_columns: List[str]) -> pd.DataFrame:
    """Document why each forecasting feature is safe or unsafe for real forecasting."""
    rows = []
    for feature in feature_columns:
        is_safe = feature.startswith("lag_") or feature.endswith("_lag_1") or feature == "trend"
        rows.append(
            {
                "feature": feature,
                "is_leakage_safe": bool(is_safe),
                "reason": _get_leakage_reason(feature),
            }
        )
    return pd.DataFrame(rows)


def _get_leakage_reason(feature_name: str) -> str:
    """Explain why a feature is safe or unsafe for one-step-ahead forecasting."""
    if feature_name == "trend":
        return "Calendar trend is known before the forecast year."
    if feature_name.startswith("lag_"):
        return "Lagged totals depend only on historical patent counts."
    if feature_name.endswith("_lag_1"):
        return "Lagged engineered feature uses information available at year t-1."
    return "Current-year feature would risk using information unavailable at forecast time."
