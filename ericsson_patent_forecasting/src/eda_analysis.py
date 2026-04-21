"""
eda_analysis.py

This file contains the visual analysis functions used in Task 1.

In simple words:
This file creates the charts that help explain Ericsson's innovation patterns.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save_figure(output_path: Optional[str]) -> None:
    """Save the current chart if an output path is provided."""
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_patents_per_year(annual_counts: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot total patents per year."""
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=annual_counts, x="year", y="total_patents", marker="o")
    plt.title("Total Patents Filed Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Patents")
    _save_figure(output_path)
    plt.close()


def plot_keyword_trends(keyword_trends: pd.DataFrame, keyword_columns: List[str], output_path: Optional[str] = None) -> None:
    """Plot yearly counts of technology-related patents."""
    plot_columns = [col for col in keyword_columns if col in keyword_trends.columns]
    melted = keyword_trends.melt(id_vars="year", value_vars=plot_columns, var_name="keyword", value_name="count")
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=melted, x="year", y="count", hue="keyword")
    plt.title("Technology Keyword Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Patents")
    plt.legend(title="Keyword", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def plot_keyword_shares(keyword_shares: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot yearly technology shares."""
    share_columns = [col for col in keyword_shares.columns if col.endswith("_share")]
    melted = keyword_shares.melt(id_vars="year", value_vars=share_columns, var_name="keyword_share", value_name="share")
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=melted, x="year", y="share", hue="keyword_share")
    plt.title("Technology Keyword Shares Over Time")
    plt.xlabel("Year")
    plt.ylabel("Share of Total Patents")
    plt.legend(title="Keyword Share", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def plot_patent_type_trends(patent_type_trends: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot the number of patents by patent type over time."""
    type_columns = [col for col in patent_type_trends.columns if col != "year"]
    melted = patent_type_trends.melt(id_vars="year", value_vars=type_columns, var_name="patent_type", value_name="count")
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=melted, x="year", y="count", hue="patent_type")
    plt.title("Patent Type Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Patents")
    plt.legend(title="Patent Type")
    _save_figure(output_path)
    plt.close()


def plot_segment_trends(
    segment_trends: pd.DataFrame,
    segment_label: str,
    output_path: Optional[str] = None,
) -> None:
    """Plot yearly counts for a categorical segmentation variable."""
    if segment_trends.empty:
        return

    segment_columns = [col for col in segment_trends.columns if col != "year"]
    melted = segment_trends.melt(
        id_vars="year",
        value_vars=segment_columns,
        var_name=segment_label,
        value_name="count",
    )
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=melted, x="year", y="count", hue=segment_label)
    plt.title(f"{segment_label.replace('_', ' ').title()} Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Patents")
    plt.legend(title=segment_label.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def plot_segment_shares(
    segment_shares: pd.DataFrame,
    segment_label: str,
    output_path: Optional[str] = None,
) -> None:
    """Plot yearly shares for a categorical segmentation variable."""
    if segment_shares.empty:
        return

    segment_columns = [col for col in segment_shares.columns if col != "year"]
    melted = segment_shares.melt(
        id_vars="year",
        value_vars=segment_columns,
        var_name=segment_label,
        value_name="share",
    )
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=melted, x="year", y="share", hue=segment_label)
    plt.title(f"{segment_label.replace('_', ' ').title()} Shares Over Time")
    plt.xlabel("Year")
    plt.ylabel("Share of Total Patents")
    plt.legend(title=segment_label.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def plot_growth_rate(growth_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot annual patent growth rate."""
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=growth_df, x="year", y="growth_rate", marker="o")
    plt.axhline(0, linestyle="--")
    plt.title("Annual Patent Growth Rate")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate")
    _save_figure(output_path)
    plt.close()


def plot_title_feature_trends(title_summary: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot title-related feature trends over time."""
    if title_summary.empty:
        return
    feature_columns = [col for col in title_summary.columns if col != "year"]
    melted = title_summary.melt(id_vars="year", value_vars=feature_columns, var_name="feature", value_name="value")
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=melted, x="year", y="value", hue="feature")
    plt.title("Title-Based Feature Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Value")
    plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def plot_keyword_correlation(df: pd.DataFrame, keyword_columns: List[str], output_path: Optional[str] = None) -> None:
    """Plot a correlation heatmap for technology keywords."""
    plot_columns = [col for col in keyword_columns if col in df.columns]
    if not plot_columns:
        return
    corr = df[plot_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap of Technology Keywords")
    _save_figure(output_path)
    plt.close()


def plot_feature_correlations(feature_table: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Plot absolute correlations between annual candidate features and total patents."""
    if feature_table.empty:
        return

    plot_df = feature_table.sort_values(by="abs_correlation_with_total_patents", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=plot_df,
        x="abs_correlation_with_total_patents",
        y="feature",
        hue="feature_group",
        dodge=False,
    )
    plt.title("Annual Feature Relevance for Forecast Input Justification")
    plt.xlabel("Absolute Correlation with Total Patents")
    plt.ylabel("Feature")
    plt.legend(title="Feature Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(output_path)
    plt.close()


def generate_eda_summary(df: pd.DataFrame, annual_counts: pd.DataFrame) -> dict:
    """Generate a short summary of the cleaned dataset and annual patent counts."""
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "start_year": int(annual_counts["year"].min()) if not annual_counts.empty else None,
        "end_year": int(annual_counts["year"].max()) if not annual_counts.empty else None,
        "total_patents": int(annual_counts["total_patents"].sum()) if not annual_counts.empty else 0,
        "average_patents_per_year": float(annual_counts["total_patents"].mean()) if not annual_counts.empty else 0.0,
        "max_patents_in_a_year": int(annual_counts["total_patents"].max()) if not annual_counts.empty else 0,
        "min_patents_in_a_year": int(annual_counts["total_patents"].min()) if not annual_counts.empty else 0,
    }
    return summary
