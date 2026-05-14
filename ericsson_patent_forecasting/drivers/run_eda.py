"""
run_eda.py

This is the driver script for Task 1.

Task 1 workflow:
1. load raw dataset
2. check missing values
3. clean the dataset
4. create annual summary tables
5. generate charts
6. save outputs
"""

from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.utils import ensure_directory, get_keyword_columns, get_patent_type_columns
from src.data_loader import load_dataset, save_dataframe
from src.data_cleaning import check_missing_values, clean_patent_data
from src.feature_engineering import (
    create_annual_patent_counts,
    create_feature_justification_table,
    create_growth_rate,
    create_keyword_shares,
    create_keyword_trends,
    create_patent_type_trends,
    create_segment_shares,
    create_segment_trends,
    create_title_feature_summary,
)
from src.eda_analysis import (
    generate_eda_summary,
    plot_feature_correlations,
    plot_growth_rate,
    plot_keyword_correlation,
    plot_keyword_shares,
    plot_keyword_trends,
    plot_patent_type_trends,
    plot_patents_per_year,
    plot_segment_shares,
    plot_segment_trends,
    plot_title_feature_trends,
)
from src.time_series_diagnostics import (
    build_acf_table,
    build_adf_results,
    build_pacf_table,
    plot_correlation_lags,
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ericsson_patent_rich_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    ensure_directory(str(PROCESSED_DIR))
    ensure_directory(str(FIGURES_DIR))
    ensure_directory(str(TABLES_DIR))

    print("Loading raw dataset...")
    df_raw = load_dataset(str(DATA_PATH))

    print("Checking missing values...")
    missing_summary = check_missing_values(df_raw)
    save_dataframe(missing_summary, str(TABLES_DIR / "missing_values_summary.csv"))

    print("Cleaning dataset...")
    binary_columns = get_keyword_columns() + get_patent_type_columns()
    df_clean = clean_patent_data(df_raw, binary_columns=binary_columns)
    save_dataframe(df_clean, str(PROCESSED_DIR / "cleaned_patents.csv"))

    print("Creating annual summary tables...")
    keyword_columns = get_keyword_columns()
    patent_type_columns = get_patent_type_columns()
    annual_counts = create_annual_patent_counts(df_clean)
    patent_series = annual_counts["total_patents"]
    keyword_trends = create_keyword_trends(df_clean, keyword_columns)
    keyword_shares = create_keyword_shares(annual_counts, keyword_trends, keyword_columns)
    patent_type_trends = create_patent_type_trends(df_clean, patent_type_columns)
    patent_type_shares = create_segment_shares(patent_type_trends)
    tech_era_trends = create_segment_trends(df_clean, segment_column="tech_era")
    tech_era_shares = create_segment_shares(tech_era_trends)
    patent_type_category_trends = create_segment_trends(df_clean, segment_column="patent_type")
    growth_df = create_growth_rate(annual_counts)
    adf_results_df = build_adf_results(patent_series, series_name="total_patents")
    acf_df = build_acf_table(patent_series)
    pacf_df = build_pacf_table(patent_series)
    title_summary = create_title_feature_summary(df_clean)
    feature_justification = create_feature_justification_table(
        annual_counts=annual_counts,
        keyword_shares=keyword_shares,
        patent_type_trends=patent_type_trends,
        title_summary=title_summary,
    )

    save_dataframe(annual_counts, str(TABLES_DIR / "annual_patent_counts.csv"))
    save_dataframe(adf_results_df, str(TABLES_DIR / "task1_adf_results.csv"))
    save_dataframe(acf_df, str(TABLES_DIR / "task1_acf_values.csv"))
    save_dataframe(pacf_df, str(TABLES_DIR / "task1_pacf_values.csv"))
    save_dataframe(keyword_trends, str(TABLES_DIR / "keyword_trends.csv"))
    save_dataframe(keyword_shares, str(TABLES_DIR / "keyword_shares.csv"))
    save_dataframe(patent_type_trends, str(TABLES_DIR / "patent_type_trends.csv"))
    save_dataframe(patent_type_shares, str(TABLES_DIR / "patent_type_shares.csv"))
    save_dataframe(tech_era_trends, str(TABLES_DIR / "tech_era_trends.csv"))
    save_dataframe(tech_era_shares, str(TABLES_DIR / "tech_era_shares.csv"))
    save_dataframe(patent_type_category_trends, str(TABLES_DIR / "patent_type_category_trends.csv"))
    save_dataframe(growth_df, str(TABLES_DIR / "annual_growth_rate.csv"))
    save_dataframe(feature_justification, str(TABLES_DIR / "forecast_feature_justification.csv"))
    if not title_summary.empty:
        save_dataframe(title_summary, str(TABLES_DIR / "title_feature_summary.csv"))

    print("Generating figures...")
    plot_patents_per_year(annual_counts, output_path=str(FIGURES_DIR / "patents_per_year.png"))
    plot_correlation_lags(
        acf_df,
        title="Autocorrelation of Annual Patent Counts",
        y_label="ACF",
        output_path=str(FIGURES_DIR / "acf_patents_per_year.png"),
    )
    plot_correlation_lags(
        pacf_df,
        title="Partial Autocorrelation of Annual Patent Counts",
        y_label="PACF",
        output_path=str(FIGURES_DIR / "pacf_patents_per_year.png"),
    )
    plot_keyword_trends(keyword_trends, keyword_columns=keyword_columns, output_path=str(FIGURES_DIR / "keyword_trends.png"))
    plot_keyword_shares(keyword_shares, output_path=str(FIGURES_DIR / "keyword_shares.png"))
    plot_patent_type_trends(patent_type_trends, output_path=str(FIGURES_DIR / "patent_type_trends.png"))
    plot_segment_trends(tech_era_trends, segment_label="tech_era", output_path=str(FIGURES_DIR / "tech_era_trends.png"))
    plot_segment_shares(tech_era_shares, segment_label="tech_era", output_path=str(FIGURES_DIR / "tech_era_shares.png"))
    plot_segment_trends(
        patent_type_category_trends,
        segment_label="patent_type",
        output_path=str(FIGURES_DIR / "patent_type_category_trends.png"),
    )
    plot_segment_shares(
        patent_type_shares,
        segment_label="patent_type",
        output_path=str(FIGURES_DIR / "patent_type_shares.png"),
    )
    plot_growth_rate(growth_df, output_path=str(FIGURES_DIR / "growth_rate.png"))
    plot_keyword_correlation(df_clean, keyword_columns=keyword_columns, output_path=str(FIGURES_DIR / "keyword_correlation_heatmap.png"))
    plot_feature_correlations(feature_justification, output_path=str(FIGURES_DIR / "forecast_feature_relevance.png"))
    if not title_summary.empty:
        plot_title_feature_trends(title_summary, output_path=str(FIGURES_DIR / "title_feature_trends.png"))

    summary = generate_eda_summary(df_clean, annual_counts)
    print("\n----- TASK 1 EDA SUMMARY -----")
    print(f"Rows in cleaned dataset       : {summary['n_rows']}")
    print(f"Columns in cleaned dataset    : {summary['n_columns']}")
    print(f"Start year                    : {summary['start_year']}")
    print(f"End year                      : {summary['end_year']}")
    print(f"Total patents                 : {summary['total_patents']}")
    print(f"Average patents per year      : {summary['average_patents_per_year']:.2f}")
    print(f"Maximum patents in a year     : {summary['max_patents_in_a_year']}")
    print(f"Minimum patents in a year     : {summary['min_patents_in_a_year']}")
    print("\nTop annual feature relationships with total patents:")
    print(feature_justification.head(8).to_string(index=False))

    print("\nTask 1 EDA pipeline completed successfully.")
    print(f"Processed data saved to : {PROCESSED_DIR}")
    print(f"Tables saved to         : {TABLES_DIR}")
    print(f"Figures saved to        : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
