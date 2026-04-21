"""
data_cleaning.py

This file contains functions used to check and lightly clean the patent dataset.

Important note:
This coursework dataset is already fairly structured, so our cleaning is not heavy.
We are mostly doing validation and basic preparation.

Why this file exists:
- To make sure the dataset is reliable before analysis
- To remove duplicate records
- To fix data type issues
- To validate binary columns such as keyword flags

In simple words:
This file checks whether the dataset is sensible and ready to use.
"""

from typing import List
import pandas as pd


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all column names.

    What this does:
    - removes extra spaces
    - converts names to lowercase

    Why we do this:
    Real datasets often contain inconsistent column names such as:
    'Patent_ID', ' patent_date ', 'KW_AI_ML'

    These differences can create coding errors later.
    By standardising column names at the start, we make the rest of the code easier and safer.

    Example:
    ' Patent_ID ' becomes 'patent_id'
    """
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def remove_duplicates(df: pd.DataFrame, id_column: str = "patent_id") -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Why this is important:
    Each patent should ideally appear only once.
    If the same patent appears multiple times, it will artificially inflate
    yearly patent counts and distort the analysis.

    Business meaning:
    If duplicates remain, Ericsson may appear to be filing more patents than it actually did.
    """
    df = df.copy()
    if id_column in df.columns:
        df = df.drop_duplicates(subset=id_column)
    else:
        df = df.drop_duplicates()
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert important columns into appropriate formats.

    Why this is needed:
    Some columns may be loaded as text even though they represent dates or numbers.

    Key conversions:
    - patent_date -> datetime format
    - year -> numeric format

    Why this matters:
    - We need year to be numeric for time series analysis
    - We need patent_date as a date if we want to explore timing-related patterns
    """
    df = df.copy()

    if "patent_date" in df.columns:
        df["patent_date"] = pd.to_datetime(df["patent_date"], errors="coerce", dayfirst=True)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def validate_binary_columns(df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
    """
    Ensure binary columns contain only 0 or 1.

    What are binary columns here?
    Examples:
    - kw_ai_ml
    - kw_5g
    - is_utility

    These variables should only indicate yes/no or present/absent.

    Why this matters:
    If a column that is supposed to mean "belongs to AI technology"
    contains invalid values like 2 or -1, the analysis becomes misleading.

    What this function does:
    - converts the column to numeric
    - replaces missing values with 0
    - forces values into the 0/1 range
    """
    df = df.copy()

    for col in binary_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].clip(lower=0, upper=1).round().astype(int)

    return df


def filter_invalid_years(df: pd.DataFrame, min_year: int = 1970, max_year: int = 2035) -> pd.DataFrame:
    """
    Remove rows with missing or unrealistic years.

    Why this matters:
    Forecasting and trend analysis rely heavily on the year column.
    If the dataset contains corrupted year values such as 1900 or 9999,
    then the time series charts and models will be distorted.
    """
    df = df.copy()

    if "year" in df.columns:
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]

    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of missing values.

    Why this is useful:
    Before analysis, it is good practice to understand whether important columns
    contain blanks or missing information.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
        }
    ).sort_values(by="missing_count", ascending=False)

    return summary


def clean_patent_data(df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
    """
    Run the full basic cleaning pipeline.

    This is the main cleaning function used by the project.

    Cleaning steps in order:
    1. standardise column names
    2. remove duplicate patents
    3. convert important columns into correct formats
    4. validate binary fields such as technology indicators
    5. remove unrealistic year values
    """
    df = standardise_column_names(df)
    df = remove_duplicates(df, id_column="patent_id")
    df = convert_data_types(df)
    df = validate_binary_columns(df, binary_columns)
    df = filter_invalid_years(df, min_year=1970, max_year=2035)

    return df
