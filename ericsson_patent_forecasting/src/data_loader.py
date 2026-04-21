"""
data_loader.py

This file is responsible for reading data into Python and saving data out of Python.

Why this file exists:
- To separate data input/output from the rest of the analysis
- To keep code organized
- To avoid writing read_csv and to_csv in multiple places

In simple words:
This file handles opening and saving files.
"""

import pandas as pd
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Ericsson patent dataset from a CSV file.

    Simple explanation:
    This function reads the raw dataset from your folder and converts it into
    a pandas DataFrame, which is the main table format we use in Python.

    Why this matters:
    All later analysis starts here. If the dataset is not loaded correctly,
    nothing else will work.

    Parameters
    ----------
    file_path : str
        The location of the CSV file on your computer.

    Returns
    -------
    pd.DataFrame
        The dataset loaded as a table.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(path)
    return df


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame into a CSV file.

    Simple explanation:
    During the project, we will create many useful outputs such as:
    - cleaned data
    - annual patent summaries
    - model results
    - forecast tables

    Instead of keeping everything only in memory, we save these outputs
    as CSV files so they can be reused later and included in the report.

    Parameters
    ----------
    df : pd.DataFrame
        The table we want to save.
    file_path : str
        The output file location.

    Returns
    -------
    None
        This function saves the file and does not return anything.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
