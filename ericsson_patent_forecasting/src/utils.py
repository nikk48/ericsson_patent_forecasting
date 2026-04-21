"""
utils.py

This file contains small helper functions that are used across the project.

Why this file exists:
- So we do not repeat common code in many places
- So the project stays neat and modular
- So commonly used settings such as keyword column names are stored in one place

Think of this file as a small toolbox used by the rest of the project.
"""

from pathlib import Path
from typing import List


def ensure_directory(path: str) -> Path:
    """
    Create a folder if it does not already exist.

    Simple explanation:
    Before saving outputs such as figures, tables, or processed data,
    we need to make sure the destination folder exists.
    If the folder does not exist, Python would throw an error while saving.

    Example:
    If we want to save a chart into outputs/figures/, this function makes sure
    that the folder is available first.

    Parameters
    ----------
    path : str
        The folder path we want to create or check.

    Returns
    -------
    Path
        A Path object representing that folder.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_keyword_columns() -> List[str]:
    """
    Return the list of technology keyword columns in the dataset.

    Simple explanation:
    The patent dataset includes binary keyword flags such as:
    - kw_ai_ml
    - kw_5g
    - kw_cloud_edge

    A value of 1 means the patent relates to that technology.
    A value of 0 means it does not.

    Why this function is useful:
    Instead of manually writing the same list again and again in different files,
    we define it once here and reuse it everywhere.
    This reduces errors and makes the code easier to maintain.
    """
    return [
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


def get_patent_type_columns() -> List[str]:
    """
    Return the list of patent type indicator columns.

    Simple explanation:
    The dataset also tells us whether a patent is:
    - a utility patent
    - a design patent
    - another type

    These columns are useful in Task 1 because they help us understand
    the composition of Ericsson's patent portfolio over time.
    """
    return ["is_utility", "is_design", "is_other_type"]
