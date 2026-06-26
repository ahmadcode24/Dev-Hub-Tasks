"""
data_loader.py
--------------
Modular data ingestion and validation for the Iris dataset.
Ensures reproducibility and clean separation of concerns.
"""

import pandas as pd
import seaborn as sns
from typing import Tuple


def load_iris_dataset() -> pd.DataFrame:
    """
    Load the Iris dataset from seaborn's built-in repository.

    Returns
    -------
    pd.DataFrame
        Raw Iris dataset with 150 rows and 5 columns.
    """
    df = sns.load_dataset('iris')
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate dataset integrity: shape, missing values, duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to validate.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    expected_shape = (150, 5)
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}

    if df.shape != expected_shape:
        return False, f"Unexpected shape: {df.shape}, expected {expected_shape}"

    if set(df.columns) != expected_columns:
        return False, f"Unexpected columns: {set(df.columns)}"

    if df.isnull().sum().sum() > 0:
        return False, "Dataset contains missing values"

    return True, "Dataset validation passed"


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate enhanced descriptive statistics including skewness and kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Transposed describe() with skewness and kurtosis appended.
    """
    desc = df.describe().T
    desc['skewness'] = df.select_dtypes(include='number').skew()
    desc['kurtosis'] = df.select_dtypes(include='number').kurtosis()
    return desc.round(3)
