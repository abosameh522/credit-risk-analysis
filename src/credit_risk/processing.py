import pandas as pd

from .constants import CATEGORY_LABELS, COLUMN_NAMES


def preprocess_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes raw credit risk data by renaming columns, mapping categories,
    and transforming the target variable.
    """
    # Rename columns
    df = df.rename(columns=COLUMN_NAMES).copy()

    # Map categories
    for column, labels in CATEGORY_LABELS.items():
        if column not in df.columns:
            continue

        # Validate codes
        unexpected = set(df[column].dropna().unique()) - set(labels.keys())
        if unexpected:
            raise ValueError(f"Unexpected codes in {column}: {unexpected}")

        df[column] = df[column].map(labels).astype("category")

    # Transform target
    if "credit_risk" not in df.columns:
        raise KeyError("Column 'credit_risk' missing from DataFrame")

    if not set(df.credit_risk.unique()).issubset({0, 1}):
        raise ValueError("Unexpected target values in 'credit_risk'. Expected {0, 1}")

    df["bad_credit"] = (df.credit_risk == 0).astype(int)
    df = df.drop(columns="credit_risk")

    return df
