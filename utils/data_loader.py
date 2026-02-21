"""
utils/data_loader.py
--------------------
Generic dataset loading and column-type inference.

Design principle: ZERO hardcoded column names. Everything is driven by
the config or inferred from the data itself.

Auto-detection rules:
  - int64 / float64  → numerical
  - object / bool / category → categorical
  - User overrides in config always win over auto-detection.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("data_loader")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file with basic type coercion.

    Args:
        path: Absolute or relative path to the CSV.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or unreadable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    logger.info(f"Loaded {path} → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def prepare_dataset(
    df:         pd.DataFrame,
    label_col:  object,
    drop_cols:  list[str],
) -> Tuple[pd.DataFrame, object]:
    """
    Split a DataFrame into features (X) and optional labels (y).

    - Drops ID / timestamp columns specified in config.
    - Drops the label column from X.
    - Does NOT modify column types — that's the caller's job.

    Args:
        df:         Raw DataFrame from load_csv().
        label_col:  Name of the target column, or None.
        drop_cols:  List of columns to drop entirely (IDs, timestamps, etc.).

    Returns:
        (X, y) where y is None if label_col is None or not found.
    """
    df = df.copy()

    # Drop irrelevant columns
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped columns: {cols_to_drop}")

    # Extract labels
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col].copy()
        df.drop(columns=[label_col], inplace=True)
        logger.info(f"Label column '{label_col}' extracted. Positive rate: {y.mean():.2%}")
    elif label_col:
        logger.warning(
            f"Label column '{label_col}' not found in dataset. "
            "Performance drift tracking will be skipped."
        )

    return df, y


# ── Column type inference ─────────────────────────────────────────────────────

def infer_column_types(
    df:              pd.DataFrame,
    numerical_hint:  list[str],
    categorical_hint: list[str],
    cardinality_threshold: int = 20,
) -> Tuple[list[str], list[str]]:
    """
    Determine which columns are numerical and which are categorical.

    Priority order:
      1. User-specified lists in config (explicit override)
      2. DataFrame dtype (float/int → numerical, object → categorical)
      3. Low-cardinality integers → reclassified as categorical

    Args:
        df:                   Feature DataFrame (no label column).
        numerical_hint:       Columns explicitly marked numerical in config.
        categorical_hint:     Columns explicitly marked categorical in config.
        cardinality_threshold: Int columns with ≤ this many unique values
                               are treated as categorical.

    Returns:
        (numerical_cols, categorical_cols)
    """
    all_cols = list(df.columns)

    # If user specified everything explicitly, trust them
    if numerical_hint and categorical_hint:
        num = [c for c in numerical_hint if c in all_cols]
        cat = [c for c in categorical_hint if c in all_cols]
        logger.info(f"Column types from config: {len(num)} numerical, {len(cat)} categorical")
        return num, cat

    # Auto-detect
    num_auto, cat_auto = [], []

    for col in all_cols:
        # User override takes precedence
        if col in numerical_hint:
            num_auto.append(col)
            continue
        if col in categorical_hint:
            cat_auto.append(col)
            continue

        dtype = df[col].dtype

        if pd.api.types.is_float_dtype(dtype):
            num_auto.append(col)

        elif pd.api.types.is_integer_dtype(dtype):
            # Low-cardinality integers (e.g. SeniorCitizen 0/1, rating 1-5)
            # are more informative as categorical for drift purposes
            n_unique = df[col].nunique()
            if n_unique <= cardinality_threshold:
                cat_auto.append(col)
            else:
                num_auto.append(col)

        elif pd.api.types.is_bool_dtype(dtype):
            cat_auto.append(col)

        elif pd.api.types.is_object_dtype(dtype) or hasattr(dtype, "categories"):
            cat_auto.append(col)

        else:
            # Unknown type — treat as categorical (safe fallback)
            logger.warning(f"Unknown dtype for '{col}' ({dtype}) — treating as categorical")
            cat_auto.append(col)

    logger.info(
        f"Auto-detected column types: {len(num_auto)} numerical, {len(cat_auto)} categorical"
    )
    return num_auto, cat_auto


def coerce_types(
    df:      pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """
    Ensure columns have consistent dtypes for statistical testing.
    Numerical → float64, Categorical → str (handles mixed types cleanly).
    """
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", pd.NA).astype(str)
    return df


def validate_schema_match(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> list[str]:
    """
    Check that production data has the expected columns.
    Returns list of warning strings (empty = no issues).
    """
    warnings = []
    ref_cols  = set(ref_df.columns)
    prod_cols = set(prod_df.columns)

    missing   = ref_cols - prod_cols
    extra     = prod_cols - ref_cols

    if missing:
        warnings.append(f"Production data is missing columns: {sorted(missing)}")
    if extra:
        warnings.append(f"Production data has extra columns (will be ignored): {sorted(extra)}")

    return warnings
