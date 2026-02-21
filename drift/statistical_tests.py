"""
drift/statistical_tests.py
---------------------------
Generic statistical drift tests.

All functions are pure: they take arrays/DataFrames and return structured
result DataFrames. No config, no logging, no side effects.

Tests implemented:
  1. Kolmogorov-Smirnov  (numerical)
  2. Chi-Square           (categorical)
  3. Population Stability Index  (numerical)
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── KS Test ───────────────────────────────────────────────────────────────────

def ks_test(reference: pd.Series, current: pd.Series, threshold: float) -> dict:
    """
    Two-sample KS test for one numerical feature.

    Returns:
        {feature, ks_statistic, p_value, drift_flag}
    """
    ref = reference.dropna()
    cur = current.dropna()

    if len(ref) == 0 or len(cur) == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_flag": False}

    stat, p_val = stats.ks_2samp(ref.values, cur.values)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value":      round(float(p_val), 4),
        "drift_flag":   bool(stat > threshold),
    }


def run_ks_tests(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    numerical_cols: list[str],
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Run KS tests across all numerical columns. Returns DataFrame indexed by feature."""
    records = []
    for col in numerical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        result = ks_test(reference_df[col], current_df[col], threshold)
        records.append({"feature": col, **result})

    if not records:
        return pd.DataFrame(columns=["feature","ks_statistic","p_value","drift_flag"]).set_index("feature")

    return pd.DataFrame(records).set_index("feature")


# ── Chi-Square Test ──────────────────────────────────────────────────────────

def chi2_test(
    reference: pd.Series,
    current:   pd.Series,
    p_threshold: float = 0.05,
) -> dict:
    """Chi-square test for one categorical feature."""
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)

    all_cats = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(all_cats, fill_value=0).values
    cur_counts = cur.value_counts().reindex(all_cats, fill_value=0).values

    contingency = np.vstack([ref_counts, cur_counts])
    nonzero = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero]

    if contingency.shape[1] < 2:
        return {"chi2_statistic": 0.0, "p_value": 1.0, "drift_flag": False}

    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value":        round(float(p_val), 4),
        "drift_flag":     bool(p_val < p_threshold),
    }


def run_chi2_tests(
    reference_df:    pd.DataFrame,
    current_df:      pd.DataFrame,
    categorical_cols: list[str],
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run Chi-square tests across all categorical columns."""
    records = []
    for col in categorical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        result = chi2_test(reference_df[col], current_df[col], p_threshold)
        records.append({"feature": col, **result})

    if not records:
        return pd.DataFrame(columns=["feature","chi2_statistic","p_value","drift_flag"]).set_index("feature")

    return pd.DataFrame(records).set_index("feature")


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(
    reference: pd.Series,
    current:   pd.Series,
    bins: int = 10,
) -> float:
    """
    Population Stability Index for one numerical feature.

    Interpretation:
        PSI < 0.10  → no significant change
        PSI 0.10–0.20 → moderate change (monitor)
        PSI > 0.20  → significant change (alert)
    """
    ref = reference.dropna().values
    cur = current.dropna().values

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    eps = 1e-6
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = (ref_counts / max(len(ref), 1)).clip(eps)
    cur_pct = (cur_counts / max(len(cur), 1)).clip(eps)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def run_psi(
    reference_df:   pd.DataFrame,
    current_df:     pd.DataFrame,
    numerical_cols: list[str],
    bins: int = 10,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Run PSI across all numerical columns."""
    records = []
    for col in numerical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi_val = compute_psi(reference_df[col], current_df[col], bins)
        records.append({
            "feature":    col,
            "psi":        psi_val,
            "drift_flag": bool(psi_val > threshold),
        })

    if not records:
        return pd.DataFrame(columns=["feature","psi","drift_flag"]).set_index("feature")

    return pd.DataFrame(records).set_index("feature")
