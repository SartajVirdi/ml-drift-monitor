"""
drift/detector.py
-----------------
Orchestrates all drift tests and produces a unified DriftReport.

Design: accepts pre-loaded DataFrames + config dict.
        Returns a structured report with no side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from drift.statistical_tests import (
    run_ks_tests, run_chi2_tests, run_psi
)
from utils.logger import get_logger

logger = get_logger("detector")


@dataclass
class DriftReport:
    """
    Complete drift analysis for one production batch.
    Designed to be serialisable to JSON for logging and API responses.
    """
    ks_results:         pd.DataFrame
    chi2_results:       pd.DataFrame
    psi_results:        pd.DataFrame
    drift_scores:       pd.Series       # unified 0–1 score per feature
    overall_drift_flag: bool
    n_features:         int
    n_drifted:          int
    frac_drifted:       float
    max_ks:             float
    max_psi:            float
    warnings:           list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_drift_flag": self.overall_drift_flag,
            "n_features":         self.n_features,
            "n_drifted":          self.n_drifted,
            "frac_drifted":       round(self.frac_drifted, 3),
            "max_ks":             self.max_ks,
            "max_psi":            self.max_psi,
            "drift_scores":       self.drift_scores.to_dict(),
            "warnings":           self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def detect_drift(
    reference_df:    pd.DataFrame,
    current_df:      pd.DataFrame,
    numerical_cols:  list[str],
    categorical_cols: list[str],
    config:          dict,
    warnings:        object = None,
) -> DriftReport:
    """
    Run full drift detection suite and return a DriftReport.

    Args:
        reference_df:     Training/baseline feature DataFrame.
        current_df:       Production batch feature DataFrame.
        numerical_cols:   List of numerical column names.
        categorical_cols: List of categorical column names.
        config:           Full config dict (thresholds read from here).
        warnings:         Optional list of schema warnings to include.

    Returns:
        DriftReport dataclass.
    """
    thresholds = config.get("thresholds", {})
    ks_thresh   = thresholds.get("ks_statistic", 0.10)
    psi_thresh  = thresholds.get("psi", 0.20)
    chi2_p      = thresholds.get("chi2_p_value", 0.05)
    frac_thresh = thresholds.get("drift_feature_fraction", 0.30)
    psi_bins    = config.get("psi", {}).get("bins", 10)

    # Only test columns present in both datasets
    num_cols = [c for c in numerical_cols  if c in reference_df.columns and c in current_df.columns]
    cat_cols = [c for c in categorical_cols if c in reference_df.columns and c in current_df.columns]

    logger.info(f"Running drift tests on {len(num_cols)} numerical, {len(cat_cols)} categorical features")

    # ── Run tests ─────────────────────────────────────────────────────────
    ks_results   = run_ks_tests(reference_df,  current_df, num_cols,  ks_thresh)
    chi2_results = run_chi2_tests(reference_df, current_df, cat_cols, chi2_p)
    psi_results  = run_psi(reference_df, current_df, num_cols, psi_bins, psi_thresh)

    # ── Unified drift score per feature (0–1) ─────────────────────────────
    drift_scores = _compute_drift_scores(
        ks_results, chi2_results, psi_results,
        num_cols, cat_cols,
        ks_thresh, psi_thresh,
    )

    # ── Overall flag ──────────────────────────────────────────────────────
    n_features  = len(drift_scores)
    n_drifted   = int((drift_scores > 0.5).sum())
    frac_drifted = n_drifted / max(n_features, 1)
    max_ks  = float(ks_results["ks_statistic"].max())  if not ks_results.empty  else 0.0
    max_psi = float(psi_results["psi"].max())           if not psi_results.empty else 0.0

    overall_drift_flag = (frac_drifted > frac_thresh) or (max_psi > psi_thresh)

    logger.info(
        f"Drift summary: {n_drifted}/{n_features} features flagged | "
        f"max_KS={max_ks:.3f} | max_PSI={max_psi:.3f} | "
        f"overall_drift={overall_drift_flag}"
    )

    return DriftReport(
        ks_results=ks_results,
        chi2_results=chi2_results,
        psi_results=psi_results,
        drift_scores=drift_scores,
        overall_drift_flag=overall_drift_flag,
        n_features=n_features,
        n_drifted=n_drifted,
        frac_drifted=frac_drifted,
        max_ks=max_ks,
        max_psi=max_psi,
        warnings=warnings or [],
    )


def _compute_drift_scores(
    ks_results:   pd.DataFrame,
    chi2_results: pd.DataFrame,
    psi_results:  pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    ks_thresh:    float,
    psi_thresh:   float,
) -> pd.Series:
    """
    Compute a unified drift score (0–1) per feature.

    Numerical score = 0.4 × (KS / threshold).clip(0,1)
                    + 0.6 × (PSI / threshold).clip(0,1)

    Categorical score = 1.0 if chi2 flagged, else 0.0

    Returns:
        pd.Series indexed by feature name, sorted descending.
    """
    scores = {}

    for col in num_cols:
        ks_s = psi_s = 0.0
        if col in ks_results.index:
            ks_s = min(ks_results.loc[col, "ks_statistic"] / max(ks_thresh, 1e-9), 1.0)
        if col in psi_results.index:
            psi_s = min(psi_results.loc[col, "psi"] / max(psi_thresh, 1e-9), 1.0)
        scores[col] = round(0.4 * ks_s + 0.6 * psi_s, 4)

    for col in cat_cols:
        if col in chi2_results.index:
            scores[col] = float(chi2_results.loc[col, "drift_flag"])
        else:
            scores[col] = 0.0

    return pd.Series(scores, name="drift_score").sort_values(ascending=False)
