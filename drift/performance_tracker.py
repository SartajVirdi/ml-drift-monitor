"""
drift/performance_tracker.py
-----------------------------
Tracks model performance over time and detects performance drift.

Gracefully handles the case where labels are not available —
in that case all functions return empty results rather than raising.
"""

import csv
import os
from datetime import datetime, timezone

import pandas as pd

from utils.logger import get_logger

logger = get_logger("performance_tracker")


def detect_performance_drift(
    current_metrics: dict,
    baseline_f1:     float,
    f1_drop_threshold: float = 0.05,
) -> tuple[bool, float]:
    """
    Compare current F1 against baseline.

    Args:
        current_metrics:   Output of model_handler.evaluate().
        baseline_f1:       F1-macro of the model at training time.
        f1_drop_threshold: Absolute drop that triggers an alert.

    Returns:
        (drift_detected: bool, f1_drop: float)
    """
    if not current_metrics:
        return False, 0.0

    current_f1 = current_metrics.get("f1_macro", 0.0)
    drop = baseline_f1 - current_f1
    flagged = drop > f1_drop_threshold

    if flagged:
        logger.warning(
            f"⚠ Performance drift: F1 dropped {drop:.4f} "
            f"(baseline={baseline_f1:.4f}, current={current_f1:.4f})"
        )
    return flagged, round(drop, 4)


def log_retrain_event(
    retrain_log_path: str,
    reason:           str,
    pre_f1:           float,
    post_f1:          float,
    extra: dict = None,
) -> None:
    """Append a retraining event row to the audit CSV."""
    os.makedirs(os.path.dirname(retrain_log_path) or ".", exist_ok=True)
    header_needed = not os.path.exists(retrain_log_path)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason":    reason,
        "pre_f1":    pre_f1,
        "post_f1":   post_f1,
        **(extra or {}),
    }

    with open(retrain_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Retrain event logged → {retrain_log_path}")
