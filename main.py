"""
main.py
-------
Generic ML Drift Monitor — command-line pipeline orchestrator.

Usage:
    python main.py                                    # uses config/default_config.yaml
    python main.py --config config/my_config.yaml     # custom config
    python main.py --reference data/train.csv \\
                   --production data/prod.csv \\
                   --label target                     # inline args (no yaml needed)

What it does:
    1. Loads reference + production datasets
    2. Auto-detects or reads column types
    3. Runs KS + Chi2 + PSI drift tests
    4. Evaluates model performance (if model + labels provided)
    5. Flags drift and triggers retraining if thresholds exceeded
    6. Saves JSON drift report + audit log
"""

import argparse
import json
import os
import sys

from utils.config_loader import load_config, config_from_dict
from utils.data_loader import (
    load_csv, prepare_dataset, infer_column_types,
    coerce_types, validate_schema_match,
)
from utils.logger import get_logger
from drift.detector import detect_drift
from drift.performance_tracker import detect_performance_drift, log_retrain_event
from training.model_handler import load_model, evaluate, retrain, build_default_classifier


def parse_args():
    p = argparse.ArgumentParser(description="ML Drift Monitor")
    p.add_argument("--config",     default="config/default_config.yaml")
    p.add_argument("--reference",  help="Override reference dataset path")
    p.add_argument("--production", help="Override production dataset path")
    p.add_argument("--model",      help="Override model path")
    p.add_argument("--label",      help="Override label column name")
    return p.parse_args()


def run_pipeline(config: dict) -> dict:
    """
    Execute the full drift monitoring pipeline.

    Args:
        config: Fully populated config dict.

    Returns:
        Results dict (also saved to disk).
    """
    log_path = config["output"].get("log_path", "logs/drift_monitor.log")
    logger = get_logger("pipeline", log_path=log_path)

    logger.info("═" * 60)
    logger.info("  ML DRIFT MONITOR — PIPELINE START")
    logger.info("═" * 60)

    # ── 1. Load datasets ──────────────────────────────────────────────────
    logger.info("\n📦 Loading datasets …")
    ref_path  = config["data"]["reference_path"]
    prod_path = config["data"]["production_path"]
    label_col = config["data"].get("label_column")
    drop_cols = config["columns"].get("drop", [])

    ref_raw  = load_csv(ref_path)
    prod_raw = load_csv(prod_path)

    # Schema validation
    schema_warnings = validate_schema_match(ref_raw, prod_raw)
    for w in schema_warnings:
        logger.warning(w)

    # Prepare features + labels
    ref_X,  ref_y  = prepare_dataset(ref_raw,  label_col, drop_cols)
    prod_X, prod_y = prepare_dataset(prod_raw, label_col, drop_cols)

    # ── 2. Column type detection ──────────────────────────────────────────
    logger.info("\n🔍 Detecting column types …")
    num_hint = config["columns"].get("numerical", [])
    cat_hint = config["columns"].get("categorical", [])

    numerical_cols, categorical_cols = infer_column_types(ref_X, num_hint, cat_hint)

    ref_X  = coerce_types(ref_X,  numerical_cols, categorical_cols)
    prod_X = coerce_types(prod_X, numerical_cols, categorical_cols)

    # Keep only columns that exist in both
    common_cols = [c for c in ref_X.columns if c in prod_X.columns]
    ref_X  = ref_X[common_cols]
    prod_X = prod_X[common_cols]
    numerical_cols   = [c for c in numerical_cols   if c in common_cols]
    categorical_cols = [c for c in categorical_cols if c in common_cols]

    logger.info(f"Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

    # ── 3. Drift detection ────────────────────────────────────────────────
    logger.info("\n📊 Running drift tests …")
    drift_report = detect_drift(
        reference_df=ref_X,
        current_df=prod_X,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        config=config,
        warnings=schema_warnings,
    )

    # ── 4. Model performance (if model + labels available) ────────────────
    model = None
    baseline_metrics = {}
    prod_metrics     = {}
    perf_drift       = False
    f1_drop          = 0.0

    model_path = config["model"].get("path")
    framework  = config["model"].get("framework", "auto")

    if model_path:
        logger.info("\n🤖 Loading model …")
        model = load_model(model_path, framework)

    has_labels = prod_y is not None and len(prod_y) > 0

    if model and has_labels:
        logger.info("📏 Evaluating model on production data …")
        prod_metrics = evaluate(model, prod_X, prod_y, label="Production")

        # Evaluate baseline on reference too
        if ref_y is not None:
            baseline_metrics = evaluate(model, ref_X, ref_y, label="Reference")

        baseline_f1 = baseline_metrics.get("f1_macro", 0.0)
        perf_drift, f1_drop = detect_performance_drift(
            prod_metrics,
            baseline_f1,
            config["thresholds"].get("f1_drop", 0.05),
        )
    elif model and not has_labels:
        logger.info("Labels not available — skipping performance evaluation.")

    # ── 5. Retraining decision ────────────────────────────────────────────
    retrain_triggered = False
    retrain_reason    = ""
    post_metrics      = {}

    retrain_cfg = config.get("retraining", {})
    retraining_enabled = retrain_cfg.get("enabled", True)
    requires_labels    = retrain_cfg.get("requires_labels", True)

    should_retrain = (
        retraining_enabled
        and (drift_report.overall_drift_flag or perf_drift)
        and (has_labels or not requires_labels)
    )

    if should_retrain:
        reasons = []
        if drift_report.overall_drift_flag:
            reasons.append(f"statistical drift ({drift_report.n_drifted}/{drift_report.n_features} features)")
        if perf_drift:
            reasons.append(f"F1 drop={f1_drop:.4f}")
        retrain_reason = " + ".join(reasons)

        logger.warning(f"🚨 RETRAINING TRIGGERED — {retrain_reason}")

        save_path = retrain_cfg.get("save_path", "models/retrained_model.pkl")

        if model:
            new_model = retrain(model, prod_X, prod_y, save_path)
        else:
            logger.info("No existing model — building default classifier.")
            new_model = build_default_classifier(prod_X, prod_y)
            import joblib; os.makedirs("models", exist_ok=True)
            joblib.dump(new_model, save_path)

        if new_model and has_labels:
            post_metrics = evaluate(new_model, ref_X, ref_y, label="Post-Retrain")

        log_retrain_event(
            retrain_log_path=config["output"].get("retrain_log_path", "logs/retrain_log.csv"),
            reason=retrain_reason,
            pre_f1=prod_metrics.get("f1_macro", 0.0),
            post_f1=post_metrics.get("f1_macro", 0.0),
        )
        retrain_triggered = True
    else:
        if not (drift_report.overall_drift_flag or perf_drift):
            logger.info("✅ No significant drift detected. No retraining needed.")

    # ── 6. Save drift report ──────────────────────────────────────────────
    results = {
        "drift": drift_report.to_dict(),
        "performance": {
            "baseline": baseline_metrics,
            "production": prod_metrics,
            "post_retrain": post_metrics,
            "perf_drift_detected": perf_drift,
            "f1_drop": f1_drop,
        },
        "retraining": {
            "triggered": retrain_triggered,
            "reason":    retrain_reason,
        },
        "meta": {
            "reference_path":  ref_path,
            "production_path": prod_path,
            "label_column":    label_col,
            "n_ref_rows":      len(ref_X),
            "n_prod_rows":     len(prod_X),
            "numerical_cols":  numerical_cols,
            "categorical_cols": categorical_cols,
            "schema_warnings": schema_warnings,
        },
    }

    report_path = config["output"].get("report_path", "data/drift_report.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n💾 Drift report saved → {report_path}")

    logger.info("\n" + "═" * 60)
    logger.info(f"  PIPELINE COMPLETE | Drift: {drift_report.overall_drift_flag} | Retrained: {retrain_triggered}")
    logger.info("═" * 60)

    return results


def main():
    args = parse_args()

    # Load base config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        # Allow running with pure CLI args (no yaml needed)
        config = load_config.__module__ and __import__(
            "utils.config_loader", fromlist=["config_from_dict"]
        ).config_from_dict({})

    # CLI overrides
    if args.reference:
        config["data"]["reference_path"] = args.reference
    if args.production:
        config["data"]["production_path"] = args.production
    if args.model:
        config["model"]["path"] = args.model
    if args.label:
        config["data"]["label_column"] = args.label

    run_pipeline(config)


if __name__ == "__main__":
    main()
