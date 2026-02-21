"""
training/model_handler.py
--------------------------
Generic model loading, prediction, evaluation, and retraining.

Supported model formats:
  .pkl / .joblib → scikit-learn compatible (any model with predict/predict_proba)
  .cbm           → CatBoost native format

Design choice: we wrap everything in try/except so a bad model path
never crashes the entire monitoring pipeline — it just skips performance tracking.
"""

import os
import joblib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

from utils.logger import get_logger

logger = get_logger("model_handler")

# ── Optional CatBoost import ──────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    _CATBOOST = True
except ImportError:
    _CATBOOST = False


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model(model_path: str, framework: str = "auto") -> Any:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model file.
        framework:  'auto' | 'sklearn' | 'catboost'
                    'auto' infers from file extension.

    Returns:
        Loaded model object, or None if loading fails.
    """
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}. Performance tracking disabled.")
        return None

    ext = os.path.splitext(model_path)[-1].lower()

    # Determine framework
    if framework == "auto":
        if ext == ".cbm":
            framework = "catboost"
        else:
            framework = "sklearn"

    try:
        if framework == "catboost":
            if not _CATBOOST:
                logger.error("CatBoost not installed. pip install catboost")
                return None
            model = CatBoostClassifier()
            model.load_model(model_path)
            logger.info(f"Loaded CatBoost model ← {model_path}")

        else:  # sklearn / joblib
            model = joblib.load(model_path)
            logger.info(f"Loaded sklearn model ← {model_path}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Run model inference on a feature DataFrame.
    Handles both sklearn and CatBoost APIs transparently.
    """
    if model is None:
        raise ValueError("Model is None — cannot predict.")

    if _CATBOOST and isinstance(model, CatBoostClassifier):
        from catboost import Pool
        pool = Pool(data=X)
        return model.predict(pool).astype(int).flatten()

    return model.predict(X).astype(int).flatten()


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return class probabilities. Returns None if model doesn't support it."""
    try:
        if _CATBOOST and isinstance(model, CatBoostClassifier):
            from catboost import Pool
            return model.predict_proba(Pool(data=X))
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
    except Exception as e:
        logger.warning(f"predict_proba failed: {e}")
    return None


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model:  Any,
    X:      pd.DataFrame,
    y:      pd.Series,
    label:  str = "Production",
) -> dict:
    """
    Evaluate a model on labelled data.

    Returns:
        dict with f1_macro, f1_binary, accuracy, and classification_report string.
        Returns empty dict if model is None or evaluation fails.
    """
    if model is None or y is None:
        return {}

    try:
        y_pred = predict(model, X)
        f1_mac = f1_score(y, y_pred, average="macro",  zero_division=0)
        f1_bin = f1_score(y, y_pred, average="binary", zero_division=0)
        acc    = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0)

        logger.info(f"[{label}] F1-macro={f1_mac:.4f} | F1-binary={f1_bin:.4f} | Accuracy={acc:.4f}")
        logger.info(f"\n{report}")

        return {
            "f1_macro":             round(f1_mac, 4),
            "f1_binary":            round(f1_bin, 4),
            "accuracy":             round(acc, 4),
            "classification_report": report,
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}


# ── Retraining ────────────────────────────────────────────────────────────────

def retrain(
    model:     Any,
    X_new:     pd.DataFrame,
    y_new:     pd.Series,
    save_path: str,
) -> Any:
    """
    Retrain the model on new data (same architecture, fresh fit).

    For sklearn models: calls model.fit(X_new, y_new).
    For CatBoost: rebuilds from scratch with same params.

    Args:
        model:     Original fitted model (used to get hyperparameters).
        X_new:     Feature DataFrame for retraining.
        y_new:     Label Series for retraining.
        save_path: Where to save the new model.

    Returns:
        Newly fitted model, or original model if retraining fails.
    """
    if model is None:
        logger.warning("No model to retrain.")
        return None

    logger.info(f"Retraining on {len(X_new):,} samples …")

    try:
        if _CATBOOST and isinstance(model, CatBoostClassifier):
            # Get original params, create fresh model
            params = model.get_params()
            params["verbose"] = 0
            new_model = CatBoostClassifier(**params)
            from catboost import Pool
            new_model.fit(Pool(data=X_new, label=y_new))

        else:
            # sklearn: clone and refit
            from sklearn.base import clone
            new_model = clone(model)
            new_model.fit(X_new, y_new)

        # Save
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if _CATBOOST and isinstance(new_model, CatBoostClassifier):
            new_model.save_model(save_path)
        else:
            joblib.dump(new_model, save_path)

        logger.info(f"Retrained model saved → {save_path}")
        return new_model

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return model  # return original as fallback


# ── Sklearn Pipeline helper ───────────────────────────────────────────────────

def build_default_classifier(X: pd.DataFrame, y: pd.Series) -> Any:
    """
    Build and fit a sensible default classifier when no model is provided
    but retraining is requested.

    Uses HistGradientBoostingClassifier — handles mixed types, no tuning needed.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.pipeline import Pipeline

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    pipe = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
    ])
    pipe.fit(X, y)
    logger.info("Built and fitted default HistGradientBoosting classifier.")
    return pipe
