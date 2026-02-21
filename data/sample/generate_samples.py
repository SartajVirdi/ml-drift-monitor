"""
data/sample/generate_samples.py
--------------------------------
Generates sample reference + production CSVs for testing the drift monitor.
Run once to populate data/sample/ with realistic demo data.

Usage:
    python data/sample/generate_samples.py
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_reference(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n).astype(float)
    monthly = rng.uniform(20, 100, size=n).round(2)
    churn_prob = 1 / (1 + np.exp(-(0.03 * monthly - 0.04 * tenure)))
    return pd.DataFrame({
        "age":            rng.integers(18, 70, size=n),
        "tenure":         tenure,
        "monthly_charges":monthly,
        "total_charges":  (tenure * monthly).clip(0).round(2),
        "num_products":   rng.integers(1, 5, size=n),
        "contract_type":  rng.choice(["Month-to-month","One year","Two year"], size=n, p=[0.55,0.25,0.20]),
        "payment_method": rng.choice(["Credit card","Bank transfer","Electronic check","Mailed check"], size=n),
        "internet_service":rng.choice(["DSL","Fiber optic","No"], size=n, p=[0.35,0.45,0.20]),
        "online_security":rng.choice(["Yes","No"], size=n),
        "senior_citizen": rng.choice([0, 1], size=n, p=[0.83, 0.17]),
        "target":         (rng.uniform(size=n) < churn_prob).astype(int),
    })


def generate_production(n: int = 400, seed: int = 99) -> pd.DataFrame:
    """Production batch with injected drift."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 25, size=n).astype(float)          # shorter tenure (drift)
    monthly = rng.uniform(70, 120, size=n).round(2)              # higher charges (drift)
    churn_prob = 1 / (1 + np.exp(-(0.04 * monthly - 0.02 * tenure)))
    return pd.DataFrame({
        "age":            rng.integers(18, 35, size=n),           # younger customers (drift)
        "tenure":         tenure,
        "monthly_charges":monthly,
        "total_charges":  (tenure * monthly).clip(0).round(2),
        "num_products":   rng.integers(1, 3, size=n),
        "contract_type":  rng.choice(["Month-to-month","One year","Two year"], size=n, p=[0.85,0.10,0.05]),  # drift
        "payment_method": rng.choice(["Electronic check","Credit card"], size=n, p=[0.80,0.20]),             # drift
        "internet_service":rng.choice(["Fiber optic","No"], size=n, p=[0.95,0.05]),                          # drift
        "online_security":rng.choice(["Yes","No"], size=n, p=[0.15,0.85]),
        "senior_citizen": rng.choice([0, 1], size=n, p=[0.60, 0.40]),                                        # drift
        "target":         (rng.uniform(size=n) < churn_prob).astype(int),
    })


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref  = generate_reference(1000)
    prod = generate_production(400)

    ref_path  = os.path.join(OUTPUT_DIR, "reference.csv")
    prod_path = os.path.join(OUTPUT_DIR, "production.csv")

    ref.to_csv(ref_path,  index=False)
    prod.to_csv(prod_path, index=False)

    print(f"✅ Reference dataset  → {ref_path}  ({len(ref):,} rows)")
    print(f"✅ Production dataset → {prod_path} ({len(prod):,} rows)")
    print(f"\nChurn rate: reference={ref['target'].mean():.1%} | production={prod['target'].mean():.1%}")
    print("\nRun the drift monitor:")
    print("  python main.py")
    print("  streamlit run dashboard/app.py")
