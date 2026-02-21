# 🔍 ML Drift Monitor

> A **generic, plug-and-play ML monitoring framework** — bring your own model and dataset, get instant drift analysis, performance tracking, and automated retraining.

---

## 🧩 What Problem Does This Solve?

ML models degrade silently in production. The data they were trained on gradually diverges from what they see in deployment — a phenomenon called **data drift**. Without monitoring:

- Model accuracy drops but no errors are raised
- Bad predictions compound undetected for weeks or months  
- Engineers only find out when business metrics collapse

This tool gives any ML team a **production-grade monitoring layer** they can attach to any existing model — in minutes, without modifying their model code.

---

## 🏗️ Architecture

```
Your Files:                    What the System Does:
────────────────               ──────────────────────────────────────────
reference.csv     ──────────►  Auto-detect column types
production.csv    ──────────►  Run KS / Chi-Square / PSI tests
model.pkl (opt.)  ──────────►  Score drift per feature (0–1)
config.yaml       ──────────►  Evaluate F1 / accuracy (if labels)
                               Flag features above threshold
                               Trigger auto-retraining if needed
                               Export JSON report + CSV scores
                               Render live Streamlit dashboard
```

### Folder Structure

```
ml_drift_monitor/
│
├── config/
│   └── default_config.yaml     # All settings in one place
│
├── drift/
│   ├── statistical_tests.py    # KS, Chi-Square, PSI — pure functions
│   ├── detector.py             # Orchestrates tests → DriftReport
│   └── performance_tracker.py  # F1 tracking + retrain audit log
│
├── training/
│   └── model_handler.py        # Format-agnostic model load/predict/retrain
│
├── utils/
│   ├── config_loader.py        # YAML loading + validation + defaults
│   ├── data_loader.py          # Generic CSV loading + column auto-detection
│   └── logger.py               # Structured logging
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard (fully config-driven)
│
├── data/
│   └── sample/
│       └── generate_samples.py # Generate demo datasets
│
├── models/                     # Saved / retrained models land here
├── logs/                       # Drift logs + retrain audit CSV
│
├── main.py                     # CLI pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/yourname/ml-drift-monitor
cd ml-drift-monitor
pip install -r requirements.txt
```

### 2. Generate sample data (optional demo)

```bash
python data/sample/generate_samples.py
```

This creates `data/sample/reference.csv` and `data/sample/production.csv` with intentional drift injected.

### 3. Run the pipeline (CLI)

```bash
python main.py
# Or with a custom config:
python main.py --config config/my_config.yaml
# Or with inline arguments:
python main.py --reference data/train.csv --production data/prod.csv --label churn
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` — upload your CSVs from the sidebar, click **Run Analysis**.

---

## 📄 How to Prepare Your CSV Files

Both CSVs must have the **same column names**. The label column is optional.

```
reference.csv                       production.csv
─────────────────────────────       ──────────────────────────────
age, income, contract, churn        age, income, contract, churn
35,  50000,  annual,   0            28,  92000,  monthly,   1
42,  62000,  monthly,  1            31,  87000,  monthly,   0
...                                 ...
```

**Rules:**
- No ID columns (drop them via `columns.drop` in config)
- Mixed types are fine — the system auto-detects numerical vs categorical
- Label column can be 0/1 or True/False or "Yes"/"No"
- If labels aren't available yet, just omit the column — drift tests still run

---

## ⚙️ Example config.yaml

```yaml
data:
  reference_path:  "data/train.csv"
  production_path: "data/prod_batch.csv"
  label_column:    "churn"        # Set to null if no labels

columns:
  numerical:   []                 # Leave empty for auto-detection
  categorical: []
  drop:        ["customer_id"]    # Drop ID/timestamp columns

model:
  path:      "models/my_model.pkl"   # null to skip performance tracking
  framework: "auto"                  # auto | sklearn | catboost

thresholds:
  ks_statistic:           0.10   # KS distance → flag feature
  psi:                    0.20   # PSI → flag feature + overall alert
  chi2_p_value:           0.05   # Chi-square p → flag categorical
  f1_drop:                0.05   # F1 degradation → performance alert
  drift_feature_fraction: 0.30   # >30% features drifted → overall alert

retraining:
  enabled:    true
  save_path:  "models/retrained.pkl"

output:
  report_path:      "data/drift_report.json"
  retrain_log_path: "logs/retrain_log.csv"
```

---

## 🔬 Drift Detection Explained

### Kolmogorov–Smirnov Test (Numerical)
Measures the maximum distance between two empirical CDFs.
- KS = 0 → identical distributions
- KS = 1 → completely different
- **Alert: KS > 0.10**

### Chi-Square Test (Categorical)
Compares observed vs expected category frequencies.
- Low p-value → distribution has shifted
- **Alert: p < 0.05**

### Population Stability Index (Numerical)
Industry standard from credit scoring. Bins both distributions, compares proportions.
```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```
- PSI < 0.10 → stable
- 0.10–0.20 → moderate change
- **PSI > 0.20 → significant drift → alert + retrain**

---

## 🤖 Supported Model Formats

| Format | Framework | How to Save |
|--------|-----------|-------------|
| `.pkl` | sklearn, xgboost, lightgbm, any | `joblib.dump(model, "model.pkl")` |
| `.joblib` | sklearn | `joblib.dump(model, "model.joblib")` |
| `.cbm` | CatBoost | `model.save_model("model.cbm")` |

Any scikit-learn compatible model works — `RandomForest`, `LogisticRegression`, `XGBClassifier`, `LGBMClassifier`, pipelines, etc.

---

## 📊 Dashboard Features

| Section | Description |
|---------|-------------|
| Summary KPIs | Total features, drifted count, max PSI, max KS, F1 |
| Overall Alert | Green (stable) or red (drift detected) banner |
| Feature Bar Chart | Drift score 0–1 per feature, coloured by status |
| KS Table | Per-feature KS statistic + p-value + flag |
| PSI Table | Per-feature PSI + flag |
| Chi-Square Table | Per-feature statistic + p-value + flag |
| Distribution Explorer | Interactive histogram / bar chart for any feature |
| Performance Metrics | Baseline vs production F1, accuracy, classification report |
| Retraining Log | Audit trail of all retrain events this session |
| Download | JSON report + CSV feature scores |

---

## 📋 Example Workflow

```
Day 1:  Train model on historical data → save as model.pkl
        Save training set as reference.csv

Day 30: Collect 1 month of production data → production.csv
        Run: python main.py --config config/prod.yaml
        → KS test flags tenure (0.23) and monthly_charges (0.31)
        → PSI flags monthly_charges (0.47) → overall_drift = True
        → F1 dropped from 0.82 → 0.74 (drop = 0.08 > threshold 0.05)
        → Auto-retraining triggered → new model saved
        → Retrain log entry written

Day 30: Open dashboard → see all charts, download drift_report.json
```

---

## 🛠️ Extending the System

- **Add a new test:** Add a function to `drift/statistical_tests.py` and call it in `drift/detector.py`
- **New model format:** Add a loading branch in `training/model_handler.py`
- **Scheduled monitoring:** Wrap `main.py` in a cron job or Airflow DAG
- **API mode:** Import `run_pipeline()` from `main.py` in a FastAPI app
- **Slack alerts:** Add a webhook call in `drift/detector.py` when `overall_drift_flag = True`

---

## 📄 Resume Bullet

> *Built a generic, production-grade ML drift monitoring framework in Python supporting plug-and-play model and dataset integration; implemented Kolmogorov–Smirnov, Chi-Square, and Population Stability Index statistical tests with auto column-type detection, configurable thresholds via YAML, automated model retraining with audit logging, and an interactive Streamlit dashboard featuring distribution exploration, feature-level drift scoring, and performance degradation tracking — designed as a reusable open-source tool applicable to any classification dataset.*
