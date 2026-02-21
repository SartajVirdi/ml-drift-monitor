<div align="center">

<br/>

```
                          ███╗   ███╗██╗     ██████╗ ██████╗ ██╗███████╗████████╗
                          ████╗ ████║██║     ██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝
                          ██╔████╔██║██║     ██║  ██║██████╔╝██║█████╗     ██║   
                          ██║╚██╔╝██║██║     ██║  ██║██╔══██╗██║██╔══╝     ██║   
                          ██║ ╚═╝ ██║███████╗██████╔╝██║  ██║██║██║        ██║   
                          ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝  
                                            M O N I T O R
```

**Production-grade ML drift detection. Plug in any model. Get instant answers.**

<br/>

[![Live Demo](https://img.shields.io/badge/LIVE_DEMO-mldriftmonitor.streamlit.app-00e5ff?style=for-the-badge&labelColor=0d1117)](https://mldriftmonitor.streamlit.app)
[![GitHub](https://img.shields.io/badge/SOURCE-SartajVirdi/ml--drift--monitor-ffffff?style=for-the-badge&logo=github&labelColor=0d1117)](https://github.com/SartajVirdi/ml-drift-monitor)

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-ffcc00?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistical_Tests-8caae6?style=flat-square&logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-a8ff78?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-00e5ff?style=flat-square)

<br/>

</div>

---

## The Problem

ML models degrade silently in production. The data they were trained on gradually diverges from what they see in deployment — a phenomenon called **data drift**. Without monitoring:

- Accuracy drops but no errors are raised
- Bad predictions compound undetected for weeks or months
- Engineers only find out when business metrics collapse

**ML Drift Monitor** gives any ML team a production-grade monitoring layer they can attach to any existing model — in minutes, without modifying their model code.

---

## At a Glance

| What | Detail |
|------|--------|
| Statistical Tests | Kolmogorov–Smirnov, Chi-Square, Population Stability Index |
| Feature Types | Numerical and Categorical — auto-detected |
| Model Formats | `.pkl`, `.joblib`, `.cbm` (sklearn, CatBoost, XGBoost, LightGBM) |
| Retraining | Automated — triggered by configurable thresholds |
| Output | Live Streamlit dashboard + JSON report + CSV scores + audit log |
| Config | Single `config.yaml` controls everything |

---

## Quickstart

**1. Clone and install**

```bash
git clone https://github.com/SartajVirdi/ml-drift-monitor
cd ml-drift-monitor
pip install -r requirements.txt
```

**2. Generate demo data (optional)**

```bash
python data/sample/generate_samples.py
```

Creates `reference.csv` and `production.csv` with intentional drift already injected — ready to run immediately.

**3. Run the pipeline**

```bash
# Default config
python main.py

# Custom config
python main.py --config config/my_config.yaml

# Inline arguments
python main.py --reference data/train.csv --production data/prod.csv --label churn
```

**4. Launch the dashboard**

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` — upload your CSVs from the sidebar and click **Run Analysis**.

---

## How to Prepare Your Data

Both CSVs must have the **same column names**. The label column is optional.

```
reference.csv                         production.csv
─────────────────────────────         ─────────────────────────────
age    income   contract   churn      age    income   contract   churn
35     50000    annual     0          28     92000    monthly    1
42     62000    monthly    1          31     87000    monthly    0
...                                   ...
```

- Drop ID or timestamp columns via `columns.drop` in config
- Mixed types are fine — the system auto-detects numerical vs categorical
- Label column accepts `0/1`, `True/False`, or `"Yes"/"No"`
- If labels are unavailable, omit the column — drift tests still run

---

## Configuration

A single `config.yaml` controls everything:

```yaml
data:
  reference_path:  "data/train.csv"
  production_path: "data/prod_batch.csv"
  label_column:    "churn"           # null if no labels

columns:
  numerical:   []                    # Leave empty for auto-detection
  categorical: []
  drop:        ["customer_id"]       # Drop ID / timestamp columns

model:
  path:      "models/my_model.pkl"   # null to skip performance tracking
  framework: "auto"                  # auto | sklearn | catboost

thresholds:
  ks_statistic:           0.10       # KS distance — flag feature
  psi:                    0.20       # PSI — flag feature + alert
  chi2_p_value:           0.05       # Chi-square p — flag categorical
  f1_drop:                0.05       # F1 degradation — performance alert
  drift_feature_fraction: 0.30       # >30% features drifted — overall alert

retraining:
  enabled:    true
  save_path:  "models/retrained.pkl"

output:
  report_path:      "data/drift_report.json"
  retrain_log_path: "logs/retrain_log.csv"
```

---

## Drift Detection Methods

<details>
<summary><strong>Kolmogorov–Smirnov Test — Numerical Features</strong></summary>

<br/>

Measures the maximum distance between two empirical cumulative distribution functions.

- KS = 0 → distributions are identical
- KS = 1 → distributions are completely different
- **Alert threshold: KS > 0.10**

```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(reference[col], production[col])
```

</details>

<details>
<summary><strong>Chi-Square Test — Categorical Features</strong></summary>

<br/>

Compares observed vs expected category frequencies to detect shifts in the category mix.

- High chi² or low p-value → distribution has shifted
- **Alert threshold: p < 0.05**

```python
from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(contingency_table)
```

</details>

<details>
<summary><strong>Population Stability Index — Numerical Features</strong></summary>

<br/>

Industry standard from credit scoring. Bins both distributions and compares proportions.

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10 | Stable — no action needed |
| 0.10 – 0.20 | Moderate change — monitor closely |
| > 0.20 | Significant drift — alert + retrain |

</details>

---

## Supported Model Formats

| Format | Frameworks | How to Save |
|--------|-----------|-------------|
| `.pkl` | sklearn, XGBoost, LightGBM, any | `joblib.dump(model, "model.pkl")` |
| `.joblib` | sklearn | `joblib.dump(model, "model.joblib")` |
| `.cbm` | CatBoost | `model.save_model("model.cbm")` |

Any scikit-learn compatible model works out of the box — `RandomForest`, `LogisticRegression`, `XGBClassifier`, `LGBMClassifier`, pipelines, and more.

---

## Dashboard

The live Streamlit dashboard is available at **[mldriftmonitor.streamlit.app](https://mldriftmonitor.streamlit.app)**

| Section | What It Shows |
|---------|--------------|
| Summary KPIs | Total features, drifted count, max PSI, max KS, F1 score |
| Overall Alert | Green (stable) or red (drift detected) status banner |
| Feature Bar Chart | Drift score 0–1 per feature, coloured by status |
| KS Table | Per-feature KS statistic, p-value, and flag |
| PSI Table | Per-feature PSI score and flag |
| Chi-Square Table | Per-feature statistic, p-value, and flag |
| Distribution Explorer | Interactive histogram or bar chart for any feature |
| Performance Metrics | Baseline vs production F1, accuracy, classification report |
| Retraining Log | Full audit trail of all retrain events this session |
| Download | JSON report and CSV feature scores |

---

## Example Workflow

```
Day 1   Train model on historical data → save as model.pkl
        Save training set as reference.csv

Day 30  Collect 1 month of production data → production.csv
        Run: python main.py --config config/prod.yaml

        Results:
        → KS test flags tenure (0.23) and monthly_charges (0.31)
        → PSI flags monthly_charges (0.47) → overall_drift = True
        → F1 dropped from 0.82 to 0.74 (drop = 0.08, threshold = 0.05)
        → Auto-retraining triggered → new model saved to models/retrained.pkl
        → Retrain log entry written to logs/retrain_log.csv

        Open dashboard → review all charts, download drift_report.json
```

---

## Extending the System

| Goal | Where to Change |
|------|----------------|
| Add a new statistical test | `drift/statistical_tests.py` + call it in `drift/detector.py` |
| Support a new model format | Add a loading branch in `training/model_handler.py` |
| Scheduled monitoring | Wrap `main.py` in a cron job or Airflow DAG |
| API mode | Import `run_pipeline()` from `main.py` into a FastAPI app |
| Slack alerts | Add a webhook call in `drift/detector.py` when `overall_drift_flag = True` |

---

## Project Structure

```
ml_drift_monitor/
│
├── config/
│   └── default_config.yaml       # All settings in one place
│
├── drift/
│   ├── statistical_tests.py      # KS, Chi-Square, PSI — pure functions
│   ├── detector.py               # Orchestrates tests → DriftReport
│   └── performance_tracker.py    # F1 tracking + retrain audit log
│
├── training/
│   └── model_handler.py          # Format-agnostic load/predict/retrain
│
├── utils/
│   ├── config_loader.py          # YAML loading + validation + defaults
│   ├── data_loader.py            # CSV loading + column auto-detection
│   └── logger.py                 # Structured logging
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (config-driven)
│
├── data/sample/
│   └── generate_samples.py       # Generate demo datasets
│
├── models/                       # Saved and retrained models
├── logs/                         # Drift logs + retrain audit CSV
│
├── main.py                       # CLI pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776ab?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-ffcc00?style=for-the-badge&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8caae6?style=for-the-badge&logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3f4f75?style=for-the-badge&logo=plotly&logoColor=white)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

[![Live Demo](https://img.shields.io/badge/TRY_IT_LIVE-mldriftmonitor.streamlit.app-00e5ff?style=for-the-badge&labelColor=0d1117)](https://mldriftmonitor.streamlit.app)

<br/>

*Built with Python · CatBoost · Streamlit · SciPy · Plotly*

</div>
