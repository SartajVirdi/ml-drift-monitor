"""
dashboard/app.py - Generic ML Drift Monitoring Dashboard
Run with: streamlit run dashboard/app.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import config_from_dict
from utils.data_loader import load_csv, prepare_dataset, infer_column_types, coerce_types, validate_schema_match
from drift.detector import detect_drift
from drift.performance_tracker import detect_performance_drift, log_retrain_event
from training.model_handler import load_model, evaluate, retrain, predict

st.set_page_config(page_title="ML Drift Monitor", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
  .alert-box  { background:#3d1515;border:1px solid #ff4444;border-radius:8px;padding:14px 18px;margin:8px 0; }
  .ok-box     { background:#153d1e;border:1px solid #44ff88;border-radius:8px;padding:14px 18px;margin:8px 0; }
  .warn-box   { background:#2d2a10;border:1px solid #ffcc00;border-radius:8px;padding:14px 18px;margin:8px 0; }
  .info-box   { background:#0d1b2e;border:1px solid #4a9eff;border-radius:8px;padding:14px 18px;margin:8px 0;color:#8bc4ff; }
  h1 { color:#c0b0ff; } h2 { color:#e0d8ff; }
</style>""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "drift_report": None, "ref_df": None, "prod_df": None,
        "ref_y": None, "prod_y": None, "model": None,
        "numerical_cols": [], "categorical_cols": [],
        "perf_metrics": {}, "baseline_metrics": {}, "config": {},
        "perf_history": [], "retrain_log": [], "analysis_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    config_mode = st.sidebar.radio("Config mode", ["📝 Manual (sidebar)", "📄 Paste YAML"], horizontal=True)
    raw_config = {}

    if config_mode == "📄 Paste YAML":
        yaml_input = st.sidebar.text_area("Paste your config.yaml contents:", height=300,
            placeholder="data:\n  reference_path: data/train.csv\n  ...")
        if yaml_input.strip():
            try:
                raw_config = yaml.safe_load(yaml_input) or {}
                st.sidebar.success("✅ YAML parsed successfully")
            except yaml.YAMLError as e:
                st.sidebar.error(f"YAML error: {e}")
    else:
        st.sidebar.markdown("### 📂 Data")
        ref_upload   = st.sidebar.file_uploader("Reference dataset (CSV)", type=["csv"], key="ref_upload")
        prod_upload  = st.sidebar.file_uploader("Production dataset (CSV)", type=["csv"], key="prod_upload")
        label_col    = st.sidebar.text_input("Label column (leave blank if none)", value="")
        st.sidebar.markdown("### 🤖 Model (optional)")
        model_upload = st.sidebar.file_uploader("Trained model (.pkl, .joblib, .cbm)", type=["pkl","joblib","cbm"], key="model_upload")
        st.sidebar.markdown("### 📐 Thresholds")
        ks_thresh    = st.sidebar.slider("KS threshold",   0.01, 0.50, 0.10, 0.01)
        psi_thresh   = st.sidebar.slider("PSI threshold",  0.05, 0.50, 0.20, 0.05)
        f1_drop      = st.sidebar.slider("F1 drop alert",  0.01, 0.20, 0.05, 0.01)
        drift_frac   = st.sidebar.slider("Drift fraction", 0.10, 0.80, 0.30, 0.05)
        st.sidebar.markdown("### 🔄 Retraining")
        retrain_enabled = st.sidebar.checkbox("Enable auto-retraining", value=True)

        ref_path = prod_path = model_path = None
        if ref_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(ref_upload.read()); tmp.flush(); ref_path = tmp.name
        if prod_upload:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(prod_upload.read()); tmp.flush(); prod_path = tmp.name
        if model_upload:
            ext = os.path.splitext(model_upload.name)[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(model_upload.read()); tmp.flush(); model_path = tmp.name

        raw_config = {
            "data": {"reference_path": ref_path, "production_path": prod_path, "label_column": label_col or None},
            "columns": {"numerical": [], "categorical": [], "drop": []},
            "model": {"path": model_path, "framework": "auto"},
            "thresholds": {"ks_statistic": ks_thresh, "psi": psi_thresh, "f1_drop": f1_drop,
                           "drift_feature_fraction": drift_frac, "chi2_p_value": 0.05},
            "retraining": {"enabled": retrain_enabled, "requires_labels": True, "save_path": "models/retrained_model.pkl"},
            "output": {"report_path": "data/drift_report.json", "log_path": "logs/drift_monitor.log",
                       "retrain_log_path": "logs/retrain_log.csv"},
        }

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    return raw_config, run_btn


def run_analysis(config):
    ref_path  = config["data"]["reference_path"]
    prod_path = config["data"]["production_path"]
    label_col = config["data"].get("label_column")
    drop_cols = config["columns"].get("drop", [])

    with st.spinner("Loading datasets …"):
        ref_raw  = load_csv(ref_path)
        prod_raw = load_csv(prod_path)

    schema_warnings = validate_schema_match(ref_raw, prod_raw)
    ref_X,  ref_y  = prepare_dataset(ref_raw,  label_col, drop_cols)
    prod_X, prod_y = prepare_dataset(prod_raw, label_col, drop_cols)

    with st.spinner("Detecting column types …"):
        num_hint = config["columns"].get("numerical",   [])
        cat_hint = config["columns"].get("categorical", [])
        num_cols, cat_cols = infer_column_types(ref_X, num_hint, cat_hint)
        ref_X  = coerce_types(ref_X,  num_cols, cat_cols)
        prod_X = coerce_types(prod_X, num_cols, cat_cols)
        common = [c for c in ref_X.columns if c in prod_X.columns]
        ref_X  = ref_X[common]; prod_X = prod_X[common]
        num_cols = [c for c in num_cols if c in common]
        cat_cols = [c for c in cat_cols if c in common]

    with st.spinner("Running drift tests …"):
        report = detect_drift(reference_df=ref_X, current_df=prod_X,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            config=config, warnings=schema_warnings)

    model = None; baseline_metrics = {}; prod_metrics = {}
    model_path = config["model"].get("path")
    if model_path:
        with st.spinner("Loading model …"):
            model = load_model(model_path, config["model"].get("framework","auto"))

    has_labels = prod_y is not None and len(prod_y) > 0
    if model and has_labels:
        with st.spinner("Evaluating model …"):
            prod_metrics     = evaluate(model, prod_X, prod_y, "Production")
            if ref_y is not None:
                baseline_metrics = evaluate(model, ref_X, ref_y, "Reference")

    retrain_log = st.session_state.get("retrain_log", [])
    retrain_cfg = config.get("retraining", {})
    baseline_f1 = baseline_metrics.get("f1_macro", 0.0)
    perf_drift, f1_drop = detect_performance_drift(prod_metrics, baseline_f1,
        config["thresholds"].get("f1_drop", 0.05)) if (model and has_labels) else (False, 0.0)

    should_retrain = (retrain_cfg.get("enabled", True)
        and (report.overall_drift_flag or perf_drift)
        and (has_labels or not retrain_cfg.get("requires_labels", True))
        and model is not None)

    retrain_triggered = False
    if should_retrain:
        reasons = []
        if report.overall_drift_flag:
            reasons.append(f"statistical drift ({report.n_drifted}/{report.n_features} features)")
        if perf_drift:
            reasons.append(f"F1 drop={f1_drop:.4f}")
        with st.spinner("Auto-retraining model …"):
            save_path = retrain_cfg.get("save_path","models/retrained_model.pkl")
            new_model = retrain(model, prod_X, prod_y, save_path)
            post_metrics = evaluate(new_model, ref_X, ref_y, "Post-Retrain") if (new_model and ref_y is not None) else {}
        retrain_log.append({"reason": " + ".join(reasons),
            "pre_f1": prod_metrics.get("f1_macro",0.0),
            "post_f1": post_metrics.get("f1_macro",0.0),
            "f1_change": post_metrics.get("f1_macro",0.0) - prod_metrics.get("f1_macro",0.0)})
        retrain_triggered = True

    st.session_state.drift_report     = report
    st.session_state.ref_df           = ref_X
    st.session_state.prod_df          = prod_X
    st.session_state.ref_y            = ref_y
    st.session_state.prod_y           = prod_y
    st.session_state.model            = model
    st.session_state.numerical_cols   = num_cols
    st.session_state.categorical_cols = cat_cols
    st.session_state.perf_metrics     = prod_metrics
    st.session_state.baseline_metrics = baseline_metrics
    st.session_state.config           = config
    st.session_state.retrain_log      = retrain_log
    st.session_state.retrain_triggered = retrain_triggered
    st.session_state.schema_warnings  = schema_warnings
    st.session_state.analysis_run     = True
    hist = st.session_state.get("perf_history", [])
    hist.append({"f1_macro": prod_metrics.get("f1_macro", None)})
    st.session_state.perf_history = hist


def _style_df(df, flag_col):
    def hl(row):
        if row.get(flag_col, False):
            return ["background-color:rgba(255,68,68,0.2)"]*len(row)
        return [""]*len(row)
    return df.style.apply(hl, axis=1)


def render_dashboard():
    report      = st.session_state.drift_report
    config      = st.session_state.config
    num_cols    = st.session_state.numerical_cols
    cat_cols    = st.session_state.categorical_cols
    ref_X       = st.session_state.ref_df
    prod_X      = st.session_state.prod_df
    perf        = st.session_state.perf_metrics
    baseline    = st.session_state.baseline_metrics
    retrain_log = st.session_state.retrain_log
    warnings    = st.session_state.get("schema_warnings", [])
    thresholds  = config.get("thresholds", {})

    for w in warnings:
        st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Features",   report.n_features)
    with c2: st.metric("Drifted Features", f"{report.n_drifted} / {report.n_features}")
    with c3: st.metric("Max PSI",          f"{report.max_psi:.3f}")
    with c4: st.metric("Max KS",           f"{report.max_ks:.3f}")
    with c5:
        f1 = perf.get("f1_macro")
        if f1 is not None:
            bl_f1 = baseline.get("f1_macro", f1)
            st.metric("F1-macro (prod)", f"{f1:.4f}", delta=f"{f1-bl_f1:+.4f}")
        else:
            st.metric("F1-macro", "N/A", help="Provide a model + labels to see this")

    st.markdown("---")
    if report.overall_drift_flag:
        st.markdown(f'<div class="alert-box">🚨 <b>DRIFT ALERT</b> — {report.n_drifted}/{report.n_features} features flagged | max PSI={report.max_psi:.3f} | max KS={report.max_ks:.3f}</div>', unsafe_allow_html=True)
        if st.session_state.get("retrain_triggered"):
            st.markdown('<div class="warn-box">♻️ <b>Auto-retraining was triggered</b> — See the Retraining Log below.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">✅ <b>No significant drift detected</b> — Production data is similar to training data.</div>', unsafe_allow_html=True)

    with st.expander("📂 Dataset Info", expanded=False):
        di1, di2 = st.columns(2)
        with di1:
            st.markdown(f"**Reference:** `{config['data']['reference_path']}`")
            st.markdown(f"Rows: **{len(ref_X):,}** | Numerical: **{len(num_cols)}** | Categorical: **{len(cat_cols)}**")
        with di2:
            st.markdown(f"**Production:** `{config['data']['production_path']}`")
            st.markdown(f"Rows: **{len(prod_X):,}** | Label: **{config['data'].get('label_column') or 'None'}**")

    st.markdown("### 🔎 Feature-Level Drift Score")
    scores = report.drift_scores
    if not scores.empty:
        score_df = scores.reset_index()
        score_df.columns = ["Feature","Drift Score"]
        score_df["Status"] = score_df["Drift Score"].apply(lambda x: "🔴 Drifted" if x>0.5 else "🟢 Stable")
        score_df["Type"]   = score_df["Feature"].apply(lambda c: "Numerical" if c in num_cols else "Categorical")
        fig = px.bar(score_df, x="Drift Score", y="Feature", orientation="h", color="Status",
            color_discrete_map={"🔴 Drifted":"#ff4444","🟢 Stable":"#44bb88"},
            pattern_shape="Type", title="Drift Score per Feature",
            height=max(350, len(score_df)*26))
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Alert Threshold")
        fig.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font_color="#fafafa",
            yaxis=dict(autorange="reversed"), legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📐 Statistical Test Results")
    t1,t2,t3 = st.tabs(["KS Test (Numerical)","PSI (Numerical)","Chi-Square (Categorical)"])
    with t1:
        if not report.ks_results.empty:
            st.dataframe(_style_df(report.ks_results.reset_index(),"drift_flag"), use_container_width=True)
            st.caption(f"Alert: KS > {thresholds.get('ks_statistic',0.10)}")
        else: st.info("No numerical features.")
    with t2:
        if not report.psi_results.empty:
            st.dataframe(_style_df(report.psi_results.reset_index(),"drift_flag"), use_container_width=True)
            st.caption("PSI < 0.10 stable | 0.10–0.20 moderate | > 0.20 significant")
        else: st.info("No numerical features.")
    with t3:
        if not report.chi2_results.empty:
            st.dataframe(_style_df(report.chi2_results.reset_index(),"drift_flag"), use_container_width=True)
            st.caption(f"Alert: p-value < {thresholds.get('chi2_p_value',0.05)}")
        else: st.info("No categorical features.")

    st.markdown("### 📊 Distribution Explorer")
    all_features = num_cols + cat_cols
    if all_features:
        selected = st.selectbox("Choose a feature:", all_features)
        if selected in num_cols:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=ref_X[selected].dropna(), name="Reference", opacity=0.65, marker_color="#4a9eff", histnorm="probability"))
            fig_dist.add_trace(go.Histogram(x=prod_X[selected].dropna(), name="Production", opacity=0.65, marker_color="#ff4d6d", histnorm="probability"))
            fig_dist.update_layout(barmode="overlay", title=f"Distribution: {selected}",
                paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font_color="#fafafa")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            ref_vc  = ref_X[selected].value_counts(normalize=True).reset_index()
            prod_vc = prod_X[selected].value_counts(normalize=True).reset_index()
            ref_vc.columns  = [selected,"Reference"]
            prod_vc.columns = [selected,"Production"]
            merged = ref_vc.merge(prod_vc, on=selected, how="outer").fillna(0)
            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(x=merged[selected],y=merged["Reference"], name="Reference", marker_color="#4a9eff"))
            fig_cat.add_trace(go.Bar(x=merged[selected],y=merged["Production"],name="Production",marker_color="#ff4d6d"))
            fig_cat.update_layout(barmode="group",title=f"Category Frequencies: {selected}",
                paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font_color="#fafafa")
            st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### 📈 Model Performance")
    if perf:
        p1,p2,p3 = st.columns(3)
        bl_f1 = baseline.get("f1_macro", perf.get("f1_macro",0))
        with p1: st.metric("Baseline F1",   f"{bl_f1:.4f}")
        with p2: st.metric("Production F1", f"{perf['f1_macro']:.4f}", delta=f"{perf['f1_macro']-bl_f1:+.4f}")
        with p3: st.metric("Accuracy",      f"{perf.get('accuracy',0):.4f}")
        with st.expander("Classification Report"):
            st.code(perf.get("classification_report","Not available"))
    else:
        st.markdown('<div class="info-box">ℹ️ Upload a model + labelled data to see performance metrics.</div>', unsafe_allow_html=True)

    st.markdown("### ♻️ Retraining Log")
    if retrain_log:
        st.dataframe(pd.DataFrame(retrain_log), use_container_width=True)
    else:
        st.info("No retraining events this session.")

    st.markdown("---")
    report_dict = report.to_dict()
    report_dict["performance"] = perf
    report_dict["baseline"]    = baseline
    st.download_button("⬇ Download Drift Report (JSON)",
        data=json.dumps(report_dict, indent=2, default=str),
        file_name="drift_report.json", mime="application/json")
    score_csv = report.drift_scores.reset_index()
    score_csv.columns = ["feature","drift_score"]
    st.download_button("⬇ Download Feature Scores (CSV)",
        data=score_csv.to_csv(index=False),
        file_name="feature_drift_scores.csv", mime="text/csv")


def render_howto():
    st.markdown("""
## 📖 How to Use This Tool

This is a **generic ML drift monitoring system**. Plug in your own data and model — no code changes needed.

---

### Step 1 — Prepare your CSV files

- **Reference CSV** = your training dataset
- **Production CSV** = new data from deployment
- Both must have the same column names
- Label column is optional

### Step 2 — Upload your model (optional)

Supported: `.pkl` / `.joblib` (sklearn) · `.cbm` (CatBoost)

### Step 3 — Configure thresholds

| Threshold | Default | Meaning |
|-----------|---------|---------|
| KS statistic | 0.10 | Distribution distance |
| PSI | 0.20 | Stability index |
| F1 drop | 0.05 | Accuracy degradation |
| Drift fraction | 30% | % features before alert |

### Step 4 — Click Run Analysis

The system auto-detects column types, runs all 3 tests, scores features 0–1, and triggers retraining if needed.

---

### Example config.yaml

```yaml
data:
  reference_path:  "data/train.csv"
  production_path: "data/prod.csv"
  label_column:    "churn"
model:
  path: "models/my_model.pkl"
thresholds:
  ks_statistic: 0.10
  psi: 0.20
  f1_drop: 0.05
retraining:
  enabled: true
```

Run via CLI: `python main.py --config config/my_config.yaml`

---

### Supported use cases

| Use case | Reference | Production | Model | Labels |
|----------|-----------|------------|-------|--------|
| Pure data drift check | ✅ | ✅ | ❌ | ❌ |
| Data + performance drift | ✅ | ✅ | ✅ | ✅ |
| Monitoring without labels | ✅ | ✅ | ✅ | ❌ |
| Auto-retraining | ✅ | ✅ | ✅ | ✅ |
    """)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 ML Drift Monitor")
    st.caption("Plug in any model + dataset. Get instant drift analysis.")

    raw_config, run_btn = render_sidebar()

    tab_dashboard, tab_howto, tab_explainer = st.tabs([
        "📊 Dashboard", "📖 How to Use", "📖 Project Explainer"
    ])

    # ── TAB 1: DASHBOARD ──────────────────────────────────────────────────
    with tab_dashboard:
        if run_btn:
            ref_path  = (raw_config.get("data") or {}).get("reference_path")
            prod_path = (raw_config.get("data") or {}).get("production_path")
            if not ref_path or not prod_path:
                st.error("⚠️ Please upload both a Reference and Production CSV before running.")
            else:
                try:
                    config = config_from_dict(raw_config)
                    run_analysis(config)
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

        if st.session_state.analysis_run:
            render_dashboard()
        else:
            st.markdown('<div class="info-box">👈 Configure your datasets in the sidebar, then click <b>Run Analysis</b>.</div>', unsafe_allow_html=True)

            st.markdown("#### No data? Try the sample dataset:")
            sample_ref  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "reference.csv")
            sample_prod = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "production.csv")
            if os.path.exists(sample_ref) and os.path.exists(sample_prod):
                col1, col2 = st.columns(2)
                with col1:
                    with open(sample_ref, "rb") as f:
                        st.download_button("Download reference.csv", data=f, file_name="reference.csv", mime="text/csv", use_container_width=True)
                with col2:
                    with open(sample_prod, "rb") as f:
                        st.download_button("Download production.csv", data=f, file_name="production.csv", mime="text/csv", use_container_width=True)
                st.caption("Upload both files in the sidebar, set label column to **Churn**, then click Run Analysis.")

            st.markdown("---")
            st.markdown("#### What this tool does:")




          
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown("📊 **KS Test**\n\nNumerical distribution shift")
            with c2: st.markdown("📋 **Chi-Square**\n\nCategorical frequency shift")
            with c3: st.markdown("📐 **PSI**\n\nDrift magnitude (industry standard)")
            with c4: st.markdown("♻️ **Auto-Retrain**\n\nRetrains when drift detected")

    # ── TAB 2: HOW TO USE ─────────────────────────────────────────────────
    with tab_howto:
        render_howto()

    # ── TAB 3: PROJECT EXPLAINER ──────────────────────────────────────────
    with tab_explainer:
        explainer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ml_drift_explainer.html"
        )
        if os.path.exists(explainer_path):
            with open(explainer_path, "r") as f:
                html_content = f.read()
            components.html(html_content, height=900, scrolling=True)
        else:
            st.warning("⚠️ ml_drift_explainer.html not found in project root.")
            st.info("Copy ml_drift_explainer.html into the ml_drift_monitor/ folder.")


if __name__ == "__main__":
    main()
