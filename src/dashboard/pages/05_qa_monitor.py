"""
05_qa_monitor.py
----------------
QA Monitor page — data quality dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.processing.validation import AsylumDataValidator

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    base      = Path("data/processed")
    df        = pd.read_csv(base / "applications_clean.csv")
    qa_report = None
    qa_path   = base / "qa_report_applications.json"
    if qa_path.exists():
        with open(qa_path) as f:
            qa_report = json.load(f)
    return df, qa_report

df_app, qa_report = load_data()

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------

st.markdown("## ✅ Data Quality Monitor")

st.info("""
Three-tier QA framework applied to all asylum datasets.
Results below reflect the last pipeline run.
Click **Re-run QA** to refresh with latest data.
""")

# ------------------------------------------------------------------
# Re-run QA
# ------------------------------------------------------------------

if st.button("🔄 Re-run QA", type="primary"):
    with st.spinner("Running QA checks..."):
        validator = AsylumDataValidator(
            df=df_app,
            dataset_name="Eurostat Applications (migr_asyappctzm)",
            required_columns=["time", "geo", "citizen", "sex", "age", "value"],
            time_column="time", value_column="value", geo_column="geo",
        )
        report    = validator.run_all()
        qa_report = json.loads(report.to_json())
        st.success("QA completed successfully.")

# ------------------------------------------------------------------
# Overall status
# ------------------------------------------------------------------

if qa_report:
    summary = qa_report.get("summary", {})
    overall = summary.get("overall", "UNKNOWN")
    icon    = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}.get(overall, "⚪")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Status", f"{icon} {overall}")
    with col2:
        st.metric("PASS", summary.get("PASS", 0))
    with col3:
        st.metric("WARN", summary.get("WARN", 0))
    with col4:
        st.metric("FAIL", summary.get("FAIL", 0))

    st.markdown(f"""
    **Dataset**: {qa_report.get('dataset', 'N/A')} |
    **Rows**: {qa_report.get('total_rows', 0):,} |
    **Run**: {qa_report.get('run_timestamp', 'N/A')[:19]}
    """)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Checks detail
    # ------------------------------------------------------------------

    st.markdown("### QA Checks Detail")

    tier_labels = {
        1: "Tier 1 — Structural",
        2: "Tier 2 — Statistical Coherence",
        3: "Tier 3 — Timeliness",
    }
    checks = qa_report.get("checks", [])

    for tier_num, tier_label in tier_labels.items():
        tier_checks = [c for c in checks if c["tier"] == tier_num]
        if not tier_checks:
            continue
        st.markdown(f"#### {tier_label}")
        for check in tier_checks:
            status = check["status"]
            icon   = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "?")
            with st.expander(f"{icon} {check['check_name']} — {status}"):
                st.markdown(f"**Message**: {check['message']}")
                if check.get("affected_rows"):
                    st.markdown(f"**Affected rows**: {check['affected_rows']:,}")
                if check.get("details"):
                    st.json(check["details"])

    # ------------------------------------------------------------------
    # Completeness chart
    # ------------------------------------------------------------------

    st.markdown("---")
    st.markdown("### Completeness by Country")

    completeness = (
        df_app.groupby("geo")["value"]
        .apply(lambda x: x.notna().mean())
        .reset_index()
    )
    completeness.columns = ["Country", "Completeness"]
    completeness = completeness.sort_values("Completeness", ascending=True)

    fig = go.Figure(go.Bar(
        x=completeness["Completeness"] * 100,
        y=completeness["Country"],
        orientation="h",
        marker_color=[
            "#28a745" if v >= 0.95 else
            "#ffc107" if v >= 0.80 else "#dc3545"
            for v in completeness["Completeness"]
        ],
        text=[f"{v:.1%}" for v in completeness["Completeness"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=320, plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   title="Completeness (%)", range=[0, 110]),
        yaxis=dict(showgrid=False),
    )
    fig.add_vline(x=95, line_dash="dash", line_color="gray",
                  annotation_text="95% threshold")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------

    st.markdown("### Data Freshness")

    latest_period = df_app["time"].max()
    lag_days      = (datetime.now() -
                     pd.Period(latest_period, freq="M").to_timestamp("M")).days

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Period", latest_period)
    with col2:
        status = "✅ Within threshold" if lag_days <= 60 else "⚠️ Exceeds threshold"
        st.metric("Reporting Lag", f"{lag_days} days",
                  delta=status,
                  delta_color="normal" if lag_days <= 60 else "inverse")

else:
    st.warning("No QA report found. Click **Re-run QA** to generate one.")