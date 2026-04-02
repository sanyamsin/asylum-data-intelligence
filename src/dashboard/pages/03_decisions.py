"""
03_decisions.py
---------------
First instance decisions page — recognition rates, rejection rates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/decisions_clean.csv")

df_dec = load_data()

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------

st.markdown("## ⚖️ First Instance Decisions")

df_total = df_dec[
    (df_dec["sex"].astype(str) == "Total") &
    (df_dec["age"].astype(str) == "Total")
].copy()

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    countries = sorted(df_total["geo"].dropna().unique())
    selected  = st.multiselect("Member States", countries, default=countries[:5])
with col2:
    # Extract year from time column
    df_total["year"] = df_total["time"].astype(int)
    years      = sorted(df_total["year"].dropna().unique().astype(int))
    year_range = st.select_slider(
        "Year range", options=years if years else [2020, 2025],
        value=(years[0], years[-1]) if years else (2020, 2025)
    )

# Extract year from time column if not present
if "year" not in df_total.columns:
    df_total["year"] = pd.to_datetime(
        df_total["time"], format="%Y-%m", errors="coerce"
    ).dt.year

df_filtered = df_total[
    df_total["geo"].isin(selected) &
    df_total["year"].between(year_range[0], year_range[1])
].copy()

# Debug info
st.caption(f"Filtered rows: {len(df_filtered):,} | Decisions: {df_filtered['decision'].unique().tolist() if 'decision' in df_filtered.columns else 'N/A'}")

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------

df_pivot = (
    df_filtered
    .groupby(["geo", "decision"])["value"]
    .sum().unstack(fill_value=0).reset_index()
)

cols_list = df_pivot.columns.tolist()
total_col = "Total" if "Total" in cols_list else (cols_list[1] if len(cols_list) > 1 else None)
if total_col is None:
    st.warning("No decision data available for selected filters.")
    st.stop()
positive_col = [c for c in df_pivot.columns if c == "Positive decision"]
positive_col = positive_col[0] if positive_col else None

if positive_col and total_col in df_pivot.columns:
    total_decisions = df_pivot[total_col].sum()
    total_positive  = df_pivot[positive_col].sum()
    avg_recognition = total_positive / total_decisions if total_decisions > 0 else 0
    negative_col    = "Negative decision"
    total_negative  = df_pivot[negative_col].sum() if negative_col in df_pivot.columns else 0
    avg_rejection   = total_negative / total_decisions if total_decisions > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Decisions", f"{total_decisions:,.0f}")
    with col2:
        st.metric("Recognition Rate", f"{avg_recognition:.1%}")
    with col3:
        st.metric("Rejection Rate", f"{avg_rejection:.1%}")
    with col4:
        st.metric("Countries", len(selected))

st.markdown("---")

# ------------------------------------------------------------------
# Chart 1: Recognition rates
# ------------------------------------------------------------------

st.markdown("### Recognition & Rejection Rates by Member State")

if positive_col and total_col in df_pivot.columns:
    df_rates = df_pivot.copy()
    df_rates["recognition_rate"] = (
        df_rates[positive_col] / df_rates[total_col].replace(0, float("nan"))
    )
    df_rates["rejection_rate"] = (
        df_rates.get("Negative decision", 0) /
        df_rates[total_col].replace(0, float("nan"))
    )
    df_rates = df_rates.sort_values("recognition_rate", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_rates["geo"], x=df_rates["recognition_rate"] * 100,
        name="Recognition rate", orientation="h",
        marker_color="#003399", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=df_rates["geo"], x=df_rates["rejection_rate"] * 100,
        name="Rejection rate", orientation="h",
        marker_color="#dc3545", opacity=0.85,
    ))
    fig.update_layout(
        barmode="group", height=380, plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   title="Rate (%)", ticksuffix="%"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Chart 2: Decision types
# ------------------------------------------------------------------

st.markdown("### Decision Types Distribution")

decision_cols = [c for c in df_pivot.columns
                 if c not in ["geo", "Total"] and "Positive" in c]

if decision_cols:
    df_melt = df_filtered[
        df_filtered["decision"].isin(decision_cols + ["Negative decision"])
    ].groupby(["geo", "decision"])["value"].sum().reset_index()

    fig2 = px.bar(
        df_melt, x="geo", y="value", color="decision",
        labels={"value": "Decisions", "geo": "Country", "decision": "Type"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode="stack",
    )
    fig2.update_layout(
        height=350, plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Data table
# ------------------------------------------------------------------

st.markdown("### Detailed Rates Table")

if positive_col and total_col in df_pivot.columns:
    df_table = df_rates[["geo", total_col, positive_col,
                          "recognition_rate", "rejection_rate"]].copy()
    df_table.columns = ["Country", "Total", "Positive",
                        "Recognition Rate", "Rejection Rate"]
    df_table["Recognition Rate"] = df_table["Recognition Rate"].map("{:.1%}".format)
    df_table["Rejection Rate"]   = df_table["Rejection Rate"].map("{:.1%}".format)
    df_table["Total"]            = df_table["Total"].map("{:,.0f}".format)
    df_table["Positive"]         = df_table["Positive"].map("{:,.0f}".format)
    st.dataframe(df_table, use_container_width=True, hide_index=True)