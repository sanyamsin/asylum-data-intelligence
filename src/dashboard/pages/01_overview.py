"""
01_overview.py
--------------
Overview page — KPIs, applications by country, top nationalities.
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
    base = Path("data/processed")
    df_app   = pd.read_csv(base / "applications_clean.csv")
    df_dec   = pd.read_csv(base / "decisions_clean.csv")
    df_unhcr = pd.read_csv(base / "unhcr_trends_clean.csv")
    return df_app, df_dec, df_unhcr

df_app, df_dec, df_unhcr = load_data()
df_total = df_app[df_app["is_total"] == True].copy()

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

st.markdown("## 📊 Overview — Asylum Applications")

col1, col2 = st.columns(2)
with col1:
    countries = sorted(df_total["geo"].dropna().unique())
    selected_countries = st.multiselect(
        "Member States", countries, default=countries
    )
with col2:
    years = sorted(df_total["year"].dropna().unique().astype(int))
    selected_years = st.select_slider(
        "Year range", options=years,
        value=(min(years), max(years)),
    )

mask = (
    df_total["geo"].isin(selected_countries) &
    df_total["year"].between(selected_years[0], selected_years[1])
)
df_filtered = df_total[mask].copy()

# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------

total_apps      = df_filtered["value"].sum()
n_countries     = df_filtered["geo"].nunique()
n_nationalities = df_app["citizen"].nunique()
latest_period   = df_filtered["time"].max()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Applications", f"{total_apps:,.0f}")
with col2:
    st.metric("Member States", n_countries)
with col3:
    st.metric("Nationalities", n_nationalities)
with col4:
    st.metric("Latest Period", latest_period)

st.markdown("---")

# ------------------------------------------------------------------
# Chart 1: Monthly trend
# ------------------------------------------------------------------

st.markdown("### Monthly Trend by Country")

df_monthly = (
    df_filtered
    .groupby(["time", "geo"])["value"]
    .sum()
    .reset_index()
)
df_monthly["time"] = pd.to_datetime(df_monthly["time"], format="%Y-%m")

fig_trend = px.line(
    df_monthly, x="time", y="value", color="geo",
    labels={"time": "Period", "value": "Applications", "geo": "Country"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_trend.update_layout(
    height=380, plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
)
fig_trend.add_vrect(
    x0="2020-03-01", x1="2020-09-01",
    fillcolor="gray", opacity=0.1,
    annotation_text="COVID-19", annotation_position="top left"
)
fig_trend.add_vrect(
    x0="2022-02-01", x1="2022-06-01",
    fillcolor="orange", opacity=0.1,
    annotation_text="Ukraine", annotation_position="top left"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------------------------------
# Chart 2: Bar + Pie
# ------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Total by Member State")
    by_country = (
        df_filtered.groupby("geo")["value"]
        .sum().sort_values(ascending=True).reset_index()
    )
    fig_bar = px.bar(
        by_country, x="value", y="geo", orientation="h",
        labels={"value": "Applications", "geo": "Country"},
        color="value", color_continuous_scale="Blues",
    )
    fig_bar.update_layout(
        height=350, plot_bgcolor="white",
        coloraxis_showscale=False,
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Top 8 Nationalities")
    by_citizen = (
        df_filtered[df_filtered["citizen"].notna()]
        .groupby("citizen")["value"]
        .sum().sort_values(ascending=False).head(8).reset_index()
    )
    fig_pie = px.pie(
        by_citizen, values="value", names="citizen",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4,
    )
    fig_pie.update_layout(height=350)
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------------
# UNHCR Global Context
# ------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🌍 Global Context — UNHCR Trends")

df_unhcr["year"]     = df_unhcr["year"].astype(int)
df_unhcr_filtered    = df_unhcr[df_unhcr["year"] >= 2010]

fig_unhcr = go.Figure()
fig_unhcr.add_trace(go.Scatter(
    x=df_unhcr_filtered["year"],
    y=df_unhcr_filtered["refugees"] / 1e6,
    name="Refugees (M)",
    fill="tozeroy", fillcolor="rgba(0,51,153,0.1)",
    line=dict(color="#003399", width=2),
))
fig_unhcr.add_trace(go.Scatter(
    x=df_unhcr_filtered["year"],
    y=df_unhcr_filtered["asylum_seekers"] / 1e6,
    name="Asylum Seekers (M)",
    fill="tozeroy", fillcolor="rgba(220,53,69,0.1)",
    line=dict(color="#dc3545", width=2),
))
fig_unhcr.update_layout(
    height=280, plot_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
               title="Millions of persons"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_unhcr, use_container_width=True)