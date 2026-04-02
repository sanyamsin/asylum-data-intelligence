"""
02_forecasting.py
-----------------
Forecasting page — ARIMA short-term forecast per country.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.trend_analysis import extract_monthly_series
from src.models.forecasting import ARIMAForecaster

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

@st.cache_data(ttl=0)
def load_data():
    return pd.read_csv("data/processed/applications_clean.csv")

df_app = load_data()

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------

st.markdown("## 📈 Asylum Flow Forecasting — ARIMA Model")

st.info("""
**Model**: ARIMA(0,1,0)×(1,0,0,12) — selected over Prophet based on
12-month backtest (MAPE: 26%). Appropriate for short series with
structural breaks (COVID-19, Afghan surge, Ukraine conflict).
""")

# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    countries = sorted(df_app["geo"].dropna().unique())
    country   = st.selectbox("Select Member State", countries, index=0)
with col2:
    horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)

# ------------------------------------------------------------------
# Run forecast
# ------------------------------------------------------------------

if st.button("Run Forecast", type="primary"):
    with st.spinner(f"Fitting ARIMA for {country}..."):
        try:
            series     = extract_monthly_series(df_app, country=country)
            forecaster = ARIMAForecaster()
            result     = forecaster.fit_and_forecast(
                series, country=country,
                forecast_periods=horizon, test_size=12,
            )

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", result.model_name)
            with col2:
                st.metric("MAE", f"{result.metrics['MAE']:,.0f}")
            with col3:
                st.metric("MAPE", f"{result.metrics['MAPE']:.1%}")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name="Historical",
                line=dict(color="#003399", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=result.forecast_df["ds"],
                y=result.forecast_df["yhat"],
                name="Forecast",
                line=dict(color="#dc3545", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([result.forecast_df["ds"],
                             result.forecast_df["ds"].iloc[::-1]]),
                y=pd.concat([result.forecast_df["yhat_upper"],
                             result.forecast_df["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(220,53,69,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            ))
            
            fig.update_layout(
                title=f"Asylum Applications Forecast — {country} ({horizon} months)",
                height=420, plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Period"),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Applications"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            st.markdown("### Forecast Values")
            df_display = result.forecast_df.copy()
            df_display["ds"]         = df_display["ds"].astype(str).str[:7]
            df_display["yhat"]       = df_display["yhat"].clip(0).round(0).astype(int)
            df_display["yhat_lower"] = df_display["yhat_lower"].clip(0).round(0).astype(int)
            df_display["yhat_upper"] = df_display["yhat_upper"].round(0).astype(int)
            df_display.columns       = ["Period", "Forecast", "Lower (95%)", "Upper (95%)"]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.caption("""
            Forecast disclaimer: confidence intervals widen beyond 3 months.
            MAPE of 26% reflects intrinsic volatility of asylum flows.
            Do not use as sole basis for operational decisions.
            """)

        except Exception as e:
            import traceback
            st.error(f"Forecast failed: {e}")
            st.code(traceback.format_exc())
else:
    st.markdown("""
    Select a **Member State** and **forecast horizon**,
    then click **Run Forecast** to generate ARIMA predictions.
    """)