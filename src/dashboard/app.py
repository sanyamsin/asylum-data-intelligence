"""
app.py
------
Main Streamlit dashboard entry point.
Asylum Data Intelligence System — EUAA-aligned operational dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Asylum Data Intelligence | EUAA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003399 0%, #0052cc 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .main-header h1 { font-size: 24px; font-weight: bold; margin: 0; }
    .main-header p  { font-size: 13px; opacity: 0.85; margin: 5px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🛡️ Asylum Data Intelligence System</h1>
    <p>EUAA-Aligned Statistical Platform | Eurostat + UNHCR Data |
    9 EU Member States | 2020–2026</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/320px-Flag_of_Europe.svg.png",
    width=120,
)
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("---")

pages = {
    "📊 Overview":            "pages/01_overview.py",
    "📈 Forecasting":         "pages/02_forecasting.py",
    "⚖️  Decisions":          "pages/03_decisions.py",
    "🔍 Anomaly Detection":   "pages/04_anomaly.py",
    "✅ QA Monitor":          "pages/05_qa_monitor.py",
}

page = st.sidebar.radio("Select page", list(pages.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Sources**
- Eurostat `migr_asyappctzm`
- Eurostat `migr_asydcfsta`
- UNHCR Refugee Data Finder

**Model**: ARIMA(0,1,0)×(1,0,0,12)
**MAPE**: 26% (12-month backtest)

**GitHub**:
[asylum-data-intelligence](https://github.com/sanyamsin/asylum-data-intelligence)
""")

# ------------------------------------------------------------------
# Page routing
# ------------------------------------------------------------------

import importlib.util

page_file = Path(__file__).parent / pages[page]

if page_file.exists():
    spec   = importlib.util.spec_from_file_location("page", page_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    st.info(f"Page under construction: {page}")