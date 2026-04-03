---
title: Asylum Data Intelligence
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: true
python_version: "3.12"
---
# Asylum Data Intelligence System
### EUAA-Aligned Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: Eurostat](https://img.shields.io/badge/Data-Eurostat%20API-orange.svg)](https://ec.europa.eu/eurostat)
[![Data: UNHCR](https://img.shields.io/badge/Data-UNHCR%20Refugee%20Data-blue.svg)](https://www.unhcr.org/refugee-statistics)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-lightgrey.svg)](.github/workflows/)

---

##  Overview

**Asylum Data Intelligence System** is a production-grade analytical platform designed
to support **information and data management functions** in the asylum and reception
sector — aligned with the operational standards and reporting requirements of the
**European Union Agency for Asylum (EUAA)**.

The system integrates multi-source asylum data (Eurostat, UNHCR), applies rigorous
**statistical modelling** (trend analysis, forecasting, anomaly detection), automates
**quality assurance** of datasets and reports, and delivers **operational dashboards**
and **high-level analytical reports** for institutional decision-makers.

> *Built as a demonstration of senior-level data management and statistical expertise
> in the humanitarian and asylum sector.*

---

##  Key Features

| Module | Description | Methods |
|--------|-------------|---------|
| **Data Ingestion** | Automated pipelines from Eurostat & UNHCR | REST APIs, SDMX-JSON |
| **Statistical Modelling** | Asylum flow forecasting, trend decomposition | ARIMA, Prophet, STL |
| **Anomaly Detection** | Identification of irregular patterns | Isolation Forest, Z-score |
| **Quality Assurance** | Automated data validation (3 tiers) | Rule-based + statistical |
| **Reporting Engine** | Periodic & ad hoc institutional reports | Jinja2, WeasyPrint |
| **Operational Dashboard** | Real-time KPI monitoring | Streamlit, Plotly |

---

##  Key Results

### Forecasting Performance — Germany (Pilot Series, 2020–2026)

| Model | MAE | RMSE | MAPE | Status |
|-------|-----|------|------|--------|
| **ARIMA(0,1,0)×(1,0,0,12)** | 2,878 | 3,897 | **26%** |  Selected |
| Prophet | 10,087 | 10,471 | 126% |  Rejected |

**Key finding**: ARIMA outperforms Prophet on this dataset due to three major
structural breaks (COVID-19, Kabul fall, Ukraine conflict) over a short 74-month
series. See [methodology.md](docs/methodology.md) for full analytical justification.

### QA Framework Results — Asylum Applications Dataset

| Tier | Checks | Result |
|------|--------|--------|
| Tier 1 — Structural | 3 checks | ✅ 2 PASS / 1 WARN |
| Tier 2 — Statistical | 3 checks | ✅ 2 PASS / 1 WARN |
| Tier 3 — Timeliness | 2 checks | ✅ 2 PASS |
| **Overall** | **8 checks** | **⚠️ WARN — 0 FAIL** |

Data freshness: **32 days lag** (within 60-day threshold) ✅

### Anomaly Detection Results — Asylum Applications Dataset

| Method | Rows Analysed | Anomalies | Rate |
|--------|--------------|-----------|------|
| Z-score (threshold 3.5) | 18,840 | 312 | 1.7% |
| Isolation Forest (5%) | 664 | 34 | 5.1% |
| **Classification** | — | **17 genuine events / 154 to review** | — |

**Key findings**:
- Polish & Swedish surges (Aug–Dec 2021) confirmed as genuine operational events — Afghan surge post-Kabul (Z-score up to 19.5)
- German record volumes (2022–2025) corroborated as genuine — Ukraine TPR regime
- 2020 Q2 anomalies linked to COVID border closures — flagged for MS verification

See [methodology.md](docs/methodology.md) for full classification details.
---

##  Project Structure
```
asylum-data-intelligence/
│
├── .github/workflows/
│   ├── ci.yml                   # Tests & linting on push
│   └── data_refresh.yml         # Scheduled weekly data update
│
├── data/
│   ├── raw/                     # Unprocessed source data
│   ├── processed/               # Cleaned, validated datasets
│   └── README.md                # Data dictionary & sources
│
├── src/
│   ├── ingestion/
│   │   ├── eurostat_client.py   # Eurostat SDMX-JSON API client
│   │   ├── unhcr_client.py      # UNHCR Refugee Data Finder
│   │   └── pipeline.py          # Orchestration
│   │
│   ├── processing/
│   │   ├── cleaning.py          # Standardisation, deduplication
│   │   └── validation.py        # QA: 3-tier framework
│   │
│   └── models/
│       ├── trend_analysis.py    # STL decomposition, correlations
│       └── forecasting.py       # ARIMA & Prophet models
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── tests/
│   └── test_ingestion.py        # Unit tests (9/9 passing)
│
├── reports/                     # Generated visualisations
├── docs/
│   └── methodology.md           # Statistical methodology
│
├── requirements.txt
└── README.md
```

---

## 📊 Statistical Modelling Approach

### Forecasting
- **ARIMA** — short-term (1–6 months), automatic order selection via AIC
- **Prophet** — medium-term decomposition with policy event markers
- **Cross-validation** — 12-month rolling hold-out; MAE, RMSE, MAPE reported

### Quality Assurance (3-Tier Framework)
```
Tier 1 — Structural Validation
  ✓ Schema conformity
  ✓ Completeness score
  ✓ Duplicate detection

Tier 2 — Statistical Coherence
  ✓ Negative value check
  ✓ Z-score outlier detection (threshold: 3.5)
  ✓ Month-over-month variation (threshold: ×5)

Tier 3 — Timeliness Monitoring
  ✓ Time series continuity
  ✓ Data freshness (threshold: 60 days)
```

---

## 📈 Data Sources

| Source | Dataset | Frequency | Access |
|--------|---------|-----------|--------|
| **Eurostat** | Asylum applications (migr_asyappctzm) | Monthly | Open API |
| **Eurostat** | First instance decisions (migr_asydcfsta) | Monthly | Open API |
| **UNHCR** | Global refugee trends | Annual | Open API |

---

## ⚠️ Methodological Notes

**Geographic scope**: 9 EU Member States (DE, FR, IT, ES, AT, BE, NL, SE, PL).

**Greece exclusion**: Eurostat API constraint (HTTP 413) for the selected
period × nationality combination. Greece is a major entry point on the
Eastern Mediterranean Route — its exclusion underestimates EU entry flows,
particularly for Afghan and Syrian nationals. Fix planned: Phase 2 Sprint 2.

---

## 🚀 Getting Started
```bash
# Clone
git clone https://github.com/sanyamsin/asylum-data-intelligence.git
cd asylum-data-intelligence

# Environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# Install
pip install -r requirements.txt

# Run pipeline
python -m src.ingestion.pipeline

# Run QA
python -m src.processing.validation

# Launch notebook
jupyter notebook notebooks/
```

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

| Phase | Status |
|-------|--------|
| Phase 1 — Core Infrastructure | ✅ Complete |
| Phase 2 — Data Processing & QA | ✅ Complete |
| Phase 3 — Statistical Modelling | 🔄 In Progress |
| Phase 4 — Anomaly Detection | ⬜ Planned |
| Phase 5 — Reporting Engine | ⬜ Planned |
| Phase 6 — Operational Dashboard | ⬜ Planned |

---

## 🧑‍💻 Author

**Serge-Alain NYAMSIN**
Data Science & AI | Humanitarian & Development Cooperation
MSc Data Science & AI — DSTI Paris
12+ years field experience | ACF · CRF · Handicap International
West & Central Africa | IRAM/PDD-CAR | VAINCRE II/Mauritanie

[![GitHub](https://img.shields.io/badge/GitHub-sanyamsin-black?logo=github)](https://github.com/sanyamsin)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Lokozu-yellow?logo=huggingface)](https://huggingface.co/spaces/Lokozu/asylum-data-intelligence)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/serge-alain-nyamsin)
[![Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-red?logo=streamlit)](https://huggingface.co/spaces/Lokozu/asylum-data-intelligence)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.