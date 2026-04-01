# Statistical Methodology — Asylum Data Intelligence System

## 1. Forecasting Framework

### 1.1 ARIMA / SARIMA
Applied for **short-term forecasting** (1–6 months) at Member State level.

- Model selection via `auto_arima` (AIC minimisation, stepwise search)
- Seasonal component (SARIMA) enabled — strong annual seasonality in asylum applications
- 95% prediction intervals reported
- **Evaluation**: 12-month rolling hold-out cross-validation; MAE, RMSE, MAPE reported

### 1.2 Prophet
Applied for **medium-term forecasting** (6–18 months) and decomposition analysis.

- Additive model: trend + yearly seasonality + holiday effects
- **Policy event markers**: EU policy changes, conflict onset dates, COVID borders closure
- 90% uncertainty intervals
- Same 12-month hold-out as ARIMA for direct comparison

---

## 2. Trend & Decomposition Analysis

### 2.1 STL Decomposition
Seasonal-Trend decomposition using Loess applied per country series:
- **Trend**: structural shifts in application volumes
- **Seasonal**: recurring monthly patterns (typically: spring/summer peaks)
- **Residual**: irregular component, input to anomaly detection

### 2.2 Year-over-Year Analysis
Rolling 12-month sum compared across years; percentage change and absolute delta reported.
Aligned with EUAA Asylum Report publication methodology.

---

## 3. Anomaly Detection

### 3.1 Z-score monitoring (univariate)
- Per country series: flag values where |Z| > 3.5
- Primary use: **data quality** (Tier 2 QA) — sudden spikes may indicate reporting errors

### 3.2 Isolation Forest (multivariate)
- Features: [applications_volume, decision_rate, recognition_rate, MoM_change]
- Contamination parameter: 0.05 (5% expected anomaly rate)
- Output: anomaly_score per observation

### 3.3 Error vs. Genuine Operational Anomaly
Anomalies classified by cross-referencing:
1. Anomaly present in multiple datasets simultaneously → genuine operational event
2. Anomaly isolated to a single dataset/country → likely reporting error

---

## 4. Quality Assurance Framework

### Tier 1 — Structural Validation
| Check | Method | Threshold |
|-------|--------|-----------|
| Schema conformity | Column presence | All required columns present |
| Value completeness | Non-null ratio | ≥ 95% (WARN below 80%) |
| Duplicate records | Exact match (time × geo) | 0 duplicates |

### Tier 2 — Statistical Coherence
| Check | Method | Threshold |
|-------|--------|-----------|
| Negative values | Direct sign check | 0 negative values |
| Outliers | Z-score | ≤ 3.5 |
| MoM variation | Ratio to prior month | ≤ ×5 |

### Tier 3 — Timeliness Monitoring
| Check | Method | Threshold |
|-------|--------|-----------|
| Time series gaps | Period range continuity | 0 missing months |
| Data freshness | Days since latest period | ≤ 60 days |

---

## 5. Data Sources

- Eurostat `migr_asyappctzm` — monthly asylum applications
- Eurostat `migr_asydcfsta` — first instance decisions
- UNHCR Refugee Data Finder — global annual trends

---

## 6. Model Selection

### 6.1 Comparison Protocol
Each model evaluated on a **12-month hold-out** (rolling-window backtest).
Reported metrics: MAE, RMSE, MAPE.

### 6.2 Results — Germany (Pilot Series, 74 months)

| Model | MAE | RMSE | MAPE | Status |
|-------|-----|------|------|--------|
| ARIMA(0,1,0)×(1,0,0,12) | 2,878 | 3,897 | **26%** | ✅ Selected |
| Prophet | 10,087 | 10,471 | 126% | ❌ Rejected |

### 6.3 Justification

The 2020–2026 dataset contains three major structural breaks
over a short 74-month series:

- **March 2020**: COVID-19 border closures
- **August 2021**: Fall of Kabul — Afghan surge
- **February 2022**: Ukraine conflict — TPR regime activated

Prophet requires longer series (5+ years) with stable seasonality.
On this short dataset with multiple breaks, it overfits and produces
a MAPE of 126% — operationally unusable.

ARIMA(0,1,0)×(1,0,0,12) is retained because it captures:
- Non-stationary trend via differencing (d=1)
- Annual seasonality (P=1, m=12) — recurring summer peaks

### 6.4 Operational Recommendations

| Horizon | Recommended Model | Confidence |
|---------|-------------------|------------|
| 1–3 months | ARIMA | High |
| 3–6 months | ARIMA | Moderate |
| 6–12 months | To be reassessed with longer series | Low |

### 6.5 Model Limitations

- MAPE of 26% reflects the intrinsic volatility of asylum flows
- Confidence intervals widen significantly beyond 3 months
- Model does not capture future exogenous shocks
- Prophet will be reassessed when series covers 10+ stable years

---

## 7. Scope & General Limitations

### 7.1 Geographic Coverage
9 EU Member States: Germany, France, Italy, Spain,
Austria, Belgium, Netherlands, Sweden, Poland.

**Greece exclusion**: Eurostat API constraint (HTTP 413 — payload too large)
for the selected period × nationality combination. Greece is a major entry
point on the Eastern Mediterranean Route. Its exclusion underestimates EU
entry flows, particularly for Afghan and Syrian nationals.
Fix planned: Phase 2 Sprint 2.

### 7.2 Nationality Coverage
10 nationalities: SY, AF, IQ, PK, NG, VE, CO, TN, MA, TR.
Applicants from other origins are not included in this scope.

### 7.3 References
- EUAA. *Asylum Report 2023.* European Union Agency for Asylum.
- Eurostat. *Asylum and managed migration statistics.*
- Box, G.E.P. et al. (2015). *Time Series Analysis: Forecasting and Control.*
- Taylor, S.J. & Letham, B. (2018). Forecasting at Scale. *The American Statistician.*
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice.*