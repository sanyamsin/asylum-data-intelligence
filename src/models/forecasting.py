"""
forecasting.py
--------------
Forecasting des flux d'asile avec ARIMA et Prophet.

Modèles :
  - ARIMA  : court terme (1-6 mois), sélection automatique via auto_arima
  - Prophet: moyen terme (6-18 mois), avec marqueurs d'événements

Évaluation :
  - Rolling-window cross-validation
  - Métriques : MAE, RMSE, MAPE
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data class résultat
# ------------------------------------------------------------------

@dataclass
class ForecastResult:
    country:          str
    model_name:       str
    forecast_periods: int
    forecast_df:      pd.DataFrame   # ds, yhat, yhat_lower, yhat_upper
    metrics:          Dict[str, float]
    notes:            str = ""


# ------------------------------------------------------------------
# Helpers métriques
# ------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calcule MAE, RMSE, MAPE."""
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) if mask.any() else np.nan
    return {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 4),
    }


# ------------------------------------------------------------------
# ARIMA Forecaster
# ------------------------------------------------------------------

class ARIMAForecaster:
    """
    Forecasting court terme avec auto-ARIMA (pmdarima).
    Sélection automatique de l'ordre (p,d,q) via AIC.
    """

    def __init__(self, seasonal: bool = True, seasonal_m: int = 12):
        self.seasonal   = seasonal
        self.seasonal_m = seasonal_m

    def fit_and_forecast(
        self,
        series:           pd.Series,
        country:          str,
        forecast_periods: int = 6,
        test_size:        int = 12,
    ) -> ForecastResult:
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("Install pmdarima: pip install pmdarima")

        if len(series) < test_size + 12:
            raise ValueError(f"Series too short (need > {test_size + 12} obs).")

        # Store original dates
        original_index = pd.to_datetime(series.index)

        # Interpolation
        series_vals = series.interpolate(method='linear').values

        # Split train / test (numeric arrays)
        train_vals = series_vals[:-test_size]
        test_vals  = series_vals[-test_size:]

        logger.info(f"[{country}] Fitting auto-ARIMA on {len(train_vals)} months...")

        model = pm.auto_arima(
            train_vals,
            seasonal=self.seasonal,
            m=self.seasonal_m,
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic',
        )

        # Backtest
        backtest_raw   = model.predict(n_periods=test_size)
        backtest_preds = np.array(backtest_raw).flatten()
        metrics        = compute_metrics(test_vals, backtest_preds)
        logger.info(f"[{country}] Backtest — MAE: {metrics['MAE']:.1f} | MAPE: {metrics['MAPE']:.1%}")

        # Refit on full series
        model.update(series_vals[-test_size:])

        # Forecast future (numeric)
        future_raw   = model.predict(n_periods=forecast_periods)
        future_preds = np.array(future_raw).flatten()

        # Reconstruct future dates from original index
        # Pure Python date arithmetic — avoids all pandas timestamp issues
        last_dt = pd.Timestamp(str(original_index[-1])[:7] + "-01")
        y, m = last_dt.year, last_dt.month
        future_list = []
        for _ in range(forecast_periods):
            m += 1
            if m > 12:
                m = 1
                y += 1
            future_list.append(pd.Timestamp(year=y, month=m, day=1))
        future_dates = pd.DatetimeIndex(future_list)

        # Confidence intervals based on MAE
        mae_val  = float(metrics["MAE"])
        forecast_df = pd.DataFrame({
            "ds":         future_dates,
            "yhat":       future_preds,
            "yhat_lower": future_preds - 1.96 * mae_val,
            "yhat_upper": future_preds + 1.96 * mae_val,
        })

        return ForecastResult(
            country=country,
            model_name=f"ARIMA{model.order}×{model.seasonal_order}",
            forecast_periods=forecast_periods,
            forecast_df=forecast_df,
            metrics=metrics,
        )


# ------------------------------------------------------------------
# Prophet Forecaster
# ------------------------------------------------------------------

class ProphetForecaster:
    """
    Forecasting moyen terme avec Prophet.
    Supporte les marqueurs d'événements (crises, politiques EU).
    """

    # Événements clés asylum par défaut
    DEFAULT_EVENTS = [
        {"ds": "2022-02-01", "holiday": "Ukraine_conflict_onset"},
        {"ds": "2021-08-01", "holiday": "Afghanistan_Kabul_fall"},
        {"ds": "2020-03-01", "holiday": "COVID_borders_closure"},
        {"ds": "2015-09-01", "holiday": "Syria_crisis_peak"},
    ]

    def __init__(self, policy_events: Optional[List[dict]] = None):
        events = policy_events or self.DEFAULT_EVENTS
        self.policy_events = pd.DataFrame(events)

    def fit_and_forecast(
        self,
        series:           pd.Series,
        country:          str,
        forecast_periods: int = 12,
        test_size:        int = 12,
    ) -> ForecastResult:
        """
        Entraîne Prophet et génère un forecast avec intervalles.
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Installe prophet : pip install prophet")

        # Préparer format Prophet
        df_p = series.reset_index()
        df_p.columns = ["ds", "y"]
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        df_p = df_p.interpolate(method='linear')

        train = df_p.iloc[:-test_size]
        test  = df_p.iloc[-test_size:]

        logger.info(f"[{country}] Fitting Prophet sur {len(train)} mois...")

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=self.policy_events,
            interval_width=0.90,
            changepoint_prior_scale=0.05,
        )
        model.fit(train)

        # Backtest
        future_bt  = model.make_future_dataframe(periods=test_size, freq="MS")
        forecast_bt = model.predict(future_bt)
        test_preds  = forecast_bt.tail(test_size)["yhat"].values
        metrics     = compute_metrics(test["y"].values, test_preds)
        logger.info(f"[{country}] Backtest — MAE: {metrics['MAE']:.1f} | "
                    f"MAPE: {metrics['MAPE']:.1%}")

        # Refit sur données complètes
        model_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=self.policy_events,
            interval_width=0.90,
            changepoint_prior_scale=0.05,
        )
        model_full.fit(df_p)
        future      = model_full.make_future_dataframe(periods=forecast_periods, freq="MS")
        forecast    = model_full.predict(future)
        forecast_df = forecast.tail(forecast_periods)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].reset_index(drop=True)

        return ForecastResult(
            country=country,
            model_name="Prophet",
            forecast_periods=forecast_periods,
            forecast_df=forecast_df,
            metrics=metrics,
        )


# ------------------------------------------------------------------
# Comparaison de modèles
# ------------------------------------------------------------------

def compare_models(results: List[ForecastResult]) -> pd.DataFrame:
    """
    Tableau comparatif des métriques de backtest.
    Utile pour sélectionner le meilleur modèle par pays.
    """
    rows = []
    for r in results:
        rows.append({
            "country": r.country,
            "model":   r.model_name,
            **r.metrics,
        })
    return pd.DataFrame(rows).sort_values(["country", "MAE"])