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
        """
        Entraîne auto-ARIMA et génère un forecast avec intervalles.

        Parameters
        ----------
        series           : Série mensuelle (DatetimeIndex)
        country          : Label pays pour le rapport
        forecast_periods : Nombre de mois à prévoir
        test_size        : Mois de hold-out pour évaluation
        """
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("Installe pmdarima : pip install pmdarima")

        if len(series) < test_size + 12:
            raise ValueError(f"Série trop courte (besoin > {test_size + 12} obs).")

        # Interpolation des NaN
        series = series.interpolate(method='linear')

        # Split train / test
        train = series.iloc[:-test_size]
        test  = series.iloc[-test_size:]

        logger.info(f"[{country}] Fitting auto-ARIMA sur {len(train)} mois...")

        model = pm.auto_arima(
            train,
            seasonal=self.seasonal,
            m=self.seasonal_m,
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic',
        )

        # Backtest
        backtest_preds = model.predict(n_periods=test_size)
        metrics = compute_metrics(test.values, backtest_preds)
        logger.info(f"[{country}] Backtest — MAE: {metrics['MAE']:.1f} | "
                    f"MAPE: {metrics['MAPE']:.1%}")

        # Refit sur série complète
        model.update(test)

        # Forecast futur
        future_preds, conf_int = model.predict(
            n_periods=forecast_periods, return_conf_int=True
        )
        last_date    = series.index[-1]
        future_dates = pd.date_range(
            start=last_date, periods=forecast_periods + 1, freq="MS"
        )[1:]

        forecast_df = pd.DataFrame({
            "ds":          future_dates,
            "yhat":        future_preds,
            "yhat_lower":  conf_int[:, 0],
            "yhat_upper":  conf_int[:, 1],
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