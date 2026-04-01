"""
trend_analysis.py
-----------------
Préparation des séries temporelles et analyse de tendances
pour le forecasting des flux d'asile.

Fonctions :
  - Extraction de séries mensuelles par pays
  - Décomposition STL (tendance, saisonnalité, résidus)
  - Corrélations entre pays
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Extraction de séries temporelles
# ------------------------------------------------------------------

def extract_monthly_series(
    df: pd.DataFrame,
    country: str,
    sex: str = "Total",
    age: str = "Total",
    applicant: str = "Total",
) -> pd.Series:
    """
    Extrait une série temporelle mensuelle pour un pays donné.
    Agrège toutes les nationalités disponibles.

    Parameters
    ----------
    df        : DataFrame nettoyé (applications_clean.csv)
    country   : Nom du pays ex. "Germany"
    sex       : "Total", "Males", "Females"
    age       : "Total"
    applicant : "Total", "First time applicant"
    """
    mask = (
        (df['geo'] == country) &
        (df['sex'] == sex) &
        (df['age'] == age) &
        (df['applicant'] == applicant)
    )

    subset = df[mask].copy()

    if subset.empty:
        logger.warning(f"Aucune donnée pour {country}")
        return pd.Series(dtype=float)

    # Agréger toutes nationalités par mois
    series = (
        subset
        .groupby('time')['value']
        .sum()
    )
    series.index = pd.to_datetime(series.index, format='%Y-%m')
    series = series.sort_index()
    series.name = country

    logger.info(f"Série extraite : {country} | {len(series)} mois | "
                f"{series.index.min().strftime('%Y-%m')} → "
                f"{series.index.max().strftime('%Y-%m')}")
    return series


def build_country_matrix(
    df: pd.DataFrame,
    countries: list,
) -> pd.DataFrame:
    """
    Construit une matrice pays × mois avec les totaux mensuels.

    Returns
    -------
    DataFrame avec DatetimeIndex et une colonne par pays
    """
    series_dict = {}
    for country in countries:
        s = extract_monthly_series(df, country=country)
        if not s.empty:
            series_dict[country] = s

    matrix = pd.DataFrame(series_dict)
    matrix.index.name = "date"
    return matrix


# ------------------------------------------------------------------
# Décomposition STL
# ------------------------------------------------------------------

def decompose_series(
    series: pd.Series,
    period: int = 12,
) -> dict:
    """
    Décomposition STL (Seasonal-Trend-Loess) d'une série mensuelle.

    Parameters
    ----------
    series : pd.Series avec DatetimeIndex
    period : Période saisonnière (12 pour mensuel)

    Returns
    -------
    dict avec clés : trend, seasonal, residual, series
    """
    if len(series) < period * 2:
        logger.warning(f"Série trop courte pour STL ({len(series)} obs)")
        return {}

    # Remplacer les NaN par interpolation linéaire
    series_filled = series.interpolate(method='linear')

    stl = STL(series_filled, period=period, robust=True)
    result = stl.fit()

    return {
        "series":   series_filled,
        "trend":    pd.Series(result.trend,    index=series.index),
        "seasonal": pd.Series(result.seasonal, index=series.index),
        "residual": pd.Series(result.resid,    index=series.index),
    }


# ------------------------------------------------------------------
# Corrélations entre pays
# ------------------------------------------------------------------

def compute_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les corrélations de Pearson entre séries pays.
    Utile pour identifier les pays avec des dynamiques similaires.
    """
    return matrix.corr(method='pearson').round(3)


# ------------------------------------------------------------------
# Entry point — test rapide
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from pathlib import Path
    df = pd.read_csv("data/processed/applications_clean.csv")

    # Liste des pays disponibles
    print("Pays disponibles :", df['geo'].unique().tolist())
    print("Colonnes :", df.columns.tolist())