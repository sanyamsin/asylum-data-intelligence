"""
cleaning.py
-----------
Nettoyage et standardisation des données d'asile brutes.

Opérations :
  - Suppression des doublons
  - Standardisation des noms de colonnes
  - Gestion des valeurs manquantes
  - Conversion des types
  - Filtrage des agrégats (TOTAL) vs données granulaires
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------

# Valeurs à traiter comme NaN dans Eurostat
EUROSTAT_NA_VALUES = [":", "n/a", "N/A", "-", ""]

# Colonnes de clé pour déduplication applications
KEY_COLS_APPLICATIONS = ["time", "geo", "citizen", "sex", "age", "applicant"]

# Colonnes de clé pour déduplication décisions
KEY_COLS_DECISIONS = ["time", "geo", "citizen", "sex", "age", "decision"]


# ------------------------------------------------------------------
# Applications
# ------------------------------------------------------------------

def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset des demandes d'asile (Eurostat migr_asyappctzm).

    Étapes :
      1. Standardise les noms de colonnes
      2. Convertit les types
      3. Remplace les valeurs invalides par NaN
      4. Supprime les doublons
      5. Sépare les agrégats des données granulaires

    Parameters
    ----------
    df : DataFrame brut issu de EurostatClient.get_asylum_applications()

    Returns
    -------
    DataFrame nettoyé
    """
    logger.info(f"Cleaning applications — {len(df):,} rows input")
    df = df.copy()

    # 1. Noms de colonnes en minuscules
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Conversion numérique de value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 3. Remplacer les valeurs invalides
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(EUROSTAT_NA_VALUES, np.nan)

    # 4. Supprimer les doublons
    n_before = len(df)
    key_cols = [c for c in KEY_COLS_APPLICATIONS if c in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(f"  Removed {n_dupes:,} duplicate rows.")

    # 5. Ajouter colonne période datetime
    df["period"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month

    # 6. Flag : agrégat total vs données granulaires
    df["is_total"] = (
        df.get("sex", pd.Series(["Total"] * len(df)))
          .str.lower()
          .str.contains("total", na=False)
        &
        df.get("age", pd.Series(["Total"] * len(df)))
          .str.lower()
          .str.contains("total", na=False)
    )

    logger.info(f"  → {len(df):,} rows after cleaning")
    logger.info(f"  → {df['value'].isna().sum():,} missing values")
    logger.info(f"  → {df['is_total'].sum():,} aggregate rows")

    return df


def clean_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset des décisions première instance (migr_asydcfsta).

    Parameters
    ----------
    df : DataFrame brut issu de EurostatClient.get_first_instance_decisions()

    Returns
    -------
    DataFrame nettoyé avec taux de reconnaissance calculé
    """
    logger.info(f"Cleaning decisions — {len(df):,} rows input")
    df = df.copy()

    # 1. Noms de colonnes
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Conversion numérique
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 3. Valeurs invalides
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(EUROSTAT_NA_VALUES, np.nan)

    # 4. Doublons
    n_before = len(df)
    key_cols = [c for c in KEY_COLS_DECISIONS if c in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(f"  Removed {n_dupes:,} duplicate rows.")

    # 5. Période datetime
    df["period"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month

    logger.info(f"  → {len(df):,} rows after cleaning")
    return df


def clean_unhcr_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les tendances mondiales UNHCR.

    Parameters
    ----------
    df : DataFrame issu de UNHCRClient.get_global_trends()

    Returns
    -------
    DataFrame nettoyé avec indicateurs dérivés
    """
    logger.info(f"Cleaning UNHCR trends — {len(df):,} rows input")
    df = df.copy()

    # Conversion numérique de toutes les colonnes sauf year
    for col in df.columns:
        if col != "year":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Indicateur dérivé : ratio demandeurs d'asile / réfugiés
    df["asylum_to_refugee_ratio"] = (
        df["asylum_seekers"] / df["refugees"].replace(0, np.nan)
    ).round(4)

    # Variation annuelle des réfugiés (%)
    df["refugees_yoy_pct"] = df["refugees"].pct_change().round(4)

    logger.info(f"  → {len(df):,} rows after cleaning")
    return df


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    DATA_RAW       = Path("data/raw")
    DATA_PROCESSED = Path("data/processed")
    DATA_PROCESSED.mkdir(exist_ok=True)

    # Applications
    df_app = pd.read_csv(DATA_RAW / "eurostat_applications.csv")
    df_app_clean = clean_applications(df_app)
    df_app_clean.to_csv(DATA_PROCESSED / "applications_clean.csv", index=False)
    logger.info("Saved: applications_clean.csv")

    # Decisions
    df_dec = pd.read_csv(DATA_RAW / "eurostat_decisions.csv")
    df_dec_clean = clean_decisions(df_dec)
    df_dec_clean.to_csv(DATA_PROCESSED / "decisions_clean.csv", index=False)
    logger.info("Saved: decisions_clean.csv")

    # UNHCR
    df_unhcr = pd.read_csv(DATA_RAW / "unhcr_global_trends.csv")
    df_unhcr_clean = clean_unhcr_trends(df_unhcr)
    df_unhcr_clean.to_csv(DATA_PROCESSED / "unhcr_trends_clean.csv", index=False)
    logger.info("Saved: unhcr_trends_clean.csv")