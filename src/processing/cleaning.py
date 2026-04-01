"""
cleaning.py
-----------
Cleaning and standardisation of raw asylum datasets.

Operations:
  - Duplicate removal
  - Column name standardisation
  - Missing value handling
  - Type conversion
  - Aggregates vs granular data flagging
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

EUROSTAT_NA_VALUES = [":", "n/a", "N/A", "-", ""]

KEY_COLS_APPLICATIONS = ["time", "geo", "citizen", "sex", "age", "applicant"]
KEY_COLS_DECISIONS    = ["time", "geo", "citizen", "sex", "age", "decision"]


# ------------------------------------------------------------------
# Applications
# ------------------------------------------------------------------

def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean asylum applications dataset (Eurostat migr_asyappctzm).

    Steps:
      1. Standardise column names
      2. Convert types
      3. Replace invalid values with NaN
      4. Remove duplicates
      5. Flag aggregates vs granular rows

    Parameters
    ----------
    df : Raw DataFrame from EurostatClient.get_asylum_applications()

    Returns
    -------
    Cleaned DataFrame
    """
    logger.info(f"Cleaning applications — {len(df):,} rows input")
    df = df.copy()

    # 1. Lowercase column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Numeric conversion
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 3. Replace invalid values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(EUROSTAT_NA_VALUES, np.nan)

    # 4. Remove duplicates
    n_before = len(df)
    key_cols = [c for c in KEY_COLS_APPLICATIONS if c in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(f"  Removed {n_dupes:,} duplicate rows.")

    # 5. Add datetime period columns
    df["period"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month

    # 6. Flag aggregate rows (sex=Total AND age=Total)
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
    Clean first instance decisions dataset (Eurostat migr_asydcfsta).

    Parameters
    ----------
    df : Raw DataFrame from EurostatClient.get_first_instance_decisions()

    Returns
    -------
    Cleaned DataFrame
    """
    logger.info(f"Cleaning decisions — {len(df):,} rows input")
    df = df.copy()

    # 1. Column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Numeric conversion
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 3. Invalid values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(EUROSTAT_NA_VALUES, np.nan)

    # 4. Duplicates
    n_before = len(df)
    key_cols = [c for c in KEY_COLS_DECISIONS if c in df.columns]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning(f"  Removed {n_dupes:,} duplicate rows.")

    # 5. Datetime period
    df["period"] = pd.to_datetime(df["time"], format="%Y-%m", errors="coerce")
    df["year"]   = df["period"].dt.year
    df["month"]  = df["period"].dt.month

    logger.info(f"  → {len(df):,} rows after cleaning")
    return df


def clean_unhcr_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean UNHCR global trends dataset.

    Parameters
    ----------
    df : DataFrame from UNHCRClient.get_global_trends()

    Returns
    -------
    Cleaned DataFrame with derived indicators
    """
    logger.info(f"Cleaning UNHCR trends — {len(df):,} rows input")
    df = df.copy()

    # Numeric conversion for all columns except year
    for col in df.columns:
        if col != "year":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived indicator: asylum seekers / refugees ratio
    df["asylum_to_refugee_ratio"] = (
        df["asylum_seekers"] / df["refugees"].replace(0, np.nan)
    ).round(4)

    # Year-over-year refugee change (%)
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