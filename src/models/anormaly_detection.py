"""
anomaly_detection.py
--------------------
Anomaly detection for asylum datasets.

Methods:
  - Z-score monitoring   : univariate, per country series
  - Isolation Forest     : multivariate, cross-feature anomalies
  - Classification       : data error vs. genuine operational event
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------

@dataclass
class AnomalyReport:
    dataset_name:   str
    method:         str
    total_rows:     int
    anomaly_count:  int
    anomaly_rate:   float
    anomalies:      pd.DataFrame = field(default_factory=pd.DataFrame)
    notes:          str = ""

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"ANOMALY REPORT — {self.dataset_name}")
        print(f"Method: {self.method}")
        print(f"{'='*60}")
        print(f"  Total rows     : {self.total_rows:,}")
        print(f"  Anomalies      : {self.anomaly_count:,} ({self.anomaly_rate:.1%})")
        if not self.anomalies.empty:
            print(f"\n  Top anomalies:")
            print(self.anomalies.head(10).to_string(index=False))
        print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Z-score Detector
# ------------------------------------------------------------------

class ZScoreDetector:
    """
    Univariate anomaly detection per country time series.
    Flags values where |Z-score| exceeds threshold.

    Usage:
        detector = ZScoreDetector(threshold=3.5)
        report = detector.detect(df, country_col='geo', value_col='value')
    """

    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold

    def detect(
        self,
        df: pd.DataFrame,
        country_col: str = "geo",
        value_col:   str = "value",
        time_col:    str = "time",
    ) -> AnomalyReport:
        """
        Detect outliers per country using Z-score.

        Parameters
        ----------
        df          : Cleaned DataFrame
        country_col : Column name for country grouping
        value_col   : Column name for numeric values
        time_col    : Column name for time period
        """
        logger.info(f"Running Z-score detection (threshold={self.threshold})...")

        anomaly_rows = []

        for country, grp in df.groupby(country_col):
            vals   = grp[value_col].dropna()
            if len(vals) < 5:
                continue
            mean   = vals.mean()
            std    = vals.std()
            if std == 0:
                continue
            z_scores = np.abs((grp[value_col] - mean) / std)
            mask     = z_scores > self.threshold
            flagged  = grp[mask].copy()
            flagged["z_score"]  = z_scores[mask].round(2)
            flagged["mean_ref"] = round(mean, 1)
            flagged["std_ref"]  = round(std, 1)
            anomaly_rows.append(flagged)

        if anomaly_rows:
            anomalies_df = pd.concat(anomaly_rows, ignore_index=True)
            # Sort by z_score descending
            anomalies_df = anomalies_df.sort_values("z_score", ascending=False)
            cols = [c for c in [time_col, country_col, value_col, "z_score",
                                "mean_ref", "std_ref"] if c in anomalies_df.columns]
            anomalies_df = anomalies_df[cols]
        else:
            anomalies_df = pd.DataFrame()

        total      = len(df)
        n_anomalies = len(anomalies_df)

        report = AnomalyReport(
            dataset_name=f"Z-score | {country_col} series",
            method=f"Z-score (threshold={self.threshold})",
            total_rows=total,
            anomaly_count=n_anomalies,
            anomaly_rate=n_anomalies / total if total > 0 else 0.0,
            anomalies=anomalies_df,
        )

        report.print_summary()
        return report


# ------------------------------------------------------------------
# Isolation Forest Detector
# ------------------------------------------------------------------

class IsolationForestDetector:
    """
    Multivariate anomaly detection using Isolation Forest.
    Detects unusual combinations across multiple features.

    Usage:
        detector = IsolationForestDetector(contamination=0.05)
        report = detector.detect(df)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state:  int   = 42,
    ):
        self.contamination = contamination
        self.random_state  = random_state

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        id_cols:      Optional[List[str]] = None,
    ) -> AnomalyReport:
        """
        Detect multivariate anomalies using Isolation Forest.

        Parameters
        ----------
        df           : DataFrame with numeric features
        feature_cols : Columns to use as features (default: all numeric)
        id_cols      : Identifier columns to keep in output
        """
        logger.info(f"Running Isolation Forest (contamination={self.contamination})...")

        # Select features
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        features = df[feature_cols].copy()
        features = features.fillna(features.median())

        # Scale features
        scaler          = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Fit Isolation Forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        preds  = model.fit_predict(features_scaled)
        scores = model.score_samples(features_scaled)

        # Flag anomalies (-1 = anomaly)
        df_out                   = df.copy()
        df_out["anomaly_flag"]   = (preds == -1).astype(int)
        df_out["anomaly_score"]  = scores.round(4)

        anomalies_df = (
            df_out[df_out["anomaly_flag"] == 1]
            .sort_values("anomaly_score")
        )

        # Keep relevant columns
        keep_cols = (id_cols or []) + feature_cols + ["anomaly_flag", "anomaly_score"]
        keep_cols = [c for c in keep_cols if c in anomalies_df.columns]
        anomalies_df = anomalies_df[keep_cols]

        total       = len(df)
        n_anomalies = len(anomalies_df)

        report = AnomalyReport(
            dataset_name="Isolation Forest | multivariate",
            method=f"Isolation Forest (contamination={self.contamination})",
            total_rows=total,
            anomaly_count=n_anomalies,
            anomaly_rate=n_anomalies / total if total > 0 else 0.0,
            anomalies=anomalies_df,
            notes=(
                "Anomaly score: lower = more anomalous. "
                "Cross-reference with Z-score results to classify "
                "data errors vs. genuine operational events."
            ),
        )

        report.print_summary()
        return report


# ------------------------------------------------------------------
# Anomaly Classifier
# ------------------------------------------------------------------

def classify_anomalies(
    zscore_report: AnomalyReport,
    iforest_report: AnomalyReport,
    time_col:  str = "time",
    geo_col:   str = "geo",
) -> pd.DataFrame:
    """
    Classify anomalies as data errors vs. genuine operational events
    by cross-referencing Z-score and Isolation Forest results.

    Logic:
      - Anomaly in BOTH methods     → likely genuine operational event
      - Anomaly in ONE method only  → possible data error, flag for review
    """
    if zscore_report.anomalies.empty or iforest_report.anomalies.empty:
        logger.warning("One or both anomaly reports are empty — skipping classification.")
        return pd.DataFrame()

    # Get keys from each method
    z_keys = set(
        zip(zscore_report.anomalies.get(time_col, []),
            zscore_report.anomalies.get(geo_col, []))
    )
    i_keys = set(
        zip(iforest_report.anomalies.get(time_col, []),
            iforest_report.anomalies.get(geo_col, []))
    )

    corroborated  = z_keys & i_keys    # Both methods agree
    z_only        = z_keys - i_keys    # Z-score only
    i_only        = i_keys - z_keys    # Isolation Forest only

    rows = []
    for key in corroborated:
        rows.append({time_col: key[0], geo_col: key[1],
                     "classification": "GENUINE_EVENT",
                     "confidence": "HIGH"})
    for key in z_only:
        rows.append({time_col: key[0], geo_col: key[1],
                     "classification": "POSSIBLE_DATA_ERROR",
                     "confidence": "MEDIUM"})
    for key in i_only:
        rows.append({time_col: key[0], geo_col: key[1],
                     "classification": "POSSIBLE_DATA_ERROR",
                     "confidence": "LOW"})

    result = pd.DataFrame(rows).sort_values(["classification", time_col])
    logger.info(f"Classification complete: "
                f"{len(corroborated)} genuine events, "
                f"{len(z_only) + len(i_only)} possible errors.")
    return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    import pandas as pd
    from pathlib import Path

    DATA_PROCESSED = Path("data/processed")
    df = pd.read_csv(DATA_PROCESSED / "applications_clean.csv")

    # Filter: totals only
    df_total = df[df["is_total"] == True].copy()

    # 1. Z-score detection
    z_detector = ZScoreDetector(threshold=3.5)
    z_report   = z_detector.detect(
        df_total,
        country_col="geo",
        value_col="value",
        time_col="time",
    )

    # 2. Isolation Forest
    # Aggregate by country-month for multivariate features
    df_agg = (
        df_total
        .groupby(["time", "geo"])
        .agg(
            total_applications=("value", "sum"),
            n_nationalities=("citizen", "nunique"),
        )
        .reset_index()
    )
    # Cap extreme MoM variations to avoid infinity values
    df_agg["mom_change"] = (
        df_agg.groupby("geo")["total_applications"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-5, 5)
    )

    if_detector = IsolationForestDetector(contamination=0.05)
    if_report   = if_detector.detect(
        df_agg,
        feature_cols=["total_applications", "n_nationalities", "mom_change"],
        id_cols=["time", "geo"],
    )

    # 3. Classification
    classification = classify_anomalies(z_report, if_report)
    if not classification.empty:
        print("\nANOMALY CLASSIFICATION:")
        print(classification.to_string(index=False))

        # Save
        classification.to_csv(
            DATA_PROCESSED / "anomaly_classification.csv",
            index=False
        )
        print("\nSaved -> data/processed/anomaly_classification.csv")