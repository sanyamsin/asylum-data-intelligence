"""
validation.py
-------------
Three-tier Quality Assurance framework for asylum datasets.

Tier 1 — Structural validation (schema, completeness, duplicates)
Tier 2 — Statistical coherence (logical consistency, outliers, variations)
Tier 3 — Timeliness monitoring (gaps, data freshness)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class QACheck:
    tier: int
    check_name: str
    status: str          # "PASS" | "WARN" | "FAIL"
    message: str
    affected_rows: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class QAReport:
    dataset_name: str
    run_timestamp: str
    total_rows: int
    checks: List[QACheck] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        statuses = [c.status for c in self.checks]
        return {
            "PASS": statuses.count("PASS"),
            "WARN": statuses.count("WARN"),
            "FAIL": statuses.count("FAIL"),
            "overall": (
                "FAIL" if "FAIL" in statuses
                else "WARN" if "WARN" in statuses
                else "PASS"
            ),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "dataset": self.dataset_name,
            "run_timestamp": self.run_timestamp,
            "total_rows": self.total_rows,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }
        output = json.dumps(data, indent=2, ensure_ascii=False, default=int)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
            logger.info(f"QA report saved to {path}")
        return output

    def print_summary(self):
        s = self.summary
        print(f"\n{'='*60}")
        print(f"QA REPORT — {self.dataset_name}")
        print(f"Run: {self.run_timestamp} | Rows: {self.total_rows:,}")
        print(f"{'='*60}")
        print(f"  PASS: {s['PASS']}  WARN: {s['WARN']}  FAIL: {s['FAIL']}")
        print(f"  Overall: {s['overall']}")
        print(f"{'-'*60}")
        for check in self.checks:
            icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(check.status, "?")
            print(f"  {icon} [Tier {check.tier}] {check.check_name}: {check.message}")
        print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------

class AsylumDataValidator:
    """
    Validate asylum datasets across three QA tiers.

    Parameters
    ----------
    df               : DataFrame to validate
    dataset_name     : Human-readable name for reporting
    required_columns : Expected column names
    time_column      : Name of the time/period column
    value_column     : Name of the numeric value column
    geo_column       : Name of the country/geography column
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        required_columns: List[str],
        time_column:  str = "time",
        value_column: str = "value",
        geo_column:   str = "geo",
    ):
        self.df               = df.copy()
        self.dataset_name     = dataset_name
        self.required_columns = required_columns
        self.time_column      = time_column
        self.value_column     = value_column
        self.geo_column       = geo_column
        self.report = QAReport(
            dataset_name=dataset_name,
            run_timestamp=datetime.now().isoformat(),
            total_rows=len(df),
        )

    # ------------------------------------------------------------------
    # TIER 1 — Structural Validation
    # ------------------------------------------------------------------

    def _check_schema(self):
        missing = [c for c in self.required_columns if c not in self.df.columns]
        if missing:
            self.report.checks.append(QACheck(
                tier=1, check_name="Schema conformity", status="FAIL",
                message=f"Missing columns: {missing}",
                details={"missing_columns": missing},
            ))
        else:
            self.report.checks.append(QACheck(
                tier=1, check_name="Schema conformity", status="PASS",
                message="All required columns present.",
            ))

    def _check_completeness(self, threshold: float = 0.95):
        if self.value_column not in self.df.columns:
            return
        total    = len(self.df)
        non_null = self.df[self.value_column].notna().sum()
        rate     = non_null / total if total > 0 else 0.0
        status   = "PASS" if rate >= threshold else ("WARN" if rate >= 0.80 else "FAIL")
        self.report.checks.append(QACheck(
            tier=1, check_name="Value completeness",
            status=status,
            message=f"Completeness: {rate:.1%} ({non_null:,}/{total:,} non-null values)",
            affected_rows=total - non_null,
            details={"completeness_rate": round(rate, 4), "threshold": threshold},
        ))

    def _check_duplicates(self):
        key_cols = [c for c in [self.time_column, self.geo_column] if c in self.df.columns]
        if not key_cols:
            return
        n_dupes = self.df.duplicated(subset=key_cols).sum()
        status  = "PASS" if n_dupes == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=1, check_name="Duplicate records",
            status=status,
            message=f"{n_dupes:,} duplicate (time x geo) combinations detected.",
            affected_rows=int(n_dupes),
        ))

    # ------------------------------------------------------------------
    # TIER 2 — Statistical Coherence
    # ------------------------------------------------------------------

    def _check_negative_values(self):
        if self.value_column not in self.df.columns:
            return
        neg    = (self.df[self.value_column] < 0).sum()
        status = "PASS" if neg == 0 else "FAIL"
        self.report.checks.append(QACheck(
            tier=2, check_name="Negative values",
            status=status,
            message=f"{neg:,} negative values detected in '{self.value_column}'.",
            affected_rows=int(neg),
        ))

    def _check_zscore_outliers(self, threshold: float = 3.5):
        if self.value_column not in self.df.columns:
            return
        vals     = self.df[self.value_column].dropna()
        if len(vals) < 10:
            return
        z_scores = np.abs((vals - vals.mean()) / vals.std())
        outliers = (z_scores > threshold).sum()
        status   = "PASS" if outliers == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=2, check_name="Statistical outliers (Z-score)",
            status=status,
            message=f"{outliers:,} values exceed Z-score threshold of {threshold}.",
            affected_rows=int(outliers),
            details={"z_threshold": threshold},
        ))

    def _check_mom_variation(self, max_ratio: float = 5.0):
        """Flag month-over-month variations exceeding max_ratio."""
        if self.time_column not in self.df.columns or self.value_column not in self.df.columns:
            return
        if self.geo_column not in self.df.columns:
            return
        alerts = 0
        for _, grp in self.df.groupby(self.geo_column):
            grp_sorted = grp.sort_values(self.time_column)
            vals       = grp_sorted[self.value_column].replace(0, np.nan)
            ratio      = vals / vals.shift(1)
            alerts    += int((ratio > max_ratio).sum())
        status = "PASS" if alerts == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=2, check_name="Month-over-month variation",
            status=status,
            message=f"{alerts:,} country-periods with >{max_ratio}x MoM change.",
            affected_rows=alerts,
            details={"max_ratio": max_ratio},
        ))

    # ------------------------------------------------------------------
    # TIER 3 — Timeliness
    # ------------------------------------------------------------------

    def _check_time_series_gaps(self):
        if self.time_column not in self.df.columns or self.geo_column not in self.df.columns:
            return
        gaps_total = 0
        for _, grp in self.df.groupby(self.geo_column):
            try:
                periods    = pd.PeriodIndex(grp[self.time_column].unique(), freq="M").sort_values()
                expected   = pd.period_range(start=periods.min(), end=periods.max(), freq="M")
                gaps_total += len(expected) - len(periods)
            except Exception:
                pass
        status = "PASS" if gaps_total == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=3, check_name="Time series continuity",
            status=status,
            message=f"{gaps_total:,} missing period(s) across country time series.",
            affected_rows=gaps_total,
        ))

    def _check_data_freshness(self, expected_lag_days: int = 60):
        if self.time_column not in self.df.columns:
            return
        try:
            latest_str  = self.df[self.time_column].max()
            latest_date = pd.Period(latest_str, freq="M").to_timestamp("M")
            lag_days    = (datetime.now() - latest_date).days
            status      = "PASS" if lag_days <= expected_lag_days else "WARN"
            self.report.checks.append(QACheck(
                tier=3, check_name="Data freshness",
                status=status,
                message=f"Latest period: {latest_str} ({lag_days} days lag). Threshold: {expected_lag_days} days.",
                details={"latest_period": str(latest_str), "lag_days": lag_days},
            ))
        except Exception as e:
            logger.warning(f"Could not check data freshness: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> QAReport:
        """Run all QA checks and return the populated QAReport."""
        logger.info(f"Running QA on '{self.dataset_name}' ({len(self.df):,} rows)...")

        # Tier 1
        self._check_schema()
        self._check_completeness()
        self._check_duplicates()

        # Tier 2
        self._check_negative_values()
        self._check_zscore_outliers()
        self._check_mom_variation()

        # Tier 3
        self._check_time_series_gaps()
        self._check_data_freshness()

        self.report.print_summary()
        return self.report


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    from pathlib import Path
    DATA_PROCESSED = Path("data/processed")

    df = pd.read_csv(DATA_PROCESSED / "applications_clean.csv")

    validator = AsylumDataValidator(
        df=df,
        dataset_name="Eurostat Applications (migr_asyappctzm)",
        required_columns=["time", "geo", "citizen", "sex", "age", "value"],
        time_column="time",
        value_column="value",
        geo_column="geo",
    )
    report = validator.run_all()
    report.to_json("data/processed/qa_report_applications.json")