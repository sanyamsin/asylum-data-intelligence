"""
validation.py
-------------
Framework QA à trois niveaux pour les datasets d'asile.

Tier 1 — Validation structurelle (schéma, complétude, doublons)
Tier 2 — Cohérence statistique (logique, outliers, variations)
Tier 3 — Surveillance timeliness (gaps, fraîcheur des données)
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
        print(f"  ✅ PASS: {s['PASS']}  ⚠️  WARN: {s['WARN']}  ❌ FAIL: {s['FAIL']}")
        print(f"  Overall: {s['overall']}")
        print(f"{'-'*60}")
        for check in self.checks:
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(check.status, "?")
            print(f"  {icon} [Tier {check.tier}] {check.check_name}: {check.message}")
        print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------

class AsylumDataValidator:
    """
    Valide les datasets d'asile sur trois niveaux QA.

    Parameters
    ----------
    df               : DataFrame à valider
    dataset_name     : Nom pour le rapport
    required_columns : Colonnes attendues
    time_column      : Nom de la colonne temporelle
    value_column     : Nom de la colonne numérique
    geo_column       : Nom de la colonne pays
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
    # TIER 1 — Validation structurelle
    # ------------------------------------------------------------------

    def _check_schema(self):
        missing = [c for c in self.required_columns if c not in self.df.columns]
        if missing:
            self.report.checks.append(QACheck(
                tier=1, check_name="Conformité schéma", status="FAIL",
                message=f"Colonnes manquantes : {missing}",
                details={"missing_columns": missing},
            ))
        else:
            self.report.checks.append(QACheck(
                tier=1, check_name="Conformité schéma", status="PASS",
                message="Toutes les colonnes requises sont présentes.",
            ))

    def _check_completeness(self, threshold: float = 0.95):
        if self.value_column not in self.df.columns:
            return
        total    = len(self.df)
        non_null = self.df[self.value_column].notna().sum()
        rate     = non_null / total if total > 0 else 0.0
        status   = "PASS" if rate >= threshold else ("WARN" if rate >= 0.80 else "FAIL")
        self.report.checks.append(QACheck(
            tier=1, check_name="Complétude des valeurs",
            status=status,
            message=f"Complétude : {rate:.1%} ({non_null:,}/{total:,} valeurs non-nulles)",
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
            tier=1, check_name="Enregistrements dupliqués",
            status=status,
            message=f"{n_dupes:,} combinaisons (time × geo) dupliquées détectées.",
            affected_rows=int(n_dupes),
        ))

    # ------------------------------------------------------------------
    # TIER 2 — Cohérence statistique
    # ------------------------------------------------------------------

    def _check_negative_values(self):
        if self.value_column not in self.df.columns:
            return
        neg = (self.df[self.value_column] < 0).sum()
        status = "PASS" if neg == 0 else "FAIL"
        self.report.checks.append(QACheck(
            tier=2, check_name="Valeurs négatives",
            status=status,
            message=f"{neg:,} valeurs négatives détectées dans '{self.value_column}'.",
            affected_rows=int(neg),
        ))

    def _check_zscore_outliers(self, threshold: float = 3.5):
        if self.value_column not in self.df.columns:
            return
        vals = self.df[self.value_column].dropna()
        if len(vals) < 10:
            return
        z_scores = np.abs((vals - vals.mean()) / vals.std())
        outliers = (z_scores > threshold).sum()
        status   = "PASS" if outliers == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=2, check_name="Outliers statistiques (Z-score)",
            status=status,
            message=f"{outliers:,} valeurs dépassent le seuil Z={threshold}.",
            affected_rows=int(outliers),
            details={"z_threshold": threshold},
        ))

    def _check_mom_variation(self, max_ratio: float = 5.0):
        """Détecte les variations mois-sur-mois anormales."""
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
            tier=2, check_name="Variation mois-sur-mois",
            status=status,
            message=f"{alerts:,} pays-périodes avec variation >×{max_ratio} (erreur ou surge réel).",
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
        for geo, grp in self.df.groupby(self.geo_column):
            try:
                periods  = pd.PeriodIndex(grp[self.time_column].unique(), freq="M").sort_values()
                expected = pd.period_range(start=periods.min(), end=periods.max(), freq="M")
                gaps_total += len(expected) - len(periods)
            except Exception:
                pass
        status = "PASS" if gaps_total == 0 else "WARN"
        self.report.checks.append(QACheck(
            tier=3, check_name="Continuité des séries temporelles",
            status=status,
            message=f"{gaps_total:,} période(s) manquante(s) dans les séries pays.",
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
                tier=3, check_name="Fraîcheur des données",
                status=status,
                message=f"Dernière période : {latest_str} ({lag_days} jours de lag). Seuil : {expected_lag_days}j.",
                details={"latest_period": str(latest_str), "lag_days": lag_days},
            ))
        except Exception as e:
            logger.warning(f"Impossible de vérifier la fraîcheur : {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> QAReport:
        """Lance tous les checks QA et retourne le rapport."""
        logger.info(f"Lancement QA sur '{self.dataset_name}' ({len(self.df):,} lignes)...")

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