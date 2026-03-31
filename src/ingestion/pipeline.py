"""
pipeline.py
-----------
Orchestration du pipeline d'ingestion de données.
Coordonne les appels Eurostat et UNHCR, sauvegarde
les données brutes et produit un rapport de collecte.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingestion.eurostat_client import EurostatClient
from src.ingestion.unhcr_client import UNHCRClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Pays EU principaux pour les données d'asile
DEFAULT_GEO = ["DE", "FR", "IT", "ES", "GR", "AT", "BE", "NL", "SE", "PL"]

# Nationalités les plus représentées
DEFAULT_CITIZENS = ["SY", "AF", "IQ", "PK", "NG", "VE", "CO", "TN", "MA", "TR"]

# Répertoires de sortie
DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def run_pipeline(
    start_period: str = "2020-01",
    end_period:   str = None,
    geo:          list = None,
    citizens:     list = None,
) -> dict:
    """
    Lance le pipeline complet d'ingestion de données.

    Parameters
    ----------
    start_period : Début de la période (format YYYY-MM)
    end_period   : Fin de la période (défaut : mois actuel)
    geo          : Liste de pays ISO2 (défaut : 10 principaux pays EU)
    citizens     : Liste de nationalités ISO2

    Returns
    -------
    dict : Rapport de collecte (sources, lignes, statut, timestamp)
    """
    start_time = datetime.utcnow()
    geo        = geo      or DEFAULT_GEO
    citizens   = citizens or DEFAULT_CITIZENS

    if end_period is None:
        end_period = datetime.utcnow().strftime("%Y-%m")

    logger.info("=" * 60)
    logger.info("ASYLUM DATA INTELLIGENCE — PIPELINE START")
    logger.info(f"Period  : {start_period} → {end_period}")
    logger.info(f"Countries: {geo}")
    logger.info("=" * 60)

    # Créer les répertoires si nécessaire
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline_run": start_time.isoformat(),
        "period": {"start": start_period, "end": end_period},
        "sources": {},
    }

    # ------------------------------------------------------------------
    # SOURCE 1 — Eurostat : demandes d'asile
    # ------------------------------------------------------------------
    try:
        logger.info("\n[1/3] Fetching Eurostat asylum applications...")
        eurostat = EurostatClient()
        df_applications = eurostat.get_asylum_applications(
            start_period=start_period,
            end_period=end_period,
            geo=geo,
            citizen=citizens,
        )
        _save(df_applications, DATA_RAW / "eurostat_applications.csv")
        report["sources"]["eurostat_applications"] = {
            "status": "OK",
            "rows": len(df_applications),
            "columns": list(df_applications.columns),
        }
        logger.info(f"  ✅ {len(df_applications):,} rows saved.")

    except Exception as e:
        logger.error(f"  ❌ Eurostat applications failed: {e}")
        report["sources"]["eurostat_applications"] = {"status": "FAILED", "error": str(e)}

    # ------------------------------------------------------------------
    # SOURCE 2 — Eurostat : décisions première instance
    # ------------------------------------------------------------------
    try:
        logger.info("\n[2/3] Fetching Eurostat first instance decisions...")
        df_decisions = eurostat.get_first_instance_decisions(
            start_period=start_period,
            end_period=end_period,
        )
        _save(df_decisions, DATA_RAW / "eurostat_decisions.csv")
        report["sources"]["eurostat_decisions"] = {
            "status": "OK",
            "rows": len(df_decisions),
            "columns": list(df_decisions.columns),
        }
        logger.info(f"  ✅ {len(df_decisions):,} rows saved.")

    except Exception as e:
        logger.error(f"  ❌ Eurostat decisions failed: {e}")
        report["sources"]["eurostat_decisions"] = {"status": "FAILED", "error": str(e)}

    # ------------------------------------------------------------------
    # SOURCE 3 — UNHCR : tendances mondiales
    # ------------------------------------------------------------------
    try:
        logger.info("\n[3/3] Fetching UNHCR global trends...")
        unhcr = UNHCRClient()
        df_unhcr = unhcr.get_global_trends(year_from=2000)
        _save(df_unhcr, DATA_RAW / "unhcr_global_trends.csv")
        report["sources"]["unhcr_global_trends"] = {
            "status": "OK",
            "rows": len(df_unhcr),
            "columns": list(df_unhcr.columns),
        }
        logger.info(f"  ✅ {len(df_unhcr):,} rows saved.")

    except Exception as e:
        logger.error(f"  ❌ UNHCR failed: {e}")
        report["sources"]["unhcr_global_trends"] = {"status": "FAILED", "error": str(e)}

    # ------------------------------------------------------------------
    # Rapport final
    # ------------------------------------------------------------------
    duration = (datetime.utcnow() - start_time).seconds
    report["duration_seconds"] = duration
    report["overall_status"]   = (
        "OK" if all(s.get("status") == "OK"
                    for s in report["sources"].values())
        else "PARTIAL"
    )

    report_path = DATA_RAW / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("\n" + "=" * 60)
    logger.info(f"PIPELINE COMPLETE — {report['overall_status']} ({duration}s)")
    logger.info("=" * 60)

    return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save(df: pd.DataFrame, path: Path):
    """Sauvegarde un DataFrame en CSV."""
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  Saved → {path} ({path.stat().st_size / 1024:.1f} KB)")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    report = run_pipeline(start_period="2020-01")
    print("\nRapport final :")
    print(json.dumps(report, indent=2))