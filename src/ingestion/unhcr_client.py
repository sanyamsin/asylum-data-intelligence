"""
unhcr_client.py
---------------
Client pour récupérer les tendances mondiales de populations
réfugiées depuis l'API UNHCR Refugee Data Finder.

Note : L'API UNHCR fournit des agrégats globaux (séries temporelles
mondiales). Pour les données par pays, Eurostat est la source
principale (voir eurostat_client.py).

Documentation : https://api.unhcr.org/population/v1/
"""

import io
import logging
import zipfile
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

UNHCR_BASE_URL = "https://api.unhcr.org/population/v1"

COLUMN_NAMES = [
    "year", "coo_name", "coo_iso", "coa_name", "coa_iso",
    "refugees", "asylum_seekers", "returned_refugees",
    "idps", "returned_idps", "stateless", "ooc", "oip", "hst",
]


class UNHCRClient:
    """
    Fetch global refugee and asylum seeker trends from UNHCR.

    Usage:
        client = UNHCRClient()
        df = client.get_global_trends(year_from=2000)
    """

    def __init__(self, retry_attempts: int = 3, retry_delay: float = 2.0):
        self.retry_attempts = retry_attempts
        self.retry_delay    = retry_delay
        self.session        = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_csv(self, endpoint: str) -> pd.DataFrame:
        """Télécharge le CSV zippé et retourne un DataFrame propre."""
        import time
        url = f"{UNHCR_BASE_URL}/{endpoint}/?download=true"

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                z  = zipfile.ZipFile(io.BytesIO(response.content))
                df = pd.read_csv(
                    z.open(f"{endpoint}.csv"),
                    na_values=["-"],
                )
                df.columns = COLUMN_NAMES
                return df
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Failed to download {endpoint} after {self.retry_attempts} attempts.")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_global_trends(
        self,
        year_from: int = 2000,
        year_to:   Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Tendances mondiales : réfugiés et demandeurs d'asile par année.
        Source : agrégats globaux UNHCR (1951 → présent).

        Parameters
        ----------
        year_from : Année de début (défaut 2000)
        year_to   : Année de fin (défaut : dernière disponible)

        Returns
        -------
        DataFrame avec colonnes :
            year, refugees, asylum_seekers, idps, stateless
        """
        logger.info(f"Fetching UNHCR global trends [{year_from} → {year_to or 'latest'}]")
        df = self._download_csv("population")

        # Garder uniquement les agrégats globaux (pas de pays spécifié)
        df_global = df[df["coo_iso"].isna()].copy()

        # Filtrer par année
        df_global = df_global[df_global["year"] >= year_from]
        if year_to:
            df_global = df_global[df_global["year"] <= year_to]

        # Colonnes pertinentes
        cols = ["year", "refugees", "asylum_seekers", "returned_refugees",
                "idps", "stateless"]
        df_global = df_global[cols].reset_index(drop=True)

        # Convertir en numérique
        for col in cols[1:]:
            df_global[col] = pd.to_numeric(df_global[col], errors="coerce")

        logger.info(f"  → {len(df_global):,} years of global data fetched.")
        return df_global