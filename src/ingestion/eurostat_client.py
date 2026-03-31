"""
eurostat_client.py
------------------
Client pour récupérer les données d'asile depuis Eurostat
via l'API REST SDMX-JSON.
"""

import logging
import time
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

EUROSTAT_BASE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
)

DATASETS = {
    "applications": "migr_asyappctzm",
    "decisions":    "migr_asydcfsta",
    "dublin":       "migr_dubti",
    "minors":       "migr_asyunaa",
}


class EurostatClient:
    """
    Fetch asylum statistics from the Eurostat SDMX-JSON API.

    Usage:
        client = EurostatClient()
        df = client.get_asylum_applications(start_period="2020-01")
    """

    def __init__(self, retry_attempts: int = 3, retry_delay: float = 2.0):
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, dataset_code: str, params: dict) -> dict:
        """HTTP GET with retry logic."""
        url = f"{EUROSTAT_BASE_URL}/{dataset_code}"
        params["format"] = "JSON"
        params["lang"]   = "EN"

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)

        raise ConnectionError(
            f"Failed to fetch {dataset_code} after {self.retry_attempts} attempts."
        )

    def _parse(self, raw: dict) -> pd.DataFrame:
        """Convert SDMX-JSON response to a tidy DataFrame."""
        dataset   = raw.get("dataset", raw)
        dimension = dataset["dimension"]
        value     = dataset["value"]

        dims      = {}
        dim_sizes = []

        for dim_name, dim_data in dimension.items():
            cat = dim_data["category"]
            # index: {label_code: position}  →  we invert to {position: label_code}
            index_map = {int(v): k for k, v in cat["index"].items()}
            # label: {label_code: display_name}
            label_map = cat.get("label", {})
            # final map: position → display_name (or code if no label)
            dims[dim_name] = {
                pos: label_map.get(code, code)
                for pos, code in index_map.items()
            }
            dim_sizes.append(len(index_map))

        records = []
        for flat_str, obs_value in value.items():
            flat_idx  = int(flat_str)
            coords    = {}
            remainder = flat_idx
            for i, (dim_name, dim_map) in enumerate(reversed(list(dims.items()))):
                size      = dim_sizes[-(i + 1)]
                coord_idx = remainder % size
                remainder //= size
                coords[dim_name] = dim_map.get(coord_idx, coord_idx)
            coords["value"] = obs_value
            records.append(coords)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_asylum_applications(
        self,
        start_period: Optional[str] = None,
        end_period:   Optional[str] = None,
        geo:          Optional[list] = None,
        citizen:      Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Asylum applications by citizenship (migr_asyappctzm).
        Fetche pays par pays pour éviter les erreurs 413.
        """
        geo_list     = geo     or ["DE"]
        citizen_list = citizen or ["SY", "AF", "IQ", "PK", "NG"]

        all_dfs = []
        for country in geo_list:
            logger.info(f"  Fetching {country}...")
            try:
                url = f"{EUROSTAT_BASE_URL}/{DATASETS['applications']}"
                multi_params = [
                    ("startPeriod", start_period or "2020-01"),
                    ("endPeriod",   end_period   or "2026-03"),
                    ("format",      "JSON"),
                    ("lang",        "EN"),
                    ("geo",         country),
                ]
                for c in citizen_list:
                    multi_params.append(("citizen", c))

                response = requests.get(url, params=multi_params, timeout=30)
                response.raise_for_status()
                raw = response.json()
                df  = self._parse(raw)
                df["dataset"] = "applications"
                all_dfs.append(df)
                logger.info(f"    → {len(df):,} rows")
            except Exception as e:
                logger.warning(f"    ⚠️ {country} failed: {e}")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total applications fetched: {len(result):,} rows.")
        return result

    def get_first_instance_decisions(
        self,
        start_period: Optional[str] = None,
        end_period:   Optional[str] = None,
        geo:          Optional[list] = None,
    ) -> pd.DataFrame:
        """
        First instance asylum decisions (migr_asydcfsta).
        Fetche pays par pays pour éviter les erreurs 413.
        """
        geo_list = geo or ["DE", "FR", "IT", "ES", "AT",
                           "BE", "NL", "SE", "PL"]

        all_dfs = []
        for country in geo_list:
            logger.info(f"  Fetching decisions {country}...")
            try:
                multi_params = [
                    ("startPeriod", start_period or "2020-01"),
                    ("endPeriod",   end_period   or "2026-03"),
                    ("format",      "JSON"),
                    ("lang",        "EN"),
                    ("geo",         country),
                ]
                response = requests.get(
                    f"{EUROSTAT_BASE_URL}/{DATASETS['decisions']}",
                    params=multi_params,
                    timeout=30,
                )
                response.raise_for_status()
                raw = response.json()
                df  = self._parse(raw)
                df["dataset"] = "decisions"
                all_dfs.append(df)
                logger.info(f"    → {len(df):,} rows")
            except Exception as e:
                logger.warning(f"    ⚠️ {country} failed: {e}")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total decisions fetched: {len(result):,} rows.")
        return result