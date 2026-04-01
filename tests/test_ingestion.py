"""
test_ingestion.py
-----------------
Tests unitaires pour les clients Eurostat et UNHCR.
Utilise des mocks pour ne pas faire d'appels réseau réels.
"""

import sys
sys.path.insert(0, '.')

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.eurostat_client import EurostatClient
from src.ingestion.unhcr_client import UNHCRClient


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def eurostat_client():
    return EurostatClient()


@pytest.fixture
def unhcr_client():
    return UNHCRClient()


@pytest.fixture
def mock_eurostat_response():
    """Simule une réponse SDMX-JSON minimale d'Eurostat."""
    return {
        "id": ["freq", "geo", "time"],
        "size": [1, 1, 2],
        "dimension": {
            "freq": {
                "category": {
                    "index": {"M": 0},
                    "label": {"M": "Monthly"},
                }
            },
            "geo": {
                "category": {
                    "index": {"DE": 0},
                    "label": {"DE": "Germany"},
                }
            },
            "time": {
                "category": {
                    "index": {"2023-01": 0, "2023-02": 1},
                    "label": {"2023-01": "2023-01", "2023-02": "2023-02"},
                }
            },
        },
        "value": {"0": 1200, "1": 1350},
    }


# ------------------------------------------------------------------
# Tests EurostatClient
# ------------------------------------------------------------------

class TestEurostatClient:

    def test_client_initializes(self, eurostat_client):
        """Le client s'initialise avec les bons paramètres."""
        assert eurostat_client.retry_attempts == 3
        assert eurostat_client.retry_delay == 2.0

    def test_parse_returns_dataframe(self, eurostat_client, mock_eurostat_response):
        """_parse() convertit correctement le JSON SDMX en DataFrame."""
        df = eurostat_client._parse(mock_eurostat_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "value" in df.columns

    def test_parse_correct_values(self, eurostat_client, mock_eurostat_response):
        """_parse() extrait les bonnes valeurs."""
        df = eurostat_client._parse(mock_eurostat_response)
        values = sorted(df["value"].tolist())
        assert values == [1200, 1350]

    def test_parse_correct_columns(self, eurostat_client, mock_eurostat_response):
        """_parse() crée les bonnes colonnes."""
        df = eurostat_client._parse(mock_eurostat_response)
        assert "freq" in df.columns
        assert "geo" in df.columns
        assert "time" in df.columns

    @patch("src.ingestion.eurostat_client.requests.get")
    def test_fetch_raises_on_failure(self, mock_get, eurostat_client):
        """_fetch() lève une ConnectionError après épuisement des retry."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")
        with pytest.raises(ConnectionError):
            eurostat_client._fetch("migr_asyappctzm", {})

    @patch("src.ingestion.eurostat_client.requests.get")
    def test_get_applications_adds_dataset_column(
        self, mock_get, eurostat_client, mock_eurostat_response
    ):
        """get_asylum_applications() ajoute la colonne dataset='applications'."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_eurostat_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = eurostat_client.get_asylum_applications(
            start_period="2023-01",
            end_period="2023-02",
            geo=["DE"],
            citizen=["SY"],
        )
        assert "dataset" in df.columns
        assert (df["dataset"] == "applications").all()


# ------------------------------------------------------------------
# Tests UNHCRClient
# ------------------------------------------------------------------

class TestUNHCRClient:

    def test_client_initializes(self, unhcr_client):
        """Le client s'initialise correctement."""
        assert unhcr_client.retry_attempts == 3

    @patch("src.ingestion.unhcr_client.requests.Session.get")
    def test_get_global_trends_returns_dataframe(self, mock_get, unhcr_client):
        """get_global_trends() retourne un DataFrame avec les bonnes colonnes."""
        import io, zipfile

        # Créer un faux CSV zippé
        csv_content = (
            "Year,Country of origin,Country of origin (ISO),"
            "Country of asylum,Country of asylum (ISO),"
            "Refugees under UNHCR's mandate,Asylum-seekers,"
            "Returned refugees,IDPs of concern to UNHCR,"
            "Returned IDPss,Stateless persons,"
            "Others of concern,"
            "Other people in need of international protection,"
            "Host Community\n"
            "2020,-,-,-,-,20000000,4000000,300000,40000000,200000,3000000,1000000,-,0\n"
            "2021,-,-,-,-,21000000,4500000,350000,42000000,220000,3200000,1100000,-,0\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("population.csv", csv_content)
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.content = buf.read()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = unhcr_client.get_global_trends(year_from=2020)

        assert isinstance(df, pd.DataFrame)
        assert "refugees" in df.columns
        assert "asylum_seekers" in df.columns
        assert len(df) == 2

    @patch("src.ingestion.unhcr_client.requests.Session.get")
    def test_get_global_trends_filters_by_year(self, mock_get, unhcr_client):
        """get_global_trends() filtre correctement par year_from."""
        import io, zipfile

        csv_content = (
            "Year,Country of origin,Country of origin (ISO),"
            "Country of asylum,Country of asylum (ISO),"
            "Refugees under UNHCR's mandate,Asylum-seekers,"
            "Returned refugees,IDPs of concern to UNHCR,"
            "Returned IDPss,Stateless persons,"
            "Others of concern,"
            "Other people in need of international protection,"
            "Host Community\n"
            "2018,-,-,-,-,18000000,3000000,200000,35000000,150000,2500000,900000,-,0\n"
            "2019,-,-,-,-,19000000,3500000,250000,38000000,180000,2800000,950000,-,0\n"
            "2020,-,-,-,-,20000000,4000000,300000,40000000,200000,3000000,1000000,-,0\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("population.csv", csv_content)
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.content = buf.read()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = unhcr_client.get_global_trends(year_from=2020)
        assert len(df) == 1
        assert df["year"].iloc[0] == 2020