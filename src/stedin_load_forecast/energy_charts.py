"""Client for the energy-charts.info public power API.

Source: https://api.energy-charts.info (Fraunhofer ISE). Free, no
authentication required. Data licensed CC BY 4.0 — attribute
"Energy-Charts.info" when displaying results derived from it.
"""

import logging
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.energy-charts.info"
ATTRIBUTION = "Data: Energy-Charts.info (Fraunhofer ISE), CC BY 4.0"


class EnergyChartsError(RuntimeError):
    """Raised when the energy-charts API response doesn't contain what we expect."""


def fetch_load(
    start: date | str,
    end: date | str,
    country: str = "nl",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch the historical electricity load series for a country/date range.

    Returns a DataFrame indexed by UTC timestamp with a single "load_mw"
    column, at the API's native ~15-minute resolution.
    """
    params = {"country": country, "start": str(start), "end": str(end)}
    logger.info("Fetching load data: country=%s start=%s end=%s", country, start, end)

    response = requests.get(f"{API_BASE_URL}/public_power", params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    df = _parse_load_response(payload)
    logger.info("Fetched %d rows of load data", len(df))
    return df


def _parse_load_response(payload: dict) -> pd.DataFrame:
    load_data = _extract_series(payload, "Load")

    if "unix_seconds" not in payload:
        raise EnergyChartsError("Response is missing 'unix_seconds'")

    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({"load_mw": load_data}, index=timestamps)
    df.index.name = "timestamp"
    return df.dropna()


def _extract_series(payload: dict, name: str) -> list:
    for series in payload.get("production_types", []):
        if series.get("name") == name:
            return series["data"]
    raise EnergyChartsError(f"Series {name!r} not found in energy-charts response")
