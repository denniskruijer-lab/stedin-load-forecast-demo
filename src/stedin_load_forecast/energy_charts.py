"""Client for the energy-charts.info public power API.

Source: https://api.energy-charts.info (Fraunhofer ISE). Chosen over
ENTSO-E's own Transparency Platform — the "official" source for this
data — because ENTSO-E requires emailing them for API access before you
can pull data programmatically, while energy-charts.info is free,
unauthenticated, and mirrors the same underlying grid data. For a same-day
build, the friction difference decided it. Data is licensed CC BY 4.0 —
attribute "Energy-Charts.info" (see ATTRIBUTION below) when displaying
results derived from it.

The API's `public_power` endpoint actually returns the full generation
mix (solar, wind, gas, ...) alongside a `"Load"` series and a `"Residual
load"` series. This client only extracts `"Load"` — actual grid demand —
since that's the forecasting target; the generation-mix data is fetched
but discarded, which is a bit wasteful bandwidth-wise but keeps this
client's surface area to exactly what the pipeline needs.
"""

import logging
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.energy-charts.info"
ATTRIBUTION = "Data: Energy-Charts.info (Fraunhofer ISE), CC BY 4.0"


class EnergyChartsError(RuntimeError):
    """The energy-charts API responded, but not with what we expected.

    A dedicated exception type (rather than letting a bare KeyError or
    IndexError bubble up) so callers can catch "the API's response shape
    surprised us" specifically, distinct from network failures (which
    surface as requests.RequestException) or programming errors.
    """


def fetch_load(
    start: date | str,
    end: date | str,
    country: str = "nl",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch the historical electricity load series for a country/date range.

    Args:
        start: Start date (inclusive), e.g. "2026-06-01" or a date object.
        end: End date (inclusive).
        country: energy-charts.info country code. Default "nl" since this
            demo targets Stedin's Dutch grid, but the API supports other
            countries unchanged.
        timeout: Request timeout in seconds. 30s is generous for a JSON
            response of this size; it exists mainly so a hung connection
            fails loudly instead of blocking the pipeline indefinitely.

    Returns:
        A DataFrame indexed by UTC timestamp ("timestamp") with a single
        "load_mw" column, at the API's native ~15-minute resolution. Rows
        with missing values are dropped (see _parse_load_response) rather
        than interpolated, since the pipeline's lag features need real,
        not invented, history.

    Raises:
        EnergyChartsError: if the response doesn't contain a "Load" series
            or timestamps in the expected shape.
        requests.HTTPError: if the request itself fails (network error,
            4xx/5xx status).
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
    """Turn the raw API JSON into a tidy load DataFrame.

    Split out from fetch_load() as a pure function (no network call) so
    the parsing logic can be unit tested directly against hand-built JSON
    fixtures, without needing to mock HTTP for every edge case (missing
    series, missing timestamps, null values).
    """
    load_data = _extract_series(payload, "Load")

    if "unix_seconds" not in payload:
        raise EnergyChartsError("Response is missing 'unix_seconds'")

    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    df = pd.DataFrame({"load_mw": load_data}, index=timestamps)
    df.index.name = "timestamp"
    # The API can return nulls for very recent timestamps that haven't
    # been finalized yet; dropping them keeps every remaining row genuine
    # rather than papering over gaps with an interpolated guess.
    return df.dropna()


def _extract_series(payload: dict, name: str) -> list:
    """Pull one named series (e.g. "Load") out of the API's production_types list."""
    for series in payload.get("production_types", []):
        if series.get("name") == name:
            return series["data"]
    raise EnergyChartsError(f"Series {name!r} not found in energy-charts response")
