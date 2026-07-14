"""Parses Stedin's own open small-consumer (kleinverbruik) data.

Source: https://www.stedin.net/zakelijk/open-data/verbruiksgegevens
CC BY 4.0. This is yearly, postcode-aggregated Standard Annual
Consumption (SJV) data — one row per postcode range per year, not a time
series — so it cannot be a forecasting target the way energy_charts'
load data is. It's used here for one supporting chart (see
top_woonplaatsen_by_connections + visualize.plot_top_woonplaatsen), to
show the demo also engages with Stedin's own published data, not only a
generic third-party API.

**Bug note, kept as documentation because it's non-obvious and easy to
reintroduce:** stedin.net's server returns HTTP 403 for the default
python-requests User-Agent string (a WAF rule blocking non-browser
clients), even though the exact same URL works fine in a browser or with
curl using a browser User-Agent. This was only caught by actually running
the pipeline against live data — a unit test with HTTP mocked away
would never exercise this. See REQUEST_HEADERS below and the test that
pins it.
"""

import logging
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_DATA_URL = (
    "https://www.stedin.net/zakelijk/-/media/project/online/files/zakelijk/"
    "open-data/stedin_kleinverbruiksgegevens_01012020.zip"
)
ATTRIBUTION = "Data: Stedin open data (kleinverbruiksgegevens), CC BY 4.0"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def download_consumption_data(
    cache_path: Path, url: str = DEFAULT_DATA_URL, timeout: float = 60.0
) -> Path:
    """Download the Stedin open-data ZIP to cache_path, unless already cached.

    The file is ~4MB, and this data only changes yearly — re-downloading
    it on every pipeline run would be slow and impolite to Stedin's
    server for no benefit, so a cache-if-present check comes first.

    Args:
        cache_path: Where to save (or look for) the ZIP.
        url: Source URL. Overridable mainly for testing.
        timeout: Request timeout in seconds; 60s because the file is
            larger than a typical API response.

    Returns:
        cache_path, whether it was just downloaded or already existed.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Using cached Stedin open data at %s", cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Stedin open data from %s", url)
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    logger.info("Cached Stedin open data at %s (%d bytes)", cache_path, len(response.content))
    return cache_path


def extract_consumption_csv(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the single CSV from the Stedin open-data ZIP, if not already extracted.

    Same caching rationale as download_consumption_data: extraction is
    cheap but pointless to repeat once the CSV is already sitting on disk.

    Args:
        zip_path: Path to the downloaded ZIP (see download_consumption_data).
        extract_dir: Directory to extract into.

    Returns:
        Path to the extracted CSV file.

    Raises:
        ValueError: if the ZIP contains no .csv file.
    """
    zip_path, extract_dir = Path(zip_path), Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path}")
        csv_name = csv_names[0]

        csv_path = extract_dir / csv_name
        if not csv_path.exists():
            logger.info("Extracting %s from %s", csv_name, zip_path)
            archive.extract(csv_name, extract_dir)

    return csv_path


def load_consumption_csv(csv_source: Path | str | IO[str]) -> pd.DataFrame:
    """Parse the Stedin kleinverbruik CSV into a DataFrame.

    The format is unusual enough to be worth spelling out: tab-separated
    (not comma — Stedin's export uses commas as decimal separators
    instead), and string fields are quoted. Accepts a path or an
    already-open file-like object (tests use the latter with an in-memory
    sample, rather than needing a real file on disk).

    Args:
        csv_source: Path to the CSV, or a file-like/StringIO object.

    Returns:
        The raw DataFrame, one row per (postcode range, product type)
        combination — see top_woonplaatsen_by_connections() for the
        aggregation typically applied next.
    """
    return pd.read_csv(csv_source, sep="\t", decimal=",", quotechar='"')


def top_woonplaatsen_by_connections(
    df: pd.DataFrame, n: int = 10, product: str = "ELK"
) -> pd.DataFrame:
    """Average electricity consumption for the n woonplaatsen with the most connections.

    Ranks by connection count (AANSLUITINGEN_AANTAL) rather than by raw
    average consumption, and weights the average by connection count
    rather than taking a plain mean across rows. Both choices exist for
    the same reason: the raw data has many rows representing only a
    handful of connections (a single short street), and naively ranking
    or averaging over those would let small-sample noise dominate the
    result — a street with 3 connections and one unusually high-usage
    household would otherwise outrank an entire city.

    Args:
        df: Output of load_consumption_csv().
        n: How many woonplaatsen to return.
        product: PRODUCTSOORT to filter on. Default "ELK" (electricity)
            — the dataset also contains "GAS" rows, which would corrupt
            the electricity average if left in.

    Returns:
        DataFrame with columns WOONPLAATS, avg_sjv_kwh (connection-
        weighted average standard annual consumption), and
        total_connections — sorted descending by total_connections.
    """
    subset = df[df["PRODUCTSOORT"] == product].copy()
    subset["weighted_sjv"] = subset["SJV_GEMIDDELD"] * subset["AANSLUITINGEN_AANTAL"]

    grouped = subset.groupby("WOONPLAATS").agg(
        total_connections=("AANSLUITINGEN_AANTAL", "sum"),
        weighted_sjv_sum=("weighted_sjv", "sum"),
    )
    grouped["avg_sjv_kwh"] = grouped["weighted_sjv_sum"] / grouped["total_connections"]

    top = grouped.sort_values("total_connections", ascending=False).head(n)
    return top[["avg_sjv_kwh", "total_connections"]].reset_index()
