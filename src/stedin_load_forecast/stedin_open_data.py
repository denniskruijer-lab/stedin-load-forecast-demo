"""Parses Stedin's own open small-consumer (kleinverbruik) data.

Source: https://www.stedin.net/zakelijk/open-data/verbruiksgegevens
CC BY 4.0. This is yearly, postcode-aggregated Standard Annual
Consumption (SJV) data — not a time series, so it isn't used as a
forecasting target. It's used here for one supporting chart, to show
the demo also engages with Stedin's own published data.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_DATA_URL = (
    "https://www.stedin.net/zakelijk/-/media/project/online/files/zakelijk/"
    "open-data/stedin_kleinverbruiksgegevens_01012020.zip"
)
ATTRIBUTION = "Data: Stedin open data (kleinverbruiksgegevens), CC BY 4.0"

# stedin.net returns 403 for the default python-requests User-Agent (WAF rule);
# a standard browser UA is accepted.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def download_consumption_data(
    cache_path: Path, url: str = DEFAULT_DATA_URL, timeout: float = 60.0
) -> Path:
    """Download the Stedin open-data ZIP to cache_path, unless already cached."""
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
    """Extract the single CSV from the Stedin open-data ZIP, if not already extracted."""
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


def load_consumption_csv(csv_source) -> pd.DataFrame:
    """Parse the Stedin kleinverbruik CSV: tab-separated, comma decimals."""
    return pd.read_csv(csv_source, sep="\t", decimal=",", quotechar='"')


def top_woonplaatsen_by_connections(
    df: pd.DataFrame, n: int = 10, product: str = "ELK"
) -> pd.DataFrame:
    """Average electricity consumption (SJV, connection-weighted) for the n
    woonplaatsen with the most connections in the dataset.

    Ranking by connection count (rather than by average consumption)
    avoids small-sample noise from postcodes with only a handful of
    connections dominating the result.
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
