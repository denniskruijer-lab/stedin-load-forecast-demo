"""CLI entry point: fetch load data, train, evaluate, and plot.

Usage: uv run python scripts/run_pipeline.py [--days 30] [--country nl]
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

from stedin_load_forecast.logging_config import configure_logging
from stedin_load_forecast.pipeline import run_pipeline
from stedin_load_forecast.stedin_open_data import (
    download_consumption_data,
    extract_consumption_csv,
    load_consumption_csv,
    top_woonplaatsen_by_connections,
)
from stedin_load_forecast.visualize import plot_top_woonplaatsen


def run_garnish_chart(output_dir: Path, cache_dir: Path) -> Path:
    """Download (if needed) and chart Stedin's own open consumption data."""
    zip_path = download_consumption_data(cache_dir / "stedin_kleinverbruiksgegevens.zip")
    csv_path = extract_consumption_csv(zip_path, cache_dir)
    df = load_consumption_csv(csv_path)
    top = top_woonplaatsen_by_connections(df, n=10)
    return plot_top_woonplaatsen(top, output_dir / "top_woonplaatsen.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the load forecasting pipeline")
    parser.add_argument(
        "--days", type=int, default=30, help="Days of history to fetch (need >=8 for weekly lag)"
    )
    parser.add_argument("--country", type=str, default="nl")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--skip-garnish", action="store_true", help="Skip the Stedin open-data chart"
    )
    args = parser.parse_args()

    configure_logging()
    output_dir = Path(args.output_dir)

    end = date.today()
    start = end - timedelta(days=args.days)

    result = run_pipeline(start=start, end=end, country=args.country, output_dir=output_dir)

    print(
        f"Model    -> MAE: {result.model_metrics['mae']:.2f} MW  "
        f"RMSE: {result.model_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.model_metrics['mape']:.2f}%"
    )
    print(
        f"Baseline -> MAE: {result.naive_metrics['mae']:.2f} MW  "
        f"RMSE: {result.naive_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.naive_metrics['mape']:.2f}%"
    )
    print(f"Plot saved to {result.plot_path}")

    if not args.skip_garnish:
        garnish_path = run_garnish_chart(output_dir, cache_dir=Path("data/raw"))
        print(f"Stedin open-data chart saved to {garnish_path}")


if __name__ == "__main__":
    main()
