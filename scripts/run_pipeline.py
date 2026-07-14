"""CLI entry point: fetch load data, train, evaluate, and plot.

This script is intentionally thin — it parses arguments, calls
run_pipeline() (the one function that knows how to actually do the work,
see stedin_load_forecast.pipeline), and prints the result. All logic
lives in the importable package under src/, not here, so the same
pipeline can be driven from a notebook, a test, or a future scheduled
job without going through argparse.

Usage: uv run python scripts/run_pipeline.py [--days 30] [--country nl] [--skip-garnish]
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

from stedin_load_forecast.logging_config import configure_logging
from stedin_load_forecast.pipeline import run_pipeline
from stedin_load_forecast.stedin_open_data import ATTRIBUTION as STEDIN_ATTRIBUTION
from stedin_load_forecast.stedin_open_data import (
    download_consumption_data,
    extract_consumption_csv,
    load_consumption_csv,
    top_woonplaatsen_by_connections,
)
from stedin_load_forecast.visualize import plot_top_woonplaatsen


def run_garnish_chart(output_dir: Path, cache_dir: Path) -> Path:
    """Download (if needed), aggregate, and chart Stedin's own open consumption data.

    Kept separate from run_pipeline() and skippable via --skip-garnish:
    it's a supporting visual, not part of the forecasting result, and it
    involves a ~4MB download that shouldn't be mandatory for every
    iteration while developing the core pipeline.

    Args:
        output_dir: Where to save the chart PNG.
        cache_dir: Where to cache the downloaded ZIP/CSV, so repeated runs
            don't re-download ~4MB every time.

    Returns:
        Path to the saved chart.
    """
    zip_path = download_consumption_data(cache_dir / "stedin_kleinverbruiksgegevens.zip")
    csv_path = extract_consumption_csv(zip_path, cache_dir)
    df = load_consumption_csv(csv_path)
    top = top_woonplaatsen_by_connections(df, n=10)
    return plot_top_woonplaatsen(
        top, output_dir / "top_woonplaatsen.png", attribution=STEDIN_ATTRIBUTION
    )


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

    print("15-minute resolution, GradientBoosting vs naive baseline:")
    print(
        f"  Model    -> MAE: {result.model_metrics['mae']:.2f} MW  "
        f"RMSE: {result.model_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.model_metrics['mape']:.2f}%"
    )
    print(
        f"  Baseline -> MAE: {result.naive_metrics['mae']:.2f} MW  "
        f"RMSE: {result.naive_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.naive_metrics['mape']:.2f}%"
    )
    print(f"  Plot saved to {result.plot_path}")

    print("Hourly resolution, SARIMA vs naive baseline (fixed 48h horizon):")
    print(
        f"  Model    -> MAE: {result.sarima_metrics['mae']:.2f} MW  "
        f"RMSE: {result.sarima_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.sarima_metrics['mape']:.2f}%"
    )
    print(
        f"  Baseline -> MAE: {result.sarima_naive_metrics['mae']:.2f} MW  "
        f"RMSE: {result.sarima_naive_metrics['rmse']:.2f} MW  "
        f"MAPE: {result.sarima_naive_metrics['mape']:.2f}%"
    )
    print(f"  Plot saved to {result.sarima_plot_path}")

    if not args.skip_garnish:
        garnish_path = run_garnish_chart(output_dir, cache_dir=Path("data/raw"))
        print(f"Stedin open-data chart saved to {garnish_path}")


if __name__ == "__main__":
    main()
