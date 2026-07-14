"""CLI entry point: fetch load data, train, evaluate, and plot.

Usage: uv run python scripts/run_pipeline.py [--days 30] [--country nl]
"""

import argparse
from datetime import date, timedelta

from stedin_load_forecast.logging_config import configure_logging
from stedin_load_forecast.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the load forecasting pipeline")
    parser.add_argument(
        "--days", type=int, default=30, help="Days of history to fetch (need >=8 for weekly lag)"
    )
    parser.add_argument("--country", type=str, default="nl")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    configure_logging()

    end = date.today()
    start = end - timedelta(days=args.days)

    result = run_pipeline(
        start=start, end=end, country=args.country, output_dir=args.output_dir
    )

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


if __name__ == "__main__":
    main()
