# Stedin Load Forecast Demo

![CI](https://github.com/denniskruijer-lab/stedin-load-forecast-demo/actions/workflows/ci.yml/badge.svg)

Short-term electricity load forecasting for the Netherlands, built as a small, production-minded reference implementation of the kind of problem Stedin's Data Science team works on. The point of this repo isn't to be the most sophisticated model possible — it's to be a complete, honest example of the whole loop: real public data in, a defensible model out, validated against a baseline, wrapped in tests/CI/logging so someone else could pick it up and trust it.

## Overview

Stedin (and grid operators generally) need to forecast electricity load on their networks to plan capacity and manage congestion. This repo tackles a scaled-down version of that problem: given historical grid load for the Netherlands, forecast near-term load and prove the forecast is actually useful (i.e. it beats the trivial "assume tomorrow looks like today" baseline).

Everything here follows one guiding principle: **proportionate engineering**. Every piece of infrastructure (tests, CI, logging, caching) solves a real problem this specific project has — nothing was added because it would look impressive. Where a "real" production system would need more (a feature store, A/B testing, blue/green deploys), that's called out explicitly rather than half-simulated. See [Design decisions](#design-decisions) and [Production next steps](#production-next-steps).

## Architecture — where to find what

The package (`src/stedin_load_forecast/`) is organized as one module per responsibility. Each module is independently testable and (mostly) has no idea the others exist — `pipeline.py` is the only place that wires them together in order.

| Module | Responsibility | Key functions |
|---|---|---|
| `energy_charts.py` | Fetch NL grid load from the energy-charts.info API | `fetch_load()` |
| `stedin_open_data.py` | Fetch/parse Stedin's own open consumption data | `download_consumption_data()`, `top_woonplaatsen_by_connections()` |
| `features.py` | Turn raw load data into model features | `build_features()`, `feature_columns()` |
| `model.py` | Train the regressor, compute the naive baseline, evaluate both | `train_regressor()`, `seasonal_naive_predict()`, `evaluate()` |
| `pipeline.py` | Orchestrate the above into one end-to-end run | `run_pipeline()` |
| `visualize.py` | Turn results into PNGs | `plot_forecast_comparison()`, `plot_top_woonplaatsen()` |
| `logging_config.py` | Console + file logging setup | `configure_logging()` |

`scripts/run_pipeline.py` is the CLI — it's a thin wrapper around `pipeline.run_pipeline()`, so the same logic is exercised whether you run it from the command line, from a test, or (eventually) from a scheduler.

Every module above has a module-level docstring explaining not just *what* it does but *why* it's built the way it is — that's the first place to look before reading the code itself.

## Data

- **Core (forecasting target):** electricity load for the Netherlands from [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE), via the `public_power` endpoint's `"Load"` series — ~15-minute resolution, free, no registration. Chosen over ENTSO-E's own Transparency Platform (the "official" source) because ENTSO-E requires emailing them for API access before you can pull data programmatically; energy-charts.info mirrors the same grid data with zero friction. CC BY 4.0 (attribution is printed on the chart itself — see `visualize.ATTRIBUTION` usage).
- **Context (supporting visual, not a forecasting input):** Stedin's own open consumption data ([stedin.net/zakelijk/open-data](https://www.stedin.net/zakelijk/open-data)), CC BY 4.0. This is yearly, postcode-aggregated Standard Annual Consumption — one row per postcode range per year, not a time series — so it can't be a forecasting target. It's used for one chart (average consumption for Stedin's largest service-area woonplaatsen), to show the demo also engages with Stedin's own published data rather than only a generic third-party API.

## Approach

- **Features** (`features.py`): calendar features (hour, day-of-week, weekend flag) and lag features (15min/1h/1day/1week back). Load is dominated by daily and weekly seasonality — these two cheap, interpretable feature families capture almost all of that structure without needing something heavier like Fourier terms.
- **Baseline** (`model.seasonal_naive_predict`): predict the same value as the same time yesterday (`lag_96`). Any model complex enough to need training has to prove it beats this — "8.7% MAPE" means little on its own, but "a third of the naive baseline's error" is immediately legible to a non-ML stakeholder.
- **Model** (`model.train_regressor`): `GradientBoostingRegressor` (scikit-learn). Chosen over a heavier time-series-specific approach (ARIMA, Prophet, a neural forecaster) because it handles non-linear feature interactions without manual feature crosses, needs no feature scaling, and is a standard, easy-to-explain, production-idiomatic choice — appropriate for the scope of this demo.
- **Validation** (`model.time_train_test_split`): chronological split, no shuffling. Random shuffling would leak near-future information into training via adjacent, highly-correlated rows (because of the lag features) and make the reported accuracy look better than it would in real deployment.
- **Metrics** (`model.evaluate`): MAE, RMSE, and MAPE, always reported for both the model and the baseline side by side. Each metric answers a different question — see the docstring in `model.py` for the detail.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest
uv run python scripts/run_pipeline.py --days 30
```

This fetches the last 30 days of NL load data, trains the model, evaluates it against the naive baseline, and writes `outputs/forecast_comparison.png` and `outputs/top_woonplaatsen.png`. Skip the second chart (and its ~4MB download) with `--skip-garnish`.

### Example result

Run against live data on 2026-07-14 (30 days, NL):

| | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Naive baseline (yesterday) | 779.3 | 1071.9 | 25.8% |
| Model (GradientBoosting) | **159.6** | **258.7** | **8.7%** |

![Forecast vs actual](docs/forecast_comparison_example.png)

![Top woonplaatsen by average consumption](docs/top_woonplaatsen_example.png)

## Project structure

```
src/stedin_load_forecast/   # importable package — see the architecture table above
scripts/run_pipeline.py     # CLI entry point (thin wrapper around pipeline.run_pipeline)
tests/                      # pytest suite, one test file per src module
.github/workflows/ci.yml    # CI: ruff + pytest on every push/PR
docs/                       # example output images referenced in this README
data/raw/                   # gitignored — cached downloads (Stedin open data)
outputs/                    # gitignored — chart output from running the pipeline
logs/                       # gitignored — logs/pipeline.log, see Logging below
```

## Design decisions

A quick-reference for the "why" behind the infrastructure choices, each expanded further in the relevant module's docstring:

- **uv + a committed `uv.lock`** for dependency management — fast installs, and the lockfile means "works on my machine" reproduces exactly on any other machine or in CI, not approximately.
- **Python pinned to 3.12** (`.python-version`) rather than whatever's newest on the build machine — avoids the risk of core data-science packages not yet having wheels for a brand-new interpreter version.
- **Chronological git history**: feature branches → `development` → `master`, each merged as a real merge commit (not squashed) with its own tests/docs/logging bundled in, each CI-verified before merging. The commit history itself is meant to be legible, not just the final code.
- **No OTAP, no A/B testing infrastructure, no feature store** in this repo — deliberately. Those solve problems a single-model demo with no real deployment target doesn't have; simulating them here would be complexity theatre, not engineering. See [Production next steps](#production-next-steps).
- **Two bugs caught during manual end-to-end runs, not by the unit tests** — worth calling out because it's a reminder that mocked-HTTP tests can't catch everything:
  1. `logging.basicConfig()` silently no-ops on any call after the first, unless `force=True` is passed — caught by a test, fixed, now impossible to regress silently.
  2. `stedin.net` returns HTTP 403 for the default `python-requests` User-Agent (a WAF rule) — only surfaced by actually running the pipeline against live data. Fixed with a browser-like User-Agent header, now pinned by a test assertion so it can't quietly break again.

## Testing

```bash
uv run pytest
```

Every HTTP call is mocked (`responses` / `unittest.mock`) — the suite never touches the network, so it's fast and deterministic in CI. That's also why the two bugs above weren't caught by tests alone: they only show up when the pipeline actually talks to the real APIs, which is why running `scripts/run_pipeline.py` against live data was a required verification step, not just a demo nicety, at every stage of the build.

## Logging

`logging_config.configure_logging()` sets up two outputs, each with a different job:

- **Console** — the full run narrative at INFO level, for watching a run happen.
- **`logs/pipeline.log`** (gitignored) — only WARNING and above. Kept separate from the console stream so that checking "did anything go wrong" doesn't mean scanning past hundreds of routine INFO lines — the file itself *is* the alert list, following Python's standard severity hierarchy (WARNING < ERROR < CRITICAL).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` on every push and pull request against `development` and `master`, using `astral-sh/setup-uv` for fast, lockfile-exact installs. Every merge in this repo's history happened only after this went green.

## Production next steps

This demo intentionally stops short of a full production setup:

- **OTAP environments** — not simulated; there's no real deployment target for a single-model demo to move through.
- **A/B testing infrastructure** — not simulated; there's no live traffic to split.
- **Feature store / retraining pipeline** — not built; the data here is small enough that re-fetching and re-training from scratch each run is fine, which wouldn't hold at production scale.
- **Containerization** — not yet included (a bare `Dockerfile` was considered and deliberately deferred rather than added for completeness's sake with no corresponding need).

These are exactly the kind of decisions worth walking through in a conversation rather than half-building in a repo nobody will run in anger — over-building this demo to simulate all of them would itself be the kind of complexity this repo is trying to avoid.
