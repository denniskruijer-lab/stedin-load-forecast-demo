# Stedin Load Forecast Demo

Short-term electricity load forecasting, built as a small, production-minded demo for a Data Scientist interview at Stedin.

**Status:** scaffolded, model and pipeline to be built 2026-07-14.

## Problem

Stedin's Data Science team forecasts load over time on transport and distribution networks to plan capacity and manage congestion. This repo demonstrates a small end-to-end forecasting pipeline for that class of problem, built with an emphasis on clean, testable, CI-backed code rather than notebook sprawl.

## Data

- **Core:** hourly electricity load data from [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE, public, no registration required).
- **Context:** Stedin's own open consumption data ([stedin.net/zakelijk/open-data](https://www.stedin.net/zakelijk/open-data)), CC-BY 4.0, used for a supporting visual rather than as a forecasting target (it's yearly and postcode-aggregated, not a time series).

## Approach

_To be filled in during the build: model choice, validation strategy, error metrics._

## Project structure

```
src/stedin_load_forecast/   # importable package — pipeline logic lives here, not in notebooks
tests/                       # pytest suite
.github/workflows/           # CI: lint + test on push
```

## Setup & run

_To be filled in during the build._

## Testing

```
pytest
```

## CI

GitHub Actions runs lint and tests on every push. See `.github/workflows/`.

## Production next steps

This demo intentionally stops short of a full production setup (no OTAP environments, no A/B testing infrastructure, no feature store) — those are discussed as part of the interview rather than simulated here, to keep the demo itself proportionate to what it's actually demonstrating.
