import pytest
import responses

from stedin_load_forecast.energy_charts import (
    API_BASE_URL,
    EnergyChartsError,
    _parse_load_response,
    fetch_load,
)


def _sample_payload(load_values=(11292.1, 11347.0, 11118.1)):
    unix_seconds = [1783634400 + 900 * i for i in range(len(load_values))]
    return {
        "unix_seconds": unix_seconds,
        "production_types": [
            {"name": "Solar", "data": [0.0] * len(load_values), "deprecated": False},
            {"name": "Load", "data": list(load_values), "deprecated": False},
        ],
        "deprecated": False,
    }


def test_parse_load_response_extracts_load_series_indexed_by_timestamp():
    df = _parse_load_response(_sample_payload())

    assert list(df.columns) == ["load_mw"]
    assert len(df) == 3
    assert df["load_mw"].iloc[0] == 11292.1
    assert df.index.name == "timestamp"
    assert df.index.tz is not None


def test_parse_load_response_drops_missing_values():
    payload = _sample_payload(load_values=(11292.1, None, 11118.1))

    df = _parse_load_response(payload)

    assert len(df) == 2


def test_parse_load_response_raises_when_load_series_missing():
    payload = _sample_payload()
    payload["production_types"] = [p for p in payload["production_types"] if p["name"] != "Load"]

    with pytest.raises(EnergyChartsError):
        _parse_load_response(payload)


@responses.activate
def test_fetch_load_calls_public_power_endpoint_and_parses_response():
    responses.add(
        responses.GET,
        f"{API_BASE_URL}/public_power",
        json=_sample_payload(),
        status=200,
    )

    df = fetch_load(start="2026-07-10", end="2026-07-10", country="nl")

    assert len(df) == 3
    request = responses.calls[0].request
    assert "country=nl" in request.url
    assert "start=2026-07-10" in request.url
