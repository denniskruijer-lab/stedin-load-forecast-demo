import zipfile
from io import StringIO
from unittest.mock import patch

import responses

from stedin_load_forecast.stedin_open_data import (
    DEFAULT_DATA_URL,
    download_consumption_data,
    extract_consumption_csv,
    load_consumption_csv,
    top_woonplaatsen_by_connections,
)

# A small, hand-built sample matching the real Stedin kleinverbruik CSV
# format: tab-separated, quoted string fields, comma decimals.
SAMPLE_CSV = (
    "NETBEHEERDER\tNETGEBIED\tSTRAATNAAM\tPOSTCODE_VAN\tPOSTCODE_TOT\tWOONPLAATS\t"
    "LANDCODE\tPRODUCTSOORT\tVERBRUIKSSEGMENT\tAANSLUITINGEN_AANTAL\t"
    "LEVERINGSRICHTING_PERC\tFYSIEKE_STATUS_PERC\tSOORT_AANSLUITING_PERC\t"
    "SOORT_AANSLUITING\tSJV_GEMIDDELD\tSJV_LAAG_TARIEF_PERC\tSLIMME_METER_PERC\t"
    "STANDAARDDEVIATIE\n"
    '"8716874000009"\t"Stedin Utrecht"\t"Kerkstraat"\t"1234AA"\t"1234AA"\t"ROTTERDAM"\t'
    '"NL"\t"ELK"\t"KVB"\t100\t90,00\t100,00\t50,00\t"3x25"\t4000,00\t90,00\t80,00\t1500,0\n'
    '"8716874000009"\t"Stedin Utrecht"\t"Dorpsstraat"\t"1234AB"\t"1234AB"\t"ROTTERDAM"\t'
    '"NL"\t"ELK"\t"KVB"\t50\t90,00\t100,00\t50,00\t"3x25"\t3000,00\t90,00\t80,00\t1500,0\n'
    '"8716874000009"\t"Stedin Utrecht"\t"Kerkstraat"\t"1234AA"\t"1234AA"\t"ROTTERDAM"\t'
    '"NL"\t"GAS"\t"KVB"\t100\t90,00\t100,00\t50,00\t"3x25"\t1200,00\t90,00\t80,00\t500,0\n'
    '"8716874000009"\t"Stedin Utrecht"\t"Hoofdweg"\t"5678CD"\t"5678CD"\t"UTRECHT"\t'
    '"NL"\t"ELK"\t"KVB"\t10\t90,00\t100,00\t50,00\t"3x25"\t10000,00\t90,00\t80,00\t1500,0\n'
)


def test_load_consumption_csv_parses_tab_separated_comma_decimal_format():
    df = load_consumption_csv(StringIO(SAMPLE_CSV))

    assert len(df) == 4
    assert df.loc[0, "WOONPLAATS"] == "ROTTERDAM"
    assert df.loc[0, "SJV_GEMIDDELD"] == 4000.00
    assert df.loc[0, "AANSLUITINGEN_AANTAL"] == 100


def test_top_woonplaatsen_by_connections_filters_product_and_weights_by_connections():
    df = load_consumption_csv(StringIO(SAMPLE_CSV))

    result = top_woonplaatsen_by_connections(df, n=10, product="ELK")

    # GAS row excluded; Rotterdam has 150 connections (100+50) vs Utrecht's 10
    assert set(result["WOONPLAATS"]) == {"ROTTERDAM", "UTRECHT"}
    rotterdam = result.set_index("WOONPLAATS").loc["ROTTERDAM"]
    assert rotterdam["total_connections"] == 150
    # weighted average: (100*4000 + 50*3000) / 150
    assert round(rotterdam["avg_sjv_kwh"], 2) == round((100 * 4000 + 50 * 3000) / 150, 2)


def test_top_woonplaatsen_by_connections_respects_n_and_ranks_by_connections():
    df = load_consumption_csv(StringIO(SAMPLE_CSV))

    result = top_woonplaatsen_by_connections(df, n=1, product="ELK")

    assert len(result) == 1
    assert result.iloc[0]["WOONPLAATS"] == "ROTTERDAM"


def test_download_consumption_data_uses_cache_when_present(tmp_path):
    cache_path = tmp_path / "stedin_kv.zip"
    cache_path.write_bytes(b"already-cached-bytes")

    with patch("stedin_load_forecast.stedin_open_data.requests.get") as mock_get:
        result = download_consumption_data(cache_path)

    mock_get.assert_not_called()
    assert result == cache_path
    assert cache_path.read_bytes() == b"already-cached-bytes"


@responses.activate
def test_download_consumption_data_downloads_when_not_cached(tmp_path):
    cache_path = tmp_path / "nested" / "stedin_kv.zip"
    responses.add(responses.GET, DEFAULT_DATA_URL, body=b"zip-bytes", status=200)

    result = download_consumption_data(cache_path)

    assert result == cache_path
    assert cache_path.read_bytes() == b"zip-bytes"
    # stedin.net returns 403 for the default python-requests User-Agent;
    # a browser-like UA must be sent (caught by manual testing, not by a
    # mocked-away test — this assertion pins the fix so it can't regress).
    assert "python-requests" not in responses.calls[0].request.headers["User-Agent"]


def test_extract_consumption_csv_pulls_csv_out_of_zip(tmp_path):
    zip_path = tmp_path / "stedin_kv.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Stedin_kleinverbruikgegevens.csv", SAMPLE_CSV)

    extract_dir = tmp_path / "extracted"
    csv_path = extract_consumption_csv(zip_path, extract_dir)

    assert csv_path.name == "Stedin_kleinverbruikgegevens.csv"
    assert csv_path.exists()
    assert csv_path.read_text() == SAMPLE_CSV


def test_extract_consumption_csv_reuses_already_extracted_file(tmp_path):
    zip_path = tmp_path / "stedin_kv.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("data.csv", SAMPLE_CSV)

    extract_dir = tmp_path / "extracted"
    first = extract_consumption_csv(zip_path, extract_dir)
    # Overwrite the extracted file to prove the second call doesn't re-extract
    first.write_text("not the original content")

    second = extract_consumption_csv(zip_path, extract_dir)

    assert second == first
    assert second.read_text() == "not the original content"
