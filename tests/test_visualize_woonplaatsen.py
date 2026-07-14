import pandas as pd

from stedin_load_forecast.visualize import plot_top_woonplaatsen


def test_plot_top_woonplaatsen_writes_a_nonempty_image(tmp_path):
    df = pd.DataFrame(
        {
            "WOONPLAATS": ["ROTTERDAM", "UTRECHT", "AMSTERDAM"],
            "avg_sjv_kwh": [3600.0, 4100.0, 3900.0],
            "total_connections": [1000, 800, 1200],
        }
    )

    output_path = tmp_path / "charts" / "woonplaatsen.png"
    result_path = plot_top_woonplaatsen(df, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
