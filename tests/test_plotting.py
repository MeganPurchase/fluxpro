import json
from pathlib import Path

import polars as pl

from fluxpro.plotting import plot_df


def test_plot_df(data_dir: Path, update_expected: bool):
    df = pl.read_csv(data_dir / "expected/case1/FTIR_0304_2_out.csv", try_parse_dates=False)
    chart = plot_df(df)

    chart = chart.to_dict(format="vega")

    if update_expected:
        with open(data_dir / "expected/plot.json", "w") as f:
            json.dump(chart, f, indent=2)

    with open(data_dir / "expected/plot.json", "r") as f:
        expected = json.load(f)

    assert chart == expected
