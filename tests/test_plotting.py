import polars as pl

from fluxpro.plotting import plot_df


def test_plot_df(data_dir):
    df = pl.read_csv(data_dir / "expected/case1/FTIR_0304_2_out.csv", try_parse_dates=False)
    chart = plot_df(df)
    chart_data = chart.to_dict()
    assert chart_data["facet"]["field"] == "gas"
