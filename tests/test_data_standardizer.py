import polars as pl
import pytest
import datetime

from fluxpro.process import read_lf
from fluxpro.data_standardizer import DataStandardizer

testdata = [
    (
        "FTIR_0304.csv",
        ["NH3", "N2O", "O3", "CH4", "CO2", "CO", "NO2", "datetime"],
        datetime.timedelta(minutes=1),
        datetime.timedelta(minutes=1),
    ),
    (
        "NOy_0404_CRED.csv",
        ["NO", "NOY", "NOY-NO", "datetime"],
        datetime.timedelta(minutes=1),
        datetime.timedelta(minutes=1),
    ),
    (
        "2025_11_08_NO2_HONO_Channel1_Data.dat",
        ["NO2", "HONO", "CO2", "datetime"],
        datetime.timedelta(seconds=29),
        datetime.timedelta(minutes=7, seconds=2),
    ),
    (
        "FTIR.log",
        ["NH3", "N2O", "O3", "CH4", "CO2", "CO", "NO2", "datetime"],
        datetime.timedelta(minutes=1),
        datetime.timedelta(minutes=1),
    ),
    (
        "TELEDYNE.txt",
        ["NO", "NOY", "NOY-NO", "datetime"],
        datetime.timedelta(minutes=1),
        datetime.timedelta(minutes=1),
    ),
]


@pytest.mark.parametrize("input_file, columns, min_gap, max_gap", testdata)
def test_data_standardizer_must_handle_different_time_formats(
    data_dir,
    input_file: str,
    columns: list[str],
    min_gap: datetime.timedelta,
    max_gap: datetime.timedelta,
):
    standardizer = DataStandardizer()

    lf = read_lf(data_dir / input_file)
    df = standardizer.run(lf).collect()

    assert df.columns == columns

    s = df.select(pl.col("datetime").diff(null_behavior="drop")).to_series()
    assert s.min() == pytest.approx(min_gap)
    assert s.max() == pytest.approx(max_gap)


def test_data_standardizer_must_throw_for_unexpected_time_format():
    lf = pl.LazyFrame({"time": ["1/2/2025 11:00AM"]})
    standardizer = DataStandardizer()
    with pytest.raises(ValueError, match="unexpected datetime format"):
        standardizer.run(lf)


def test_data_standardizer_must_throw_for_unexpected_unit():
    lf = pl.LazyFrame({"time": ["10/02/2025 21:00", "10/02/2025 21:01"], "test gas (ppt)": [1, 2]})
    standardizer = DataStandardizer()
    standardizer.GAS_IDENTIFIERS = {"test gas (ppt)": "gas"}

    with pytest.raises(ValueError, match="failed to detect unit for column"):
        standardizer.run(lf)
