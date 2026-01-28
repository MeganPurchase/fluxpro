from pathlib import Path

import pytest
import polars as pl
from polars.testing import assert_frame_equal

from fluxpro.process import (
    detect_header_row,
    detect_separator,
    label_rows_by_time,
    remove_transition_minutes,
)


@pytest.mark.parametrize(
    "file,expected",
    [
        ("FTIR_0304.csv", ","),
        ("NOy_0404_CRED.csv", ","),
        ("2025_11_08_NO2_HONO_Channel1_Data.dat", "\t"),
        ("FTIR.log", "\t"),
        ("AIRYX.dat", "\t"),
        ("TELEDYNE.txt", ","),
    ],
)
def test_detect_separator_must_succeed_for_supported_formats(
    data_dir: Path, file: str, expected: int
):
    actual = detect_separator(data_dir / file)
    assert actual == expected


def test_detect_separator_must_throw_when_it_fails(tmp_path: Path):
    path = tmp_path / "test.txt"
    with open(path, "w") as file:
        file.write("x" * 50)

    with pytest.raises(ValueError):
        detect_separator(path)


@pytest.mark.parametrize(
    "file,separator,expected",
    [
        ("FTIR_0304.csv", ",", 2),
        ("NOy_0404_CRED.csv", ",", 0),
        ("2025_11_08_NO2_HONO_Channel1_Data.dat", "\t", 2),
        ("FTIR.log", "\t", 2),
        ("AIRYX.dat", "\t", 2),
        ("TELEDYNE.txt", ",", 0),
    ],
)
def test_detect_header_row_must_succeed_for_supported_formats(
    data_dir: Path, file: str, separator: str, expected: int
):
    actual = detect_header_row(data_dir / file, separator)
    assert actual == expected


@pytest.mark.parametrize(
    "file,separator",
    [
        ("FTIR_0304.csv", "\t"),
        ("NOy_0404_CRED.csv", "\t"),
        ("2025_11_08_NO2_HONO_Channel1_Data.dat", ","),
    ],
)
def test_detect_header_row_must_throw_for_incorrect_separator(
    data_dir: Path, file: str, separator: str
):
    with pytest.raises(ValueError):
        detect_header_row(data_dir / file, separator)


def test_label_rows_by_time_must_add_cycle_and_sample_columns():

    cycles = 3
    sample_time = 1
    samples = 2

    start = pl.datetime(2025, 1, 1, 12, 0, 0)
    datetime = pl.datetime_range(
        start=start,
        end=start + pl.duration(minutes=cycles * sample_time * samples),
        interval="30s",
        closed="left",
        eager=True,
    )

    lf = pl.LazyFrame({"datetime": datetime})

    lf = label_rows_by_time(lf, cycles, sample_time, samples)

    assert_frame_equal(
        lf,
        pl.LazyFrame(
            {
                "datetime": datetime,
                "cycle": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "sample": [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            }
        ),
    )


def test_remove_transition_minutes_must_remove_first_n_minutes_from_each_group():

    cycles = 2
    sample_time = 2
    samples = 2

    start = pl.datetime(2025, 1, 1, 12, 0, 0)
    datetime = pl.datetime_range(
        start=start,
        end=start + pl.duration(minutes=cycles * sample_time * samples),
        interval="30s",
        closed="left",
        eager=True,
    )

    lf = pl.LazyFrame(
        {
            "datetime": datetime,
            "cycle": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            "sample": [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    )

    lf = remove_transition_minutes(lf, 1)
    assert_frame_equal(
        lf,
        pl.LazyFrame(
            {
                "datetime": pl.concat(
                    [datetime[2:4], datetime[6:8], datetime[10:12], datetime[14:16]]
                ),
                "cycle": [1, 1, 1, 1, 2, 2, 2, 2],
                "sample": [1, 1, 2, 2, 1, 1, 2, 2],
            }
        ),
    )
