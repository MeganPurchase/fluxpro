from argparse import Namespace
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from fluxpro.main import (
    assign_row_labels,
    assign_row_labels_by_time,
    remove_transition_minutes,
    filter_relevant_columns,
    reformat_data,
    subtract_blank,
    compute_averages,
    calculate_flux,
    run,
)


@pytest.fixture
def sample_df():
    """Create a small dummy dataset mimicking instrument output."""
    data = {
        "ppm (cal) CO2": [1, 2, 3, 4, 5, 6],
        "Conc CH4": [10, 11, 12, 13, 14, 15],
    }
    df = pd.DataFrame(data)
    return df


def test_assign_row_labels(sample_df):
    df = assign_row_labels(sample_df.copy(), cycles=2, sample_time=3, samples=1)
    assert "cycle" in df.columns
    assert "sample" in df.columns
    # There are 6 rows = 2 cycles × 3 minutes × 1 sample
    assert df["cycle"].tolist() == [1, 1, 1, 2, 2, 2]
    assert df["sample"].tolist() == [1, 1, 1, 1, 1, 1]


def test_assign_row_labels_by_time():
    data = {
        "time": [
            "2025-11-08 00:00:21",
            "2025-11-08 00:00:51",
            "2025-11-08 00:01:21",
            "2025-11-08 00:01:51",
            "2025-11-08 00:02:20",
            "2025-11-08 00:02:50",
            "2025-11-08 00:03:20",
            "2025-11-08 00:03:50",
            "2025-11-08 00:04:20",
            "2025-11-08 00:04:50",
            "2025-11-08 00:05:20",
            "2025-11-08 00:05:50",
            "2025-11-08 00:06:20",
            "2025-11-08 00:06:49",
            "2025-11-08 00:07:19",
            "2025-11-08 00:07:49",
            "2025-11-08 00:08:19",
            "2025-11-08 00:08:49",
            "2025-11-08 00:09:19",
            "2025-11-08 00:09:49",
        ]
    }
    df = pd.DataFrame(data)

    df = assign_row_labels_by_time(df, time_col="time", cycles=3, sample_time=1, samples=2)
    assert np.isclose(
        df["sample"], [1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2]
    ).all()
    assert np.isclose(
        df["cycle"], [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    ).all()


def test_remove_transition_minutes(sample_df):
    df = assign_row_labels(sample_df.copy(), cycles=1, sample_time=6, samples=1)
    df_trimmed = remove_transition_minutes(df, sample_time=6, buffer=2)
    # Expect last 4 rows per group (since buffer=2)
    assert len(df_trimmed) == 4
    assert df_trimmed["cycle"].nunique() == 1


def test_filter_relevant_columns(sample_df):
    df = assign_row_labels(sample_df.copy(), cycles=1, sample_time=3, samples=2)
    filtered = filter_relevant_columns(df)
    # Should only keep relevant columns
    assert all(col in filtered.columns for col in ["ppm (cal) CO2", "Conc CH4"])


def test_reformat_data(sample_df):
    df = assign_row_labels(sample_df.copy(), cycles=1, sample_time=3, samples=2)
    df = filter_relevant_columns(df)
    melted = reformat_data(df)
    assert set(melted.columns) == {"cycle", "sample", "gas", "value"}
    assert "ppm (cal) CO2" in melted["gas"].values


def test_subtract_blank():
    df = pd.DataFrame(
        {
            "cycle": [1, 1, 1, 1],
            "sample": [1, 2, 1, 2],
            "gas": ["CO2", "CO2", "CH4", "CH4"],
            "value": [5, 10, 20, 30],
        }
    )
    # sample 1 = blank
    df_sub = subtract_blank(df, blank_index=1)
    assert "value_reduced" in df_sub.columns
    # Non-blank samples remain
    assert all(df_sub["sample"] != 1)
    # Check subtraction is correct
    mean_blank_co2 = df[df["sample"] == 1].query("gas == 'CO2'")["value"].mean()
    co2_row = df_sub.query("sample == 2 and gas == 'CO2'").iloc[0]
    assert np.isclose(co2_row["value_reduced"], 10 - mean_blank_co2)


def test_compute_averages():
    df = pd.DataFrame(
        {
            "cycle": [1, 1, 1, 1],
            "sample": [1, 1, 1, 1],
            "gas": ["CO2", "CO2", "CO2", "CO2"],
            "value_reduced": [1, 2, 3, 4],
        }
    )
    res = compute_averages(df)
    assert np.isclose(res["mean"].iloc[0], 2.5)
    assert np.isclose(res["std"].iloc[0], np.std([1, 2, 3, 4], ddof=1))


def test_calculate_flux():
    df = pd.DataFrame(
        {
            "mean": [1.0, 2.0],
            "cycle": [1, 1],
            "sample": [1, 2],
            "gas": ["CO2", "CH4"],
        }
    )
    res = calculate_flux(df, flow=2.0, chamber_volume=0.01, soil_surface_area=0.005)
    assert "flux" in res.columns
    assert np.isclose(res["flux"].iloc[0], 1.0 * 2.0 * (0.01 / 0.005))


def assert_csv_equal_or_update(actual_path, expected_path, update=False, **read_csv_kwargs):
    """
    Compare a generated CSV file to an expected one, or update the expected file if --update is used.
    """
    actual_df = pd.read_csv(actual_path, **read_csv_kwargs)

    if update:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        actual_df.to_csv(expected_path, index=False)
        print(f"✅ Updated expected file: {expected_path}")
        return

    expected_df = pd.read_csv(expected_path, **read_csv_kwargs)
    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, check_like=True)


def test_pipeline_integration_1(tmp_path, request):

    args = Namespace(
        input_file="tests/FTIR_0304.csv",
        cycles=22,
        samples=6,
        sample_time=10,
        blank=1,
        flow=1,
        chamber_volume=1,
        soil_surface_area=1,
        out=tmp_path,
        header=2,
        buffer=2,
    )

    run(args)

    update = request.config.getoption("--update")

    for gas in [
        "ammonia",
        "carbon_dioxide",
        "methane",
        "nitrogen_dioxide",
        "nitrous_oxide",
        "ozone",
    ]:
        for suffix in ["_all.csv", "_avg.csv"]:
            actual = tmp_path / f"{gas}{suffix}"
            expected = Path(f"tests/expected/case1/{gas}{suffix}")
            assert_csv_equal_or_update(actual, expected, update=update)


def test_pipeline_integration_2(tmp_path, request):

    args = Namespace(
        input_file="tests/NOy_0404_CRED.csv",
        cycles=21,
        samples=6,
        sample_time=10,
        blank=1,
        flow=1,
        chamber_volume=1,
        soil_surface_area=1,
        out=tmp_path,
        header=0,
        buffer=2,
    )

    run(args)

    update = request.config.getoption("--update")

    for gas in ["no", "noy-no", "noy"]:
        for suffix in ["_all.csv", "_avg.csv"]:
            actual = tmp_path / f"{gas}{suffix}"
            expected = Path(f"tests/expected/case2/{gas}{suffix}")
            assert_csv_equal_or_update(actual, expected, update=update)


def test_pipeline_integration_3(tmp_path, request):

    args = Namespace(
        input_file="tests/2025_11_08_NO2_HONO_Channel1_Data.dat",
        cycles=24,
        samples=2,
        sample_time=30,
        blank=1,
        flow=1,
        chamber_volume=1,
        soil_surface_area=1,
        out=tmp_path,
        header=2,
        buffer=5,
    )

    run(args)

    update = request.config.getoption("--update")

    for gas in ["co2", "hono", "no2"]:
        for suffix in ["_all.csv", "_avg.csv"]:
            actual = tmp_path / f"{gas}{suffix}"
            expected = Path(f"tests/expected/case3/{gas}{suffix}")
            assert_csv_equal_or_update(actual, expected, update=update)
