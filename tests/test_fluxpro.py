import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from fluxpro.main import (
    assign_row_labels,
    remove_transition_minutes,
    filter_relevant_columns,
    reformat_data,
    subtract_blank,
    compute_averages,
    calculate_flux,
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
    df_sub = subtract_blank(df, blank=1)
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


def test_pipeline_integration(tmp_path):
    """Test a minimal version of the pipeline with fake data."""
    df = pd.DataFrame({"ppm (cal) CO2": np.arange(12)})
    df = assign_row_labels(df, cycles=2, sample_time=3, samples=2)
    df = remove_transition_minutes(df, sample_time=3, buffer=1)
    df = filter_relevant_columns(df)
    df = reformat_data(df)
    df = subtract_blank(df, blank=1)
    avg = compute_averages(df)
    avg = calculate_flux(avg, flow=1.0, chamber_volume=0.1, soil_surface_area=0.01)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Simulate write_output manually
    avg.to_csv(out_dir / "test.csv", index=False)

    assert (out_dir / "test.csv").exists()
