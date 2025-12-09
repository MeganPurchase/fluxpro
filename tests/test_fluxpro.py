import polars as pl
import numpy as np
import pytest
from pathlib import Path
from argparse import Namespace

from fluxpro.main import (
    label_rows_by_time,
    remove_transition_minutes,
    filter_relevant_columns,
    unpivot,
    subtract_blank,
    compute_statistics,
    compute_flux,
    process_file,
)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Small dummy dataset representing instrument output."""
    return pl.DataFrame(
        {
            "ppm (cal) CO2": [1, 2, 3, 4, 5, 6],
            "Conc CH4": [10, 11, 12, 13, 14, 15],
        }
    )


def test_label_rows_by_time():
    df = pl.DataFrame(
        {
            "time": [
                "2025-11-08 00:00:21",
                "2025-11-08 00:00:51",
                "2025-11-08 00:01:21",
                "2025-11-08 00:01:51",
                "2025-11-08 00:02:20",
            ]
        }
    )

    lf = df.lazy()
    lf2 = label_rows_by_time(lf, cycles=3, sample_time=1, samples=2)
    out = lf2.collect()

    assert "cycle" in out.columns
    assert "sample" in out.columns
    assert out["cycle"].dtype == pl.Int64()
    assert out["sample"].dtype == pl.Int64()


def test_remove_transition_minutes(sample_df: pl.DataFrame):
    # Add fake cycle/sample
    df = sample_df.with_columns(
        cycle=pl.lit(1),
        sample=pl.lit(1),
    )

    lf = df.lazy()
    lf_trimmed = remove_transition_minutes(lf, sample_time=6, buffer=2)

    out = lf_trimmed.collect()
    assert len(out) == 4
    assert out["cycle"].n_unique() == 1


def test_filter_relevant_columns(sample_df: pl.DataFrame):
    df = sample_df.with_columns(cycle=1, sample=1)
    lf = df.lazy()

    out = filter_relevant_columns(lf).collect()

    assert "ppm (cal) CO2" in out.columns
    assert "Conc CH4" in out.columns


def test_unpivot(sample_df: pl.DataFrame):
    df = sample_df.with_columns(cycle=1, sample=1)
    lf = df.lazy()

    melted = unpivot(lf).collect()

    assert set(melted.columns) == {"cycle", "sample", "gas", "value"}
    assert "ppm (cal) CO2" in melted["gas"].unique().to_list()


def test_subtract_blank():
    df = pl.DataFrame(
        {
            "cycle": [1, 1, 1, 1],
            "sample": [1, 2, 1, 2],
            "gas": ["CO2", "CO2", "CH4", "CH4"],
            "value": [5, 10, 20, 30],
        }
    )

    lf = df.lazy()
    out = subtract_blank(lf, blank_index=1).collect()

    assert "value_reduced" in out.columns
    assert (out["sample"] != 1).all()

    # Check subtraction
    blank_mean_co2 = df.filter((pl.col("sample") == 1) & (pl.col("gas") == "CO2"))["value"].mean()
    test_row = out.filter((pl.col("sample") == 2) & (pl.col("gas") == "CO2")).row(0)
    assert np.isclose(test_row[out.columns.index("value_reduced")], 10 - blank_mean_co2)


def test_compute_statistics():
    df = pl.DataFrame(
        {
            "cycle": [1, 1, 1, 1],
            "sample": [1, 1, 1, 1],
            "gas": ["CO2", "CO2", "CO2", "CO2"],
            "value_reduced": [1.0, 2.0, 3.0, 4.0],
        }
    )

    lf = df.lazy()
    out = compute_statistics(lf).collect()

    assert np.isclose(out["mean"][0], 2.5)
    assert np.isclose(out["std"][0], np.std([1, 2, 3, 4], ddof=1))


def test_compute_flux():
    df = pl.DataFrame(
        {
            "mean": [1.0, 2.0],
            "cycle": [1, 1],
            "sample": [1, 2],
            "gas": ["CO2", "CH4"],
        }
    )

    lf = df.lazy()
    out = compute_flux(lf, flow=2.0, chamber_volume=0.01, soil_surface_area=0.005).collect()

    assert "flux" in out.columns
    assert np.isclose(out["flux"][0], 1.0 * 2.0 * (0.01 / 0.005))


def read_csv_polars(path: Path, **kwargs) -> pl.DataFrame:
    """Unified CSV loader for tests."""
    return pl.read_csv(path, **kwargs)


def assert_csv_equal_or_update(
    actual_path: Path, expected_path: Path, update=False, **read_csv_kwargs
):
    """
    Compare a generated CSV file to an expected one, or update the expected file if --update is used.
    Uses Polars for CSV loading and table comparison.
    """
    actual_df = read_csv_polars(actual_path, **read_csv_kwargs)

    if update:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        actual_df.write_csv(expected_path)
        print(f"✅ Updated expected file: {expected_path}")
        return

    expected_df = read_csv_polars(expected_path, **read_csv_kwargs)

    # Polars has built-in strict frame comparison
    try:
        actual_df.equals(expected_df, null_equal=True)
    except AssertionError:
        raise AssertionError(f"CSV mismatch:\nActual:   {actual_path}\nExpected: {expected_path}")


def run_integration_case(args: Namespace, expected_dir: Path, gases: list[str], tmp_path, request):
    """Shared integration runner for all 3 cases."""
    df_all, df_avg = process_file(
        input_file=args.input_file,
        cycles=args.cycles,
        samples=args.samples,
        sample_time=args.sample_time,
        blank=args.blank,
        flow=args.flow,
        chamber_volume=args.chamber_volume,
        soil_surface_area=args.soil_surface_area,
        buffer=args.buffer,
        outdir=args.out,
        header=args.header,
    )

    update = request.config.getoption("--update")

    for gas in gases:
        for suffix in ["_all.csv", "_avg.csv"]:
            actual = tmp_path / f"{gas}{suffix}"
            expected = expected_dir / f"{gas}{suffix}"
            assert_csv_equal_or_update(actual, expected, update=update)


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

    run_integration_case(
        args,
        expected_dir=Path("tests/expected/case1"),
        gases=[
            "ammonia",
            "carbon_dioxide",
            "methane",
            "nitrogen_dioxide",
            "nitrous_oxide",
            "ozone",
        ],
        tmp_path=tmp_path,
        request=request,
    )


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

    run_integration_case(
        args,
        expected_dir=Path("tests/expected/case2"),
        gases=["no", "noy-no", "noy"],
        tmp_path=tmp_path,
        request=request,
    )


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

    run_integration_case(
        args,
        expected_dir=Path("tests/expected/case3"),
        gases=["co2", "hono", "no2"],
        tmp_path=tmp_path,
        request=request,
    )
