import pytest
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from polars import testing

from fluxpro.config import Config
from fluxpro.process import process_file


def assert_csv_equal_or_update(actual_path: Path, expected_path: Path, update=False):
    actual_df = pl.read_csv(actual_path)

    if update:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        actual_df.write_csv(expected_path)
        print(f"Updated expected file: {expected_path}")
        return

    expected_df = pl.read_csv(expected_path)
    testing.assert_frame_equal(actual_df, expected_df)


def run_integration_case(
    input_file: Path, config: Config, expected_dir: Path, tmp_path: Path, request
):
    """Shared integration runner for all 3 cases."""

    file = shutil.copy(input_file, tmp_path)

    process_file(Path(file), config)

    update = request.config.getoption("--update")

    actual = tmp_path / f"{input_file.stem}_out.csv"
    expected = expected_dir / f"{input_file.stem}_out.csv"
    assert_csv_equal_or_update(actual, expected, update=update)


def test_pipeline_integration_1(data_dir: Path, tmp_path: Path, request):

    config = Config(
        samples=Config.SampleConfig(
            total_cycles=22,
            samples_per_cycle=6,
            minutes_per_sample=10,
            discard_minutes=2,
        ),
        flux=Config.FluxConfig(
            flow_rate=1,
            chamber_volume=1,
            soil_surface_area=1,
        ),
        blank=Config.BlankConfig(
            mode="sample",
            index=1,
        ),
    )

    run_integration_case(
        data_dir / "FTIR_0304.csv",
        config,
        expected_dir=data_dir / "expected/case1",
        tmp_path=tmp_path,
        request=request,
    )


def test_pipeline_integration_2(data_dir: Path, tmp_path: Path, request):

    config = Config(
        samples=Config.SampleConfig(
            total_cycles=21,
            samples_per_cycle=6,
            minutes_per_sample=10,
            discard_minutes=2,
        ),
        flux=Config.FluxConfig(
            flow_rate=1,
            chamber_volume=1,
            soil_surface_area=1,
        ),
        blank=Config.BlankConfig(
            mode="sample",
            index=1,
        ),
    )

    run_integration_case(
        data_dir / "NOy_0404_CRED.csv",
        config,
        expected_dir=data_dir / "expected/case2",
        tmp_path=tmp_path,
        request=request,
    )


def test_pipeline_integration_3(data_dir: Path, tmp_path: Path, request):

    config = Config(
        samples=Config.SampleConfig(
            total_cycles=24,
            samples_per_cycle=2,
            minutes_per_sample=30,
            discard_minutes=5,
        ),
        flux=Config.FluxConfig(
            flow_rate=1,
            chamber_volume=1,
            soil_surface_area=1,
        ),
        blank=Config.BlankConfig(
            mode="cycle",
            index=1,
        ),
    )

    run_integration_case(
        data_dir / "2025_11_08_NO2_HONO_Channel1_Data.dat",
        config,
        expected_dir=data_dir / "expected/case3",
        tmp_path=tmp_path,
        request=request,
    )
