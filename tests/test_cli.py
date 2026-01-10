import os
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from fluxpro.cli import cli
from fluxpro.config import Config


def test_generate_must_write_valid_file(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["generate"])
    assert result.exit_code == 0
    assert result.output == "Example config file written to config.toml\n"

    with open(tmp_path / "config.toml") as f:
        assert f.read() == Config.generate_example_toml()


def test_run_must_fail_without_arguments():
    runner = CliRunner()
    result = runner.invoke(cli, ["run"])
    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_run_must_succeed(monkeypatch, tmp_path: Path, data_dir: Path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            (data_dir / "FTIR_0304.csv").as_posix(),
            "--config",
            (data_dir / "config.toml").as_posix(),
        ],
    )
    assert result.exit_code == 0
    output_files = [data_dir / f"FTIR_0304_{i}_out.csv" for i in range(2, 7)]

    for output_file in output_files:
        assert output_file.exists()
        assert output_file.as_posix() in result.output
        os.remove(output_file)


def test_plot_must_succeed(data_dir: Path, monkeypatch):

    mock_chart = MagicMock()
    monkeypatch.setattr("fluxpro.cli.plotting.plot_df", lambda _: mock_chart)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "plot",
            (data_dir / "expected/case1/FTIR_0304_2_out.csv").as_posix(),
        ],
    )
    assert result.exit_code == 0
    mock_chart.show.assert_called_once()
