from pathlib import Path

from fluxpro.temperature import read_temperature


def test_read_temperature_must_build_lazyframe(data_dir: Path):
    lf = read_temperature(data_dir / "temperature.csv")
    print(lf.collect())
