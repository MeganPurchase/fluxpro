from pathlib import Path

from fluxpro.config import Config


def test_from_toml_must_read_config(data_dir: Path):
    config = Config.from_toml(data_dir / "config.toml")
    assert config.samples.total_cycles == 22
    assert config.samples.samples_per_cycle == 6
    assert config.samples.minutes_per_sample == 10
    assert config.samples.discard_minutes == 2
    assert config.flux.flow_rate == 0.1
    assert config.flux.chamber_volume == 0.01
    assert config.flux.soil_surface_area == 0.05
    assert config.blank.mode == "sample"
    assert config.blank.index == 1


def test_generate_example_toml_must_write_file(data_dir: Path, request):
    config = Config(
        samples=Config.SampleConfig(
            total_cycles=22,
            samples_per_cycle=6,
            minutes_per_sample=10,
            discard_minutes=2,
        ),
        flux=Config.FluxConfig(
            flow_rate=0.1,
            chamber_volume=0.1,
            soil_surface_area=0.05,
        ),
        blank=Config.BlankConfig(
            mode="sample",
            index=1,
        ),
    )

    toml = config.generate_example_toml()

    if request.config.getoption("--update"):
        with open(data_dir / "config.toml", "w") as f:
            f.write(toml)

    with open(data_dir / "config.toml") as f:
        assert toml == f.read()
