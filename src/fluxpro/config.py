from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib
from typing import Literal


@dataclass
class Config:

    @dataclass
    class SampleConfig:
        total_cycles: int = field(metadata={"doc": "total number of cycles", "example": 22})
        samples_per_cycle: int = field(
            metadata={"doc": "number of samples per cycle (including the blank)", "example": 6}
        )
        minutes_per_sample: int = field(
            metadata={"doc": "number of minutes per sample", "example": 10}
        )
        discard_minutes: int = field(
            metadata={
                "doc": "number of minutes at the start of each sample that are removed from the analysis to allow the readings to settle",
                "example": 2,
            }
        )

    @dataclass
    class FluxConfig:
        flow_rate: float = field(
            metadata={"doc": "flow rate through the chamber (L/min)", "example": 0.1}
        )
        chamber_volume: float = field(
            metadata={"doc": "volume of the chamber headspace (m^3)", "example": 0.01}
        )
        soil_surface_area: float = field(
            metadata={"doc": "surface area of the soil (m^2)", "example": 0.05}
        )

    @dataclass
    class BlankConfig:
        mode: Literal["sample"] | Literal["cycle"] = field(
            metadata={
                "doc": "specifies whether to use a sample or a cycle as the blank reading. Options: 'sample' or 'cycle'",
                "example": "sample",
            }
        )
        index: int = field(
            metadata={"doc": "index of the blank (counting up from 1)", "example": 1}
        )

    samples: SampleConfig
    flux: FluxConfig
    blank: BlankConfig

    @staticmethod
    def from_toml(file: Path) -> Config:
        with open(file, "rb") as f:
            data = tomllib.load(f)
            return Config(
                samples=Config.SampleConfig(**data["samples"]),
                flux=Config.FluxConfig(**data["flux"]),
                blank=Config.BlankConfig(**data["blank"]),
            )

    @staticmethod
    def generate_example_toml() -> str:
        def section_to_toml(
            section_name: str,
            config: type[Config.SampleConfig] | type[Config.FluxConfig] | type[Config.BlankConfig],
        ):
            lines = [f"[{section_name}]"]
            for f in fields(config):
                lines.append(f"# {f.metadata.get("doc")}")
                example = f.metadata.get("example")
                lines.append(f"{f.name} = {repr(example)}")
            return "\n".join(lines)

        toml_sections = [
            section_to_toml("samples", Config.SampleConfig),
            section_to_toml("flux", Config.FluxConfig),
            section_to_toml("blank", Config.BlankConfig),
        ]

        return "\n\n".join(toml_sections) + "\n"
