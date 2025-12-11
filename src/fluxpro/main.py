from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
import sys
import tomllib
from pathlib import Path
from typing import Tuple

import polars as pl
import polars.selectors as cs

GREEN = "\033[32m"
RESET = "\033[0m"
ASCII_ART = """\
 ___ _
|  _| |_ _ _ _ ___ ___ ___
|  _| | | |_'_| . |  _| . |
|_| |_|___|_,_|  _|_| |___|
              |_|
"""


@dataclass
class Config:

    @dataclass
    class FileConfig:
        input_file: Path = field(
            metadata={"doc": "file containing the gas flux data"}, default=Path("input.csv")
        )
        output_directory: Path = field(
            metadata={"doc": "path to the output directory"}, default=Path("output")
        )

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
        blank_sample_index: int = field(
            metadata={"doc": "index of the blank (counting up from 1)", "example": 1}
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

    files: FileConfig
    samples: SampleConfig
    flux: FluxConfig

    @staticmethod
    def from_toml(file: Path) -> Config:
        with open(file, "rb") as f:
            data = tomllib.load(f)
            return Config(
                files=Config.FileConfig(**data["files"]),
                samples=Config.SampleConfig(**data["samples"]),
                flux=Config.FluxConfig(**data["flux"]),
            )

    @staticmethod
    def generate_example_toml() -> str:
        def section_to_toml(
            section_name: str,
            config: type[Config.FileConfig] | type[Config.SampleConfig] | type[Config.FluxConfig],
        ):
            lines = [f"[{section_name}]"]
            for f in fields(config):
                lines.append(f"# {f.metadata.get("doc")}")
                example = f.metadata.get("example")
                if not example:
                    example = f.default
                if isinstance(example, Path):
                    example = example.as_posix()
                lines.append(f"{f.name} = {repr(example)}")
            return "\n".join(lines)

        toml_sections = [
            section_to_toml("files", Config.FileConfig),
            section_to_toml("samples", Config.SampleConfig),
            section_to_toml("flux", Config.FluxConfig),
        ]

        return "\n\n".join(toml_sections) + "\n"


def label_rows_by_time(
    lf: pl.LazyFrame, cycles: int, sample_time: float, samples: int
) -> pl.LazyFrame:
    """
    Assign cycle and sample numbers using timestamp information.
    """
    cycle_duration = sample_time * samples
    time_col = lf.collect_schema().names()[0]

    return (
        lf.with_columns(pl.col(time_col).str.strptime(pl.Datetime))
        .with_columns(
            elapsed=((pl.col(time_col) - pl.col(time_col).first()).dt.total_seconds() / 60.0)
        )
        .with_columns(
            cycle=((pl.col("elapsed") / cycle_duration).floor().cast(pl.Int64) + 1).clip(
                upper_bound=cycles
            ),
            sample=(
                ((pl.col("elapsed") % cycle_duration) / sample_time).floor().cast(pl.Int64) + 1
            ).clip(upper_bound=samples),
        )
    )


def remove_transition_minutes(lf: pl.LazyFrame, sample_time: int, buffer: int) -> pl.LazyFrame:
    """
    Remove the first `buffer` minutes from each (cycle, sample) group.
    """
    keep_n = sample_time - buffer

    return lf.sort(["cycle", "sample"]).group_by(["cycle", "sample"]).tail(keep_n)


def filter_relevant_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter columns matching relevant gas names or metadata.
    """
    regex = r"(cycle|sample|ppm \(cal\)|Conc|NO2 \(ppb\)|HONO \(ppb\)|CO2 \(ppm\))"

    return lf.select(cs.matches(regex))


def unpivot(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.unpivot(index=["cycle", "sample"], variable_name="gas", value_name="value")


def subtract_blank(lf: pl.LazyFrame, blank_index: int) -> pl.LazyFrame:
    """
    Subtract blank sample means from all other samples.
    """
    blank_means = (
        lf.filter(pl.col("sample") == blank_index)
        .drop("sample")
        .group_by(["cycle", "gas"])
        .agg(mean_blank=pl.mean("value"))
    )

    return (
        lf.filter(pl.col("sample") != blank_index)
        .join(blank_means, on=["cycle", "gas"], how="left")
        .with_columns(value_reduced=pl.col("value") - pl.col("mean_blank"))
    )


def compute_statistics(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute mean/std/sem for each gas × cycle × sample.
    """
    return lf.group_by(["sample", "cycle", "gas"]).agg(
        mean=pl.mean("value_reduced"),
        std=pl.std("value_reduced"),
        sem=pl.std("value_reduced") / pl.len().sqrt(),
    )


def compute_flux(
    lf: pl.LazyFrame, flow: float, chamber_volume: float, soil_surface_area: float
) -> pl.LazyFrame:
    """
    Compute flux from reduced values.
    """
    factor = chamber_volume / soil_surface_area
    return lf.with_columns(flux=pl.col("mean") * flow * factor)


def sanitize_gas_name(name: str) -> str:
    """
    Convert gas name into a file-safe lowercase value.
    """
    if "ppm (cal)" in name:
        return name.split(" / ")[0].replace(" ", "_").lower()
    if "Conc" in name or "(ppb)" in name or "(ppm)" in name:
        return name.split()[0].lower()
    return name.replace(" ", "_").lower()


def write_output(df: pl.DataFrame, output_directory: Path, suffix: str) -> None:
    """
    Write each gas into its own CSV, based on sanitized gas names.
    """
    output_directory.mkdir(exist_ok=True)

    for gas_value, gdf in df.group_by("gas"):
        fname: str = sanitize_gas_name(gas_value[0])
        print(gas_value)
        print(fname)
        gdf = gdf.drop("gas")
        gdf.write_csv(output_directory / f"{fname}_{suffix}.csv")


def process_file(
    input_file: str,
    cycles: int,
    samples: int,
    sample_time: int,
    blank: int,
    flow: float,
    chamber_volume: float,
    soil_surface_area: float,
    buffer: int,
    outdir: Path,
    header: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Execute the entire data-processing pipeline in lazy mode.
    """

    separator = ","
    if input_file.endswith("dat"):
        separator = "\t"

    lf = pl.scan_csv(
        input_file, has_header=True, skip_rows=header, infer_schema_length=1000, separator=separator
    )
    time_col = lf.collect_schema().names()[0]

    lf_all = (
        lf.pipe(label_rows_by_time, cycles, sample_time, samples)
        .pipe(remove_transition_minutes, sample_time, buffer)
        .pipe(filter_relevant_columns)
        .pipe(unpivot)
        .pipe(subtract_blank, blank)
    )

    df_all = lf_all.collect()
    write_output(df_all, outdir, "all")

    lf_avg = lf_all.pipe(compute_statistics).pipe(
        compute_flux, flow, chamber_volume, soil_surface_area
    )

    df_avg = lf_avg.collect()
    write_output(df_avg, outdir, "avg")

    return df_all, df_avg


def validate_input(input_file: str, header: int) -> None:
    """
    Primitive validation to detect early empty lines.
    """
    with open(input_file) as f:
        for i, line in enumerate(f):
            print(repr(line))
            if i > 10:
                raise ValueError(f"empty line at line {i}")


def print_header(args: argparse.Namespace) -> None:
    print(f"{GREEN}{ASCII_ART}{RESET}")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Process data from Teledyne NOy analyser and FTIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_file", type=str, help=".csv file containing the gas flux data")
    parser.add_argument("cycles", type=int, help="total number of cycles")
    parser.add_argument(
        "samples", type=int, help="number of samples per cycle (including the blank)"
    )
    parser.add_argument("sample_time", type=int, help="number of minutes per sample")
    parser.add_argument("blank", type=int, help="index of the blank (counting up from 1)")
    parser.add_argument("flow", type=float, help="flow rate through the chamber (L/min)")
    parser.add_argument("chamber_volume", type=float, help="volume of the chamber headspace (m^3)")
    parser.add_argument("soil_surface_area", type=float, help="surface area of the soil (m^2)")
    parser.add_argument(
        "--buffer",
        "-b",
        type=int,
        default=2,
        help="number of minutes at the start of each sample that are removed from the analysis to allow the readings to settle",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default="output",
        help="name of output directory",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        help="line number of the header",
    )

    args = parser.parse_args()
    print_header(args)

    validate_input(args.input_file, args.header)

    process_file(
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


if __name__ == "__main__":
    sys.exit(main())
