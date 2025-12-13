from typing import Tuple
import dateutil.parser
from pathlib import Path

import polars as pl
import polars.selectors as cs

from .config import Config


def detect_header_row(input_file: Path, separator: str) -> int:
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            splits = line.split(separator)
            if len(splits) > 0:
                try:
                    dateutil.parser.parse(splits[0])
                    return i - 1
                except ValueError:
                    continue
        raise ValueError("failed to detect header")


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
    for gas_value, gdf in df.group_by("gas"):
        fname: str = sanitize_gas_name(gas_value[0])
        gdf = gdf.drop("gas")
        gdf.write_csv(output_directory / f"{fname}_{suffix}.csv")


def process_file(input_file: Path, config: Config) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Execute the entire data-processing pipeline in lazy mode.
    """

    separator = ","
    if input_file.as_posix().endswith("dat"):
        separator = "\t"

    skip_rows = detect_header_row(input_file, separator)

    lf = pl.scan_csv(
        input_file,
        has_header=True,
        skip_rows=skip_rows,
        infer_schema_length=1000,
        separator=separator,
    )

    lf_all = (
        lf.pipe(
            label_rows_by_time,
            config.samples.total_cycles,
            config.samples.minutes_per_sample,
            config.samples.samples_per_cycle,
        )
        .pipe(
            remove_transition_minutes,
            config.samples.minutes_per_sample,
            config.samples.discard_minutes,
        )
        .pipe(filter_relevant_columns)
        .pipe(unpivot)
        .pipe(subtract_blank, config.samples.blank_sample_index)
    )

    df_all = lf_all.collect()
    write_output(df_all, input_file.parent, "all")

    lf_avg = lf_all.pipe(compute_statistics).pipe(
        compute_flux,
        config.flux.flow_rate,
        config.flux.chamber_volume,
        config.flux.soil_surface_area,
    )

    df_avg = lf_avg.collect()
    write_output(df_avg, input_file.parent, "avg")

    return df_all, df_avg
