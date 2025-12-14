import dateutil.parser
from pathlib import Path
from datetime import timedelta

from fluxpro.blank_handler import BlankHandler
from fluxpro.data_standardizer import DataStandardizer
import polars as pl

from .config import Config


def detect_separator(input_file: Path) -> str:
    if input_file.suffix == ".dat":
        return "\t"
    else:
        return ","


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
    cycle_duration = sample_time * samples

    return (
        lf.with_columns(
            elapsed=((pl.col("datetime") - pl.col("datetime").first()).dt.total_minutes())
        )
        .with_columns(
            cycle=((pl.col("elapsed") / cycle_duration).floor().cast(pl.Int64) + 1).clip(
                upper_bound=cycles
            ),
            sample=(
                ((pl.col("elapsed") % cycle_duration) / sample_time).floor().cast(pl.Int64) + 1
            ).clip(upper_bound=samples),
        )
        .drop("elapsed")
    )


def remove_transition_minutes(lf: pl.LazyFrame, buffer: int) -> pl.LazyFrame:
    start_times = pl.col("datetime").min().over(["cycle", "sample"])

    return lf.sort(["cycle", "sample", "datetime"]).filter(
        pl.col("datetime") >= (start_times + timedelta(minutes=buffer))
    )


def unpivot(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.unpivot(
        index=["cycle", "sample", "datetime"], variable_name="gas", value_name="concentration"
    )


def compute_statistics(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        flux_corrected_avg=pl.mean("flux_corrected").over(["sample", "cycle", "gas"]),
        flux_corrected_std=pl.std("flux_corrected").over(["sample", "cycle", "gas"]),
        flux_corrected_sem=(
            pl.std("flux_corrected").over(["sample", "cycle", "gas"])
            / pl.len().over(["sample", "cycle", "gas"]).sqrt()
        ),
    )


def compute_flux(
    lf: pl.LazyFrame, flow: float, chamber_volume: float, soil_surface_area: float
) -> pl.LazyFrame:

    nitrogen_mass = 14.006747
    carbon_mass = 12.0111
    oxygen_mass = 15.99943
    molar_masses = pl.LazyFrame(
        {
            "gas": ["NH3", "NO2", "N2O", "O3", "HONO", "NO", "NOY", "NOY-NO", "CO", "CO2", "CH4"],
            # g/mol
            "molar_mass": [
                nitrogen_mass,
                nitrogen_mass,
                nitrogen_mass,
                oxygen_mass,
                nitrogen_mass,
                nitrogen_mass,
                nitrogen_mass,
                nitrogen_mass,
                carbon_mass,
                carbon_mass,
                carbon_mass,
            ],
        }
    )
    nano = 1e9

    # see https://amt.copernicus.org/articles/15/2807/2022/ for equation
    # conc  * flow  / soil_surface_area * molar_mass * nano
    # mol/L * L/min / m^2               * g/mol      * n
    # concentration has units of mol/L from DataStandardizer
    # flux returned in units of ng/m^2/min
    return (
        lf.join(molar_masses, on="gas")
        .with_columns(
            flux=pl.col("concentration") * flow / soil_surface_area * pl.col("molar_mass") * nano
        )
        .drop("concentration", "molar_mass")
    )


def reformat_for_output(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.unpivot(
            index=["cycle", "sample", "datetime", "gas"],
            variable_name="metric",
            value_name="value",
        )
        .with_columns((pl.col("gas") + "_" + pl.col("metric")).alias("column_name"))
        .collect()
        .pivot(
            index=["cycle", "sample", "datetime"],
            on="column_name",
            values="value",
            aggregate_function="first",
        )
        .sort("cycle", "sample", "datetime")
    )


def process_file(input_file: Path, config: Config):

    separator = detect_separator(input_file)
    skip_rows = detect_header_row(input_file, separator)

    lf = pl.scan_csv(
        input_file,
        has_header=True,
        skip_rows=skip_rows,
        infer_schema_length=1000,
        separator=separator,
    )

    standardizer = DataStandardizer()

    blank_handler = BlankHandler.create_handler(config.blank)

    df = (
        lf.pipe(standardizer.run)
        .pipe(
            label_rows_by_time,
            config.samples.total_cycles,
            config.samples.minutes_per_sample,
            config.samples.samples_per_cycle,
        )
        .pipe(
            remove_transition_minutes,
            config.samples.discard_minutes,
        )
        .pipe(unpivot)
        .pipe(
            compute_flux,
            config.flux.flow_rate,
            config.flux.chamber_volume,
            config.flux.soil_surface_area,
        )
        .pipe(blank_handler.run)
        .pipe(compute_statistics)
        .pipe(reformat_for_output)
    )

    output_file = input_file.with_name(f"{input_file.stem}_out.csv")
    df.write_csv(output_file)
