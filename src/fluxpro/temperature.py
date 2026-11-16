from pathlib import Path

from fluxpro.data_standardizer import DataStandardizer
import polars as pl


def read_temperature(input_file: Path) -> pl.LazyFrame:

    lf = (
        pl.scan_csv(
            input_file,
            has_header=True,
            skip_rows=1,
        )
        .select(pl.nth([1, 2]))
        .drop_nulls()
    )

    standardizer = DataStandardizer()
    temp_col_name = "temperature"

    name_mapping = {lf.collect_schema().names()[1]: temp_col_name}

    return (
        lf.pipe(standardizer.standardize_datetime)
        .rename(name_mapping)
        .pipe(_fahrenheit_to_kelvin, temp_col_name)
    )


def _fahrenheit_to_kelvin(lf: pl.LazyFrame, temp_col_name: str):
    return lf.with_columns((pl.col(temp_col_name) - 32) * 5 / 9 + 273.15)
