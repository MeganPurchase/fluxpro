import polars as pl
import datetime


class DataStandardizer:
    COL_DATETIME = "datetime"
    GAS_IDENTIFIERS = {
        "Ammonia / ppm (cal)": "NH3",
        "NH3 / ppm (cal)": "NH3",
        "NO2 (ppb)": "NO2",
        "Nitrogen Dioxide / ppm (cal)": "NO2",
        "NO2 / ppm (cal)": "NO2",
        "Nitrous Oxide / ppm (cal)": "N2O",
        "N2O / ppm (cal)": "N2O",
        "Ozone / ppm (cal)": "O3",
        "O3 / ppm (cal)": "O3",
        "HONO (ppb)": "HONO",
        "NO Conc": "NO",
        "NOY Conc": "NOY",
        "NOY-NO Conc": "NOY-NO",
        "Carbon Monoxide / ppm (cal)": "CO",
        "CO / ppm (cal)": "CO",
        "CO2 (ppm)": "CO2",
        "Carbon Dioxide / ppm (cal)": "CO2",
        "CO2 / ppm (cal)": "CO2",
        "Methane / ppm (cal)": "CH4",
        "CH4 / ppm (cal)": "CH4",
    }

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.pipe(self._standardize_datetime)
            .pipe(self._select_input_columns)
            .pipe(self._strip_whitespace)
            .rename(self._sanitize_column_names)
            .pipe(self._standardize_units)
            .rename(self._remove_units_from_names)
        )

    def _standardize_datetime(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        possible_formats = [
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
        ]

        mapping = {lf.collect_schema().names()[0]: self.COL_DATETIME}
        lf = lf.rename(mapping)

        for format in possible_formats:
            try:
                lf_with_time = lf.with_columns(pl.col(self.COL_DATETIME).str.to_datetime(format))
                lf_diff = lf_with_time.select(
                    diff=pl.col(self.COL_DATETIME).diff(null_behavior="drop")
                )
                range = lf_diff.select(pl.col("diff").max() - pl.col("diff").min()).collect().item()
                if range < datetime.timedelta(hours=1):
                    return lf_with_time
            except pl.exceptions.InvalidOperationError:
                continue
        raise ValueError("unexpected datetime format")

    def _select_input_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        names = [
            name
            for name in lf.collect_schema().names()
            if name.strip() in self.GAS_IDENTIFIERS.keys()
        ]

        return lf.select(*names, "datetime")

    def _strip_whitespace(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(pl.col(pl.String).str.strip_chars().cast(pl.Float64, strict=False))

    def _sanitize_column_names(self, name: str) -> str:
        name = name.strip()
        if name in self.GAS_IDENTIFIERS:
            gas = self.GAS_IDENTIFIERS[name]
            unit = self._detect_unit(name)
            return f"{gas}_{unit}"
        else:
            return name

    def _detect_unit(self, name: str) -> str:
        if "ppm" in name:
            return "ppm"
        if "ppb" in name:
            return "ppb"
        if "Conc" in name:
            return "ppm"
        raise ValueError(f"failed to detect unit for column '{name}'")

    def _standardize_units(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        R = 0.082057366080960  # L*atm/K/mol
        T = 298.15  # K
        P = 1  # atm
        molar_volume = R * T / P  # L/mol

        expressions = []
        for column in lf.collect_schema().names():
            if column == self.COL_DATETIME:
                continue

            scale_factor = 0
            if column.endswith("ppm"):
                scale_factor = 1e-6
            elif column.endswith("ppb"):
                scale_factor = 1e-9

            expressions.append(pl.col(column) * scale_factor / molar_volume)

        # All gases now have units of mol/L
        return lf.with_columns(expressions)

    def _remove_units_from_names(self, name: str) -> str:
        return name.split("_")[0]
