import polars as pl


class DataStandardizer:
    COL_DATETIME = "datetime"
    GAS_IDENTIFIERS = {
        "Ammonia / ppm (cal)": "NH3",
        "NO2 (ppb)": "NO2",
        "Nitrogen Dioxide / ppm (cal)": "NO2",
        "Nitrous Oxide / ppm (cal)": "N2O",
        "Ozone / ppm (cal)": "O3",
        "HONO (ppb)": "HONO",
        "NO Conc": "NO",
        "NOY Conc": "NOY",
        "NOY-NO Conc": "NOY-NO",
        "Carbon Monoxide / ppm (cal)": "CO",
        "CO2 (ppm)": "CO2",
        "Carbon Dioxide / ppm (cal)": "CO2",
        "Methane / ppm (cal)": "CH4",
    }

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf.pipe(self._standardize_datetime)
            .pipe(self._select_input_columns)
            .rename(self._sanitize_column_names)
            .pipe(self._standardize_units)
            .rename(self._remove_units_from_names)
        )

    def _standardize_datetime(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.rename({lf.collect_schema().names()[0]: self.COL_DATETIME}).with_columns(
            pl.col(self.COL_DATETIME).str.strptime(pl.Datetime)
        )

    def _select_input_columns(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        names = [
            name
            for name in lf.collect_schema().names()
            if name.strip() in self.GAS_IDENTIFIERS.keys()
        ]

        return lf.select(*names, "datetime")

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
