from __future__ import annotations

import polars as pl
from abc import ABC, abstractmethod
from typing import override, Literal


class BlankHandler(ABC):

    def __init__(self, index: int):
        self.index = index

    @staticmethod
    def create_handler(mode: Literal["sample"] | Literal["cycle"], index: int) -> BlankHandler:
        if mode == "sample":
            return SampleBlankHandler(index)
        elif mode == "cycle":
            return CycleBlankHandler(index)

    def run(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.pipe(self._join_blank).pipe(self._subtract_blank)

    @abstractmethod
    def _join_blank(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError  # pragma: no cover

    def _subtract_blank(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(flux_corrected=pl.col("flux") - pl.col("flux_blank_avg"))


class SampleBlankHandler(BlankHandler):

    @override
    def _join_blank(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        blank_avg = (
            lf.filter(pl.col("sample") == self.index)
            .drop("sample")
            .group_by(["cycle", "gas"])
            .agg(flux_blank_avg=pl.mean("flux"))
        )

        return lf.filter(pl.col("sample") != self.index).join(
            blank_avg, on=["cycle", "gas"], how="left"
        )


class CycleBlankHandler(BlankHandler):

    @override
    def _join_blank(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        blank_avg = (
            lf.filter(pl.col("cycle") == self.index)
            .group_by("gas")
            .agg(flux_blank_avg=pl.mean("flux"))
        )

        return lf.filter(pl.col("cycle") != self.index).join(blank_avg, on="gas", how="left")
