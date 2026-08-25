from pathlib import Path

import polars as pl
from polars import testing

from fluxpro.temperature import read_temperature


def test_read_temperature_must_build_lazyframe(data_dir: Path):
    lf = read_temperature(data_dir / "temperature.csv")
    stats = lf.describe()

    expected_stats = pl.DataFrame(
        [
            pl.Series(
                "statistic",
                ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                dtype=pl.String,
            ),
            pl.Series(
                "datetime",
                [
                    "8634",
                    "0",
                    "2026-07-26 14:25:07.999999",
                    None,
                    "2026-07-23 14:28:38",
                    "2026-07-25 02:26:38",
                    "2026-07-26 14:25:38",
                    "2026-07-28 02:23:38",
                    "2026-07-29 14:21:38",
                ],
                dtype=pl.String,
            ),
            pl.Series(
                "temperature",
                [
                    8634.0,
                    0.0,
                    299.35984447790383,
                    5.348364066675414,
                    292.25888888888886,
                    294.9188888888889,
                    297.7511111111111,
                    303.41277777777776,
                    313.06611111111107,
                ],
                dtype=pl.Float64,
            ),
        ]
    )

    testing.assert_frame_equal(stats, expected_stats)
