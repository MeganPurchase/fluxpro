import polars as pl
import re
import altair as alt

alt.data_transformers.enable("vegafusion")


def plot_df_altair(df: pl.DataFrame):

    pattern = re.compile(r"^(.*)_flux$")
    gases = [pattern.match(c).group(1) for c in df.columns if pattern.match(c)]

    records = []
    for gas in gases:
        records.append(
            df.select(
                pl.col("cycle"),
                pl.lit(gas).alias("gas"),
                pl.col(f"{gas}_flux_corrected").alias("raw"),
                pl.col(f"{gas}_flux_corrected_avg").alias("avg"),
                pl.col(f"{gas}_flux_corrected_sem").alias("sem"),
            )
        )

    tidy = pl.concat(records)

    return tidy


def plot_df(df: pl.DataFrame) -> alt.FacetChart:
    tidy = plot_df_altair(df)

    base = alt.Chart(tidy)

    avg = base.mark_point(color="red").encode(
        alt.X("average(cycle):Q", title="Cycle"),
        alt.Y(
            "avg:Q",
            axis=alt.Axis(title="Flux /ng·m⁻²·min⁻¹"),
        ),
    )

    errorbars = (
        base.transform_aggregate(
            x="mean(cycle)",
            y="mean(avg)",
            yerr="mean(sem)",
            groupby=["gas", "avg"],
        )
        .mark_errorbar(ticks=True)
        .encode(
            alt.X("x:Q"),
            alt.Y("y:Q", axis=alt.Axis(title="Flux /ng·m⁻²·min⁻¹")),
            alt.YError("yerr:Q"),
        )
    )

    raw = base.mark_point(color="pink").encode(
        alt.X("cycle:Q"),
        alt.Y("raw:Q", axis=alt.Axis(title="Flux /ng·m⁻²·min⁻¹")),
    )

    chart = (
        (raw + errorbars + avg)
        .facet(facet=alt.Facet("gas:N", header=alt.Header(title=None)), columns=3)
        .resolve_scale(y="independent")
    )

    return chart
