import argparse
import numpy as np
import pandas as pd
import sys


def assign_row_labels(df, cycles, sample_time, samples):
    minutes_per_cycle = sample_time * samples
    df["cycle"] = np.repeat(np.arange(1, cycles + 1), minutes_per_cycle)
    df["sample"] = np.tile(
        np.array([np.full(sample_time, i) for i in range(1, samples + 1)]).flatten(), cycles
    )
    return df


def remove_transition_minutes(df, sample_time, buffer):
    gdf = df.groupby(["sample", "cycle"])
    return gdf.tail(sample_time - buffer)


def compute_averages(df):
    def aggregate_column(gdf, column: str):
        result = gdf.agg(mean=(column, "mean"), std=(column, "std"), sem=(column, "sem"))
        result["gas"] = column.split(" / ")[0]
        return result

    gdf = df.groupby(["sample", "cycle"])
    datacols = [column for column in df if "ppm (cal)" in column]
    all_dfs = [aggregate_column(gdf, column) for column in datacols]
    return pd.concat(all_dfs)


def subtract_blank_mean(df, blank):
    gdf = df.groupby("sample")
    non_blanks = gdf.filter(lambda x: x.name != blank).reset_index()
    blank = gdf.filter(lambda x: x.name == blank).reset_index()
    blank.drop(columns=["sample"], inplace=True)

    df = non_blanks.merge(blank, how="left", on=["cycle", "gas"], suffixes=("", "_blank"))
    df["mean_reduced"] = df["mean"] - df["mean_blank"]
    return df


parser = argparse.ArgumentParser(
    description="Process data from Teledyne NOy analyser and FTIR",
    epilog="example: fluxpro inp.csv 24 10 6 10 --buffer 2",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("input_file", type=str, help=".csv file containing the gas flux data")
parser.add_argument("cycles", type=int, help="total number of cycles")
parser.add_argument("samples", type=int, help="number of samples per cycle (including the blank)")
parser.add_argument("sample_time", type=int, help="number of minutes per sample")
parser.add_argument("blank", type=int, help="index of the blank (counting up from 1)")
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
    type=str,
    default="out.csv",
    help="name of output file",
)


def main():
    args = parser.parse_args()

    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    print(f"{GREEN}Reading data from {BLUE}`{args.input_file}`{GREEN}.{RESET}")
    df = pd.read_csv(args.input_file, header=2)

    df = assign_row_labels(df, args.cycles, args.sample_time, args.samples)

    df = remove_transition_minutes(df, args.sample_time, args.buffer)
    print(f"{GREEN}Deleted {args.buffer} minutes from the beginning of each sample.{RESET}")

    print(f"{GREEN}Computing averages.{RESET}")
    df = compute_averages(df)

    print(f"{GREEN}Subtracting the blank.{RESET}")
    df = subtract_blank_mean(df, args.blank)

    df.to_csv(args.out, index=False)
    print(f"{GREEN}Results written to {BLUE}`{args.out}`{GREEN}.{RESET}")


if __name__ == "__main__":
    sys.exit(main())
