import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

GREEN = "\033[32m"
RESET = "\033[0m"
ASCII_ART = """\
 ___ _
|  _| |_ _ _ _ ___ ___ ___
|  _| | | |_'_| . |  _| . |
|_| |_|___|_,_|  _|_| |___|
              |_|
"""


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


def filter_relevant_columns(df):
    regex = r"(cycle|sample|ppm \(cal\)|Conc)"
    return df.filter(regex=regex)


def reformat_data(df):
    return df.melt(id_vars=["cycle", "sample"], var_name="gas")


def subtract_blank(df, blank):
    gdf = df.groupby("sample")
    non_blanks = gdf.filter(lambda x: x.name != blank).reset_index()

    blank = gdf.filter(lambda x: x.name == blank).reset_index()
    blank.drop(columns=["sample"], inplace=True)
    blank = blank.groupby(["cycle", "gas"]).agg(mean_blank=("value", "mean")).reset_index()

    df = non_blanks.merge(blank, how="left", on=["cycle", "gas"], suffixes=("", "_blank"))
    df["value_reduced"] = df["value"] - df["mean_blank"]

    return df.drop(columns=["index"])


def compute_averages(df):
    return df.groupby(["sample", "cycle", "gas"]).agg(
        mean=("value_reduced", "mean"), std=("value_reduced", "std"), sem=("value_reduced", "sem")
    )


def write_output(df, output_directory, suffix):
    def write(group):
        fname = group.name
        if "ppm (cal)" in group.name:
            fname = group.name.split(" / ")[0].replace(" ", "_").lower()
        elif "Conc" in group.name:
            fname = group.name.strip().split(" ")[0].lower()

        group = group.reset_index()
        group.drop(columns="gas", inplace=True)
        group.to_csv(output_directory / f"{fname}_{suffix}.csv", index=False)

    output_directory.mkdir(exist_ok=True)
    df.groupby("gas").apply(write)


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


def print_header(args):
    print(f"{GREEN}{ASCII_ART}{RESET}")
    print("Cycles:", args.cycles)
    print("Samples:", args.samples)
    print("Sample time:", args.sample_time)
    print("Blank index:", args.blank)
    print("Buffer minutes:", args.buffer)
    print("Output directory:", args.out)
    print("Header index:", args.header)
    print()


def main():
    args = parser.parse_args()

    print_header(args)

    print(f"Reading data from {GREEN}`{args.input_file}`{RESET}.")
    df = pd.read_csv(args.input_file, header=args.header)

    df = assign_row_labels(df, args.cycles, args.sample_time, args.samples)

    df = remove_transition_minutes(df, args.sample_time, args.buffer)
    print(f"Deleted {args.buffer} minutes from the beginning of each sample.")

    df = filter_relevant_columns(df)
    df = reformat_data(df)

    print("Subtracting the blank.")
    df = subtract_blank(df, args.blank)
    write_output(df.set_index(["sample", "cycle", "gas"]), args.out, "all")

    print("Computing averages.")
    df = compute_averages(df)

    write_output(df, args.out, "avg")
    print(f"Results written to {GREEN}`{args.out}`{RESET}.")


if __name__ == "__main__":
    sys.exit(main())
