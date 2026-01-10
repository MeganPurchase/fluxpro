from pathlib import Path

import polars as pl


def write_output(input_file: Path, df: pl.DataFrame) -> list[Path]:
    output_files: list[Path] = []
    for sample, df_sample in df.partition_by("sample", as_dict=True).items():
        output_file = input_file.with_name(f"{input_file.stem}_{sample[0]}_out.csv")
        df_sample.write_csv(output_file)
        output_files.append(output_file)

    return output_files
