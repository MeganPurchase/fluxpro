import argparse
import sys
from pathlib import Path

from .config import Config
from . import process


GREEN = "\033[32m"
RESET = "\033[0m"
ASCII_ART = """\
 ___ _
|  _| |_ _ _ _ ___ ___ ___
|  _| | | |_'_| . |  _| . |
|_| |_|___|_,_|  _|_| |___|
              |_|
"""


def print_header() -> None:
    print(f"{GREEN}{ASCII_ART}{RESET}")
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Process data from Teledyne NOy analyser and FTIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="generate an example config file")

    default_config_path = "config.toml"

    run = subparsers.add_parser("run")
    run.add_argument("input_file", type=Path, help="path to the input file")
    run.add_argument(
        "-c", "--config", type=Path, default=default_config_path, help="path to the config file"
    )

    args = parser.parse_args()

    print_header()

    if args.command == "generate":
        with open(default_config_path, "w") as f:
            f.write(Config.generate_example_toml())

        print(f"Example config file written to {default_config_path}", file=sys.stderr)

    elif args.command == "run":
        config = Config.from_toml(args.config)

        process.process_file(args.input_file, config)


if __name__ == "__main__":
    sys.exit(main())
