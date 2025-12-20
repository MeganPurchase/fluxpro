# fluxpro

This package provides the `fluxpro` program for processing the output of the Teledyne NOy analyser and FTIR.

## Installation

Install the program from GitHub with [uv](https://docs.astral.sh/uv/) (recommended)
```
uv tool install git+https://github.com/MeganPurchase/fluxpro
```
or with pip
```
pip install git+https://github.com/MeganPurchase/fluxpro
```
After installation, the `fluxpro` program will be available from the command line.

## Usage

After the installation, run `fluxpro generate` to create an example configuration
file. This file should be edited to match the parameters of your experiment.

When the configuration file is ready, use
```bash
fluxpro run path/to/input --config path/to/config`
```
This will run the analysis and write an output file for each sample.

Finally, you can preview your results with `fluxpro plot path/to/output-file`.
This will open a browser window showing the flux over time for each of the gases.
