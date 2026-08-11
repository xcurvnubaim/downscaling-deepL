# Weather-field downscaling with deep learning

This project provides a config-driven, staged deep-learning pipeline for
statistical downscaling of daily meteorological fields over Southeast Asia.
The current pipeline maps UKESM1 `tasmax`, `hurs`, and `pr` fields from a
`36 Ã— 27` low-resolution grid to the native ERA5 `181 Ã— 201` grid using a
Residual U-Net.

The stages are independent: a failed training run can reuse completed
preprocessing artifacts, and training can resume from `last.pt`.

## Environment

Python 3.12 is the supported baseline. Create an isolated environment and
install the project with its test dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For optional xESMF and plotting dependencies:

```bash
python -m pip install -e ".[all]"
```

Install a CUDA-enabled PyTorch build separately using the command appropriate
for the target CUDA runtime. The project does not pin a machine-specific CUDA
wheel.

## Configuration

Copy an example configuration and set the local NetCDF paths:

```bash
cp configs/development.example.yaml configs/development.yaml
```

The model is selected by import path, for example:

```yaml
model:
  class_path: truss_downscaling.models.residual_unet.ResidualUNet
```

## Pipeline

Run stages independently:

```bash
truss-downscale preprocess --config configs/development.yaml
truss-downscale train --config configs/development.yaml
truss-downscale evaluate --config configs/development.yaml
```

Or run the complete development sequence:

```bash
truss-downscale run --config configs/development.yaml
```

Training resumes from `runs/<run-id>/training/last.pt`. Use `--restart` to
start from new weights, or `--force` to rerun a completed stage.

For future inference, use a production checkpoint:

```bash
truss-downscale infer --config configs/inference.example.yaml
```

## Data and preprocessing

The current pipeline uses three variables:

| Dataset variable | Description |
| --- | --- |
| `tasmax` | Daily maximum near-surface temperature |
| `hurs` | Near-surface relative humidity |
| `pr` | Daily precipitation |

Preprocessing is performed before training and records its metadata and
normalization statistics as run artifacts. Climate datasets and trained model
checkpoints are intentionally excluded from this repository because of their
size.

## Stage artifacts

Each run stores artifacts under `runs/<run-id>/`:

```text
runs/<run-id>/
â”œâ”€â”€ preprocessing/
â”‚   â”œâ”€â”€ X.npy
â”‚   â”œâ”€â”€ Y.npy
â”‚   â”œâ”€â”€ splits.npz
â”‚   â”œâ”€â”€ normalization.json
â”‚   â”œâ”€â”€ metadata.json
â”‚   â””â”€â”€ _SUCCESS
â”œâ”€â”€ training/
â”‚   â”œâ”€â”€ last.pt
â”‚   â”œâ”€â”€ best.pt
â”‚   â”œâ”€â”€ history.csv
â”‚   â””â”€â”€ _SUCCESS
â””â”€â”€ evaluation/
    â”œâ”€â”€ test_metrics.csv
    â””â”€â”€ _SUCCESS
```

## Notebook experiments

The repository history also contains four Jupyter notebooks for an earlier
IFS/ERA5 downscaling experiment. These notebooks use four channels and a 6Ã—
resolution change, and should be treated as a separate experimental workflow
rather than as the interface for the staged pipeline.

| Notebook | Model | Input/output |
| --- | --- | --- |
| [`unet.ipynb`](unet.ipynb) | SRUNet | `4 Ã— 24 Ã— 32` â†’ `4 Ã— 144 Ã— 192` |
| [`unet_preupsample.ipynb`](unet_preupsample.ipynb) | U-Net | `4 Ã— 144 Ã— 192` â†’ `4 Ã— 144 Ã— 192` |
| [`gan.ipynb`](gan.ipynb) | RRDB generator with residual discriminator | `4 Ã— 24 Ã— 32` â†’ `4 Ã— 144 Ã— 192` |
| [`gan_preupsample.ipynb`](gan_preupsample.ipynb) | RRDB generator with residual discriminator | `4 Ã— 144 Ã— 192` â†’ `4 Ã— 144 Ã— 192` |

The notebook variables are U10, V10, T2m, and 24-hour precipitation. Their
expected datasets are:

```text
data/
â”œâ”€â”€ ifs_lowres_indonesia_2018-2022.zarr
â””â”€â”€ era5_indonesia_2018-2022.zarr
```

The notebooks select a one-day forecast lead, split samples by date, compute
normalization statistics from the training split only, and restore
precipitation with `expm1` during evaluation. They require Jupyter and the
scientific Python stack used in the notebooks, including `xarray`, `zarr`,
PyTorch, `cartopy`, `xesmf`, and `kornia`.

## Project status

The staged CLI pipeline is the recommended entry point. The notebook workflow
is retained for reproducibility of the earlier IFS/ERA5 experiments. Neither
workflow includes the climate datasets or trained checkpoints in source
control.