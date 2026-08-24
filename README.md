# Weather-field downscaling with deep learning

This project provides a config-driven, staged deep-learning pipeline for
statistical downscaling of daily meteorological fields over Southeast Asia.
The current pipeline maps UKESM1 `tasmax`, `hurs`, and `pr` fields from a
`36 × 27` low-resolution grid to the native ERA5 `181 × 201` grid using a
Residual U-Net.

The stages are independent: a failed training run can reuse completed
preprocessing artifacts, and training can resume from `last.pt`.

## Environment

Python 3.12 is the supported baseline. For preprocessing and inference, use
the conda-forge environment because xESMF depends on ESMPy, which is not
available from PyPI:

On Linux, install Mamba through Miniforge:

```bash
cd /tmp
curl -L -O \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash
source ~/miniforge3/etc/profile.d/conda.sh
mamba --version
```

For a non-interactive shell or script, use the batch installer instead. This
automatically accepts the license and avoids stopping at the license prompt:

```bash
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
"$HOME/miniforge3/bin/mamba" --version
```

Then create the project environment:

```bash
cd /home/xcurv/work/downscaling-deepL
mamba env create -f environment.yml
mamba activate truss-xesmf
python -m pip install -e ".[test,tracking]"
```

Verify the xESMF installation:

```bash
python -c "import xesmf, ESMF; print('xESMF OK:', xesmf.__version__)"
```

Conda can be used instead of mamba:

```bash
conda env create -f environment.yml
conda activate truss-xesmf
python -m pip install -e ".[test,tracking]"
```

For tests or code paths that do not regrid data, a regular Python virtual
environment with `python -m pip install -e ".[test]"` is sufficient. The
preprocessing and inference commands require the conda-forge xESMF
environment. Diagnostic maps are included in `environment.yml`; W&B tracking
is optional and enabled with `tracking.enabled: true`.

Install a CUDA-enabled PyTorch build separately using the command appropriate
for the target CUDA runtime. The project does not pin a machine-specific CUDA
wheel.

The package pins PyTorch's public version to `2.11.0`. For a CUDA 12.8 driver,
install the matching local build before the editable project install:

```bash
python -m pip install "torch==2.11.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[test,tracking]"
```

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

The development configuration follows the UKESM1/ERA5 log1p notebook: it uses
300 epochs, AdamW with cosine annealing, the global SSIM-plus-L1 loss, bilinear
xESMF regridding, and `log1p` precipitation transforms on both inputs and
targets. Enable `tracking.enabled` to send live batch loss, epoch metrics,
timing, throughput, and execution metadata to Weights & Biases. CSV artifacts
and a timestamped `training.log` are always written locally.

Configure tracking and completed-run checkpoint publishing with:

```yaml
tracking:
  enabled: true
  project: truss-downscaling
  run_name: dev-log1p
  log_every_n_steps: 10
  artifacts:
    enabled: true
    type: model
    aliases: [latest, best]
```

Install the tracking extra and authenticate once with `wandb login`. During
training, W&B receives batch and epoch metrics. After successful completion,
`best.pt`, `history.csv`, and the training manifest are published as a W&B
model artifact named `<run-id>-model`. Set `tracking.artifacts.name` to use a
custom artifact name. Artifact upload failures do not remove or invalidate the
local checkpoint.

For future inference, use a production checkpoint:

```bash
truss-downscale infer --config configs/inference.example.yaml
```

The input NetCDF and a W&B checkpoint artifact can be supplied without editing
the configuration:

```bash
truss-downscale infer --config configs/inference.example.yaml \
  --input-file data/UKESM1_future.nc \
  --artifact ENTITY/truss-downscaling/production-r1-model:best
```

Artifacts are downloaded under `artifacts/` by default. Use `--artifact-dir`
to select another location. W&B authentication and the `tracking` extra are
required for artifact downloads.

Use `--checkpoint` instead when `best.pt` is already available locally:

```bash
truss-downscale infer --config configs/inference.example.yaml \
  --input-file data/UKESM1_future.nc \
  --checkpoint artifact/dev-r1-baseline-model/best.pt \
  --force
```

The output filename comes from `scenario.climate_scenario` and
`scenario.member` in the configuration, not from the input filename. Every
invocation creates a UTC timestamped version under
`runs/<run-id>/inference/<timestamp>/` and prints the resulting NetCDF path.
Each version also records the exact input, target grid, and checkpoint in its
manifest.

Long time series are regridded, predicted, and written incrementally. Control
CPU RAM independently from the GPU batch size with:

```yaml
inference:
  device: cuda
  batch_size: 1
  time_chunk_size: 32
```

Reduce `time_chunk_size` if the process is terminated by the operating
system's out-of-memory killer.

The inference CLI displays progress, throughput, ETA, process CPU utilization,
process RAM, NVML GPU utilization, and device-wide used/total GPU memory by
default. Use `--no-progress` for logs, scripts, or other non-interactive
execution where progress updates are not wanted. GPU metrics show `n/a` when
NVML cannot query the selected device.

## Data and preprocessing

The current pipeline uses three variables:

| Dataset variable | Description |
| --- | --- |
| `tasmax` | Daily maximum near-surface temperature |
| `hurs` | Near-surface relative humidity |
| `pr` | Daily precipitation |

Preprocessing is performed before training and records its metadata and
normalization statistics as run artifacts. Statistics and edge fill values are
computed from the training period only. Climate datasets and trained model
checkpoints are intentionally excluded from this repository because of their
size.

## Stage artifacts

Each run stores artifacts under `runs/<run-id>/`:

```text
runs/<run-id>/
├── preprocessing/
│   ├── X.npy
│   ├── Y.npy
│   ├── splits.npz
│   ├── normalization.json
│   ├── metadata.json
│   └── _SUCCESS
├── training/
│   ├── last.pt
│   ├── best.pt
│   ├── history.csv
│   ├── training.log
│   ├── manifest.json
│   └── _SUCCESS
└── evaluation/
    ├── test_metrics.csv
    ├── bias_metrics.csv
    ├── figures/
    │   └── tasmax_map_input_pred_target_bias.png
    └── _SUCCESS
```

`test_metrics.csv` contains standardized RMSE, MAE, SSIM, and physical-unit
precipitation-aware RMSE per channel. `bias_metrics.csv` compares the initial
GCM bias with the post-prediction bias in physical units. Existing artifacts
created before the dual-sided precipitation transform are not compatible with
the current checkpoint format and must be regenerated with `--force` and
`--restart`.

## Notebook experiments

The repository history also contains four Jupyter notebooks for an earlier
IFS/ERA5 downscaling experiment. These notebooks use four channels and a 6×
resolution change, and should be treated as a separate experimental workflow
rather than as the interface for the staged pipeline.

| Notebook | Model | Input/output |
| --- | --- | --- |
| [`unet.ipynb`](unet.ipynb) | SRUNet | `4 × 24 × 32` → `4 × 144 × 192` |
| [`unet_preupsample.ipynb`](unet_preupsample.ipynb) | U-Net | `4 × 144 × 192` → `4 × 144 × 192` |
| [`gan.ipynb`](gan.ipynb) | RRDB generator with residual discriminator | `4 × 24 × 32` → `4 × 144 × 192` |
| [`gan_preupsample.ipynb`](gan_preupsample.ipynb) | RRDB generator with residual discriminator | `4 × 144 × 192` → `4 × 144 × 192` |

The notebook variables are U10, V10, T2m, and 24-hour precipitation. Their
expected datasets are:

```text
data/
├── ifs_lowres_indonesia_2018-2022.zarr
└── era5_indonesia_2018-2022.zarr
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
