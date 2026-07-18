# Weather-field downscaling with U-Net and GANs

This project contains four Jupyter notebooks for statistical downscaling of four meteorological fields over Indonesia. Each model maps a 1-day, low-resolution forecast to a 6× higher-resolution field on the ERA5 grid.

## Notebooks

| Notebook | Model | Input/output | Upsampling strategy |
| --- | --- | --- | --- |
| [`unet.ipynb`](unet.ipynb) | SRUNet | `4 × 24 × 32` → `4 × 144 × 192` | Learned U-Net head with 2× and 3× bilinear stages |
| [`unet_preupsample.ipynb`](unet_preupsample.ipynb) | U-Net | `4 × 144 × 192` → `4 × 144 × 192` | Bilinear interpolation before the model |
| [`gan.ipynb`](gan.ipynb) | RRDB generator + residual discriminator | `4 × 24 × 32` → `4 × 144 × 192` | Generator performs the 6× upsampling |
| [`gan_preupsample.ipynb`](gan_preupsample.ipynb) | RRDB generator + residual discriminator | `4 × 144 × 192` → `4 × 144 × 192` | Bilinear interpolation before the GAN |

All notebooks use four channels:

| Dataset variable | Label | Units |
| --- | --- | --- |
| `10m_u_component_of_wind` | U10 | m/s |
| `10m_v_component_of_wind` | V10 | m/s |
| `2m_temperature` | T2m | K |
| `total_precipitation_24hr` | TP 24hr | mm |

## Data layout

The notebooks expect the following files, which are not included in this directory because they are large:

```text
data/
├── ifs_lowres_indonesia_2018-2022.zarr
└── era5_indonesia_2018-2022.zarr
```

The forecast dataset must contain `prediction_timedelta`; the notebooks select `lead_days = 1`. The truth and forecast grids are aligned over the Indonesia domain, producing a low-resolution grid of `24 × 32` and a high-resolution grid of `144 × 192`.

## Preprocessing

Each notebook performs the following steps:

1. Loads the IFS forecast and ERA5 truth with `xarray`.
2. Sorts latitude coordinates and aligns forecast valid time with ERA5 time.
3. Fills missing forecast values with the channel mean.
4. Applies `log1p` to non-negative precipitation values.
5. Splits samples by date:
   - train: 2018-01-01 through 2021-12-31
   - validation: 2022-01-01 through 2022-06-30
   - test: 2022-07-01 through 2022-12-31
6. Computes separate z-score statistics for inputs and targets using the training split only.

At evaluation time, predictions are denormalized and precipitation is restored with `expm1`.

## Requirements

Use a Python 3.10+ environment with Jupyter and the scientific Python stack used in the notebooks:

```bash
python -m pip install jupyterlab numpy pandas xarray zarr torch matplotlib cartopy xesmf kornia
```

`cartopy` and `xesmf` may require platform-specific geospatial/ESMF libraries. A CUDA-enabled PyTorch installation is recommended for training; the notebooks automatically fall back to CPU when CUDA is unavailable.

## Running a notebook

1. Put the two Zarr datasets under `data/`.
2. Start Jupyter from the project directory:

   ```bash
   jupyter lab
   ```

3. Open one notebook and run its cells from top to bottom.
4. Adjust the configuration cell if you need a different lead time, seed, batch size, dates, or target visualization date.

The first cell changes the working directory to `/home/jovyan/work`, which is the expected mounted path in the original notebook environment. In another environment, change or remove that `os.chdir(...)` line and ensure the relative `data/` path is correct.

The default training configuration is 100 epochs with AdamW, cosine learning-rate scheduling, gradient clipping, and early stopping after seven validation epochs without improvement. The default batch size varies by notebook.

## Outputs

Each run creates a timestamped directory such as `runs/YYYYMMDD_HHMMSS_lead1d/` containing:

- `config.json` and `runtime.log`
- `best_model.pt`
- training-curve and spatial-evaluation PNG files
- chart configuration JSON files
- `metrics.csv`

`metrics.csv` reports RMSE, MAE, bias, correlation, bilinear-interpolation baseline RMSE, and skill for each variable. Positive `Skill` indicates lower RMSE than the bilinear baseline.

## Project status

This repository currently contains executable notebooks rather than a packaged Python module or standalone command-line training script. No dataset, trained checkpoint, or pinned dependency file is included.
