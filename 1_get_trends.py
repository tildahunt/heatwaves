import xarray as xr
import pandas as pd
import numpy as np
import dask
from scipy.stats import theilslopes
import pymannkendall as mk
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

BASE_FOLDER = "/gws/nopw/j04/aopp/tildah/HW_metrics/"
MODELS = [
    "CanESM5", "ACCESS-ESM1-5", "CMCC-CM2-SR5", 
    "HadGEM3-GC31-LL", "MIROC6", "MPI-ESM1-2-LR",
    "NorESM2-LM"
]

def theil_sen_slope(y, time):
    """Compute Theil-Sen slope."""
    mask = ~np.isnan(y)  # Mask NaN values
    if np.sum(mask) < 5:  # Require at least 5 valid points
        return np.nan

    slope, _, _, _ = theilslopes(y[mask], time[mask])
    return slope

def mann_kendall_significance(y):
    """Compute Mann-Kendall test significance (1 if p < 0.05, else 0)."""
    mask = ~np.isnan(y)
    if np.sum(mask) < 5:
        return np.nan
    
    mk_result = mk.original_test(y[mask])
    return 1 if mk_result.p < 0.05 else 0

def compute_trends_vectorized(ds):
    time = ds.coords['year'].values  # Extract time as NumPy array
    trends = {}

    for var in ds.data_vars:
        data = ds[var].load()  # Ensure data is loaded into memory if using Dask

        for period, (start, end) in zip(["early", "late", "whole"], [(1940, 1979), (1980, 2020), (1940,2020)]):
            time_mask = (time >= start) & (time <= end)
            time_subset = time[time_mask]
            data_subset = data.sel(year=time_mask)

            # Compute Theil-Sen slope
            slopes = xr.apply_ufunc(
                theil_sen_slope,
                data_subset,
                xr.DataArray(time_subset, dims=["year"], coords={"year": time_subset}),
                input_core_dims=[["year"], ["year"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64]
            )

            # Compute Mann-Kendall significance
            significance = xr.apply_ufunc(
                mann_kendall_significance,
                data_subset,
                input_core_dims=[["year"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64]
            )

            # Store trend and significance mask
            trends[f'{var}_slope_{period}_period'] = slopes * 10 
            trends[f'{var}_significance_{period}_period'] = significance

    return xr.Dataset(trends, coords={"lat": ds.lat, "lon": ds.lon})

# Compute trends
def load_models(exp):
    for model in MODELS:
        print(model, exp)
        folder = os.path.join(BASE_FOLDER, model, exp)
        try:
            files = [f for f in os.listdir(folder) if f.endswith(".nc")]
            print(len(files))
        except FileNotFoundError:
            print(f"Missing: {folder}")
            continue
    
        for file in files:
            member = file.split('_')[2]
            output_file = f"data_all_members/{model}_{exp}__{member}_trends.nc"
            if os.path.exists(output_file):
                print(f"Output file already exists for {model}, {exp}, skipping computation.")
                continue 
            ds = xr.open_dataset(os.path.join(folder, file))
            ds = ds.sel(year = slice(1940, 2020))
            trends_ds = compute_trends_vectorized(ds)
            trends_ds = trends_ds.squeeze().drop_vars([v for v in ['quantile', 'height', 'type'] if v in trends_ds])
            trends_ds.to_netcdf(output_file)

EXPERIMENTS = ["historical"]

for exp in EXPERIMENTS:
    model_trends = load_models(exp)
