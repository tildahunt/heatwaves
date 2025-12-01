exp = 'hist-aer'
model = 'CanESM5'

import os
import glob
import xarray as xr
import numpy as np
from pathlib import Path

def preprocess_zg500(ds):
    ds['lon'] = (ds['lon'] + 180) % 360 - 180
    ds = ds.sortby('lon')
    ds = ds.sel(lon=slice(-90, 90), lat=slice(20, 90))
    ds = ds.isel(time=ds['time'].dt.month.isin([6, 7, 8]))
    ds = ds.sel(time=slice('1940-06-01', '2014-08-31'))
    return ds

folder = f"/gws/nopw/j04/leader_epesc/incoming/day/zg500_day/{exp}/{model}/"
zg_pattern = os.path.join(folder, f"zg_500_day_{exp}_{model}_*_interp.nc")
codes = sorted({
    os.path.basename(fp).split('_')[5]
    for fp in glob.glob(zg_pattern)
})

zg500arr_list = []
zg500anom_list = []

for code in codes:
    print(code)
    zg_input = f"//gws/nopw/j04/leader_epesc/incoming/day/zg500_day/{exp}/{model}/zg_500_day_{exp}_{model}_{code}_interp.nc"
    zg500raw = xr.open_mfdataset(zg_input, preprocess=preprocess_zg500)['zg500']
    climatology = zg500raw.sel(time=slice("1981-06-01", "2010-08-31")).groupby("time.dayofyear").mean(dim="time")
    zg500anom = zg500raw.groupby("time.dayofyear") - climatology

    hw_input = f"~/Internship/CIRCULATION/heatwave_in_*/{model}_{exp}/{model}_{exp}_{code}.nc"
    hw_days = xr.open_dataset(hw_input)['heatwave_in_europe']
    hw_days = hw_days.isel(time=hw_days['time'].dt.month.isin([6, 7, 8]))
    zg500anom_hw = zg500anom.where(hw_days, drop=True)
    
    zg500arr = zg500anom_hw.stack(point=["lat", "lon"]) # Stacks lat/lon into a single dimension
    zg500arr_list.append(zg500arr)
    zg500anom_list.append(zg500anom_hw)

zg500arr_all = xr.concat(zg500arr_list, dim="time")
zg500anom_all = xr.concat(zg500anom_list, dim="time")

np.save(f'{model}_{exp}_data_train.npy', zg500arr_all)
zg500anom_all.coords.to_dataset().to_netcdf(f"{model}_{exp}coords.nc")