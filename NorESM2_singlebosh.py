import xarray as xr
import pandas as pd
import numpy as np
import dask
import glob
import os
import warnings
import time as timer
import sys

if len(sys.argv) > 1:
    exp = sys.argv[1]
else:
    exp = "error"  # default

model = 'NorESM2-LM'
daysperyear = 365
dayspersummer = 92


def extract_heatwaves(above_threshold):
    shift_forward = above_threshold.shift(time=-1, fill_value=False)
    shift_2forward = above_threshold.shift(time=-2, fill_value=False)
    shift_backward = above_threshold.shift(time=1, fill_value=False)
    shift_2backward = above_threshold.shift(time=2, fill_value=False)

    heatwave_days = above_threshold & (  
        (shift_forward & shift_backward) |
        (shift_forward & shift_2forward) |
        (shift_backward & shift_2backward)
    )
    
    return heatwave_days

# HEATWAVE DAYS
def count_days(heatwaves):
    return heatwaves.groupby('time.year').sum(dim='time')

def crop_JJA(data):
    summer_data = data.sel(time=data['time.month'].isin([6, 7, 8]) | 
                       ((data['time.month'] == 9) & (data['time.day'] == 1)))
    summer_data = summer_data.where(~((summer_data['time.month'] == 9) & (summer_data['time.day'] == 1)), other=False) #set sept1 to 0
    return summer_data

def heatwave_in_region(land_heatwaves, lat1, lat2):
    weights = np.cos(np.deg2rad(land_heatwaves['lat']))
    region = land_heatwaves.sel(lat = slice(lat1, lat2))
    region_avg = region.weighted(weights).mean(dim=('lat', 'lon'), skipna=True)
    region_avg_dataset = xr.Dataset({"tasmax": region_avg}) 
    region_avg_dataset = region_avg_dataset.chunk(dict(time=-1))
    percentile = region_avg_dataset.tasmax.quantile(0.9, dim="time", skipna=True)
    heatwave_in_region = (region_avg_dataset.tasmax > percentile)
    return heatwave_in_region

input_pattern = f"/gws/nopw/j04/aopp/tildah/tasmax_data/{model}/{exp}/*.nc"
file_list = glob.glob(input_pattern)
print(len(file_list))

# Group files by unique code
for file_path in file_list:
    start_time = timer.time()
    filename = os.path.basename(file_path)
    unique_code = filename.split('_')[4]  # Extracts part like "r11i1p1f3"
    print(unique_code)
    output_path = f"/gws/nopw/j04/aopp/tildah/HW_metrics/{model}/{exp}/{model}_{exp}_{unique_code}_HWF.nc"
    
    if os.path.exists(output_path):
            print(f"Skipping {unique_code} as {output_path} already exists.")
            continue
    data = xr.open_mfdataset(f"/gws/nopw/j04/aopp/tildah/tasmax_data/{model}/{exp}/tasmax_day_{model}_{exp}_{unique_code}_gn_*.nc", combine='by_coords', chunks={'time': daysperyear})   
    #print(data)
    data['lon'] = (data['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    data = data.sel(time = slice('1940-01-01', '2020-12-30'))
    europe_data = data.sel(lon=slice(-25,45), lat=slice(36, 72))
    europe_data = europe_data.chunk({'time': daysperyear, 'lat': len(europe_data.lat), 'lon': len(europe_data.lon)})
    #print(europe_data)
    tasmax = europe_data['tasmax']
    time = tasmax['time']
    tasmax = tasmax.assign_coords(dayofyear=('time', time.dt.dayofyear.data))
    threshold_years = tasmax.sel(time = slice('1940-01-01', '2014-12-30'))

    may25 = pd.Timestamp('2001-05-25').dayofyear 
    sep7  = pd.Timestamp('2001-09-07').dayofyear
    tasmax_JJA_plus = threshold_years.where((tasmax.dayofyear >= may25) & (tasmax.dayofyear <= sep7), drop=True)# 7-summer-7
    tasmax_sorted = tasmax_JJA_plus.sortby('dayofyear')
    #print(tasmax_JJA_plus)
    
    n_years = len(np.unique(tasmax_JJA_plus['time'].dt.year.data)) #75
    window_size = 15 * n_years
    tasmax_rolled = tasmax_sorted.rolling(time=window_size, center=True, min_periods=1).construct('window')
    middle_year = tasmax_sorted['time'].dt.year.values[n_years // 2]
    
    middles = tasmax_rolled.sel(time=tasmax_rolled.time.dt.year == middle_year)
    threshold = middles.quantile(0.9, dim='window')
    #print(threshold)
    JJA_threshold = crop_JJA(threshold)
    JJA_tasmax = crop_JJA(tasmax)
    threshold_by_doy = JJA_threshold.groupby('dayofyear').mean('time')
    above_threshold = JJA_tasmax > threshold_by_doy.sel(dayofyear=JJA_tasmax['dayofyear'])
    above_threshold = above_threshold
    above_threshold = above_threshold.chunk({'time': dayspersummer + 1, 'lat': len(above_threshold.lat), 'lon': len(above_threshold.lon)})
    #print(above_threshold)
    heatwaves = extract_heatwaves(above_threshold)
    #print(heatwaves)
    heatwave_days = count_days(heatwaves)
    #print(heatwave_days)
    
    stats = xr.Dataset({
        'days': heatwave_days,
    })

    stats.to_netcdf(output_path)
    
    end_time = timer.time()
    print(f"Time taken to save NetCDF: {end_time - start_time:.2f} seconds")
    print(f"Processed and saved: {output_path}")