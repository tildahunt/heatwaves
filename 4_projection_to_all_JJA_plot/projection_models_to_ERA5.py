import sys
import numpy as np
import minisom
import xarray as xr
import pickle
import warnings
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample
from scipy.stats import theilslopes
import glob
import os
import cftime
np.warnings = warnings
import pandas as pd
from sklearn.utils import resample
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from matplotlib.lines import Line2D
import xesmf as xe

def load_som(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def weight_data(data_train, zg500_anom):
    latitudes = zg500_anom['lat'].values  # Extract latitude array
    latitudes_stacked = np.repeat(latitudes, len(zg500_anom['lon']))
    latitudes_expanded = np.tile(latitudes_stacked, (data_train.shape[0], 1))
    lat_weights = np.sqrt(np.cos(np.deg2rad(latitudes_expanded)))
    weighted_data = data_train * lat_weights
    return weighted_data


def get_HW_mask(zg500anom, HW_zg500_anom, verbose = True):
    import cftime
    from collections import Counter
    """RETURNS A BOOLEAN MASK GIVEN THE COORDINATE DATASETS FOR HEATWAVES AND NON-HEATWAVES
    FOR MULTIPLE ENSEMBLE MEMBERS IN ONE ARRAY"""

    times = zg500anom.time.values
    hw_times_raw = HW_zg500_anom.time.values
    
    # helper to compute year array robustly (your snippet)
    def years_from_times(tarr):
        return np.array([
            t.year if isinstance(t, (cftime.DatetimeNoLeap, cftime.DatetimeGregorian)) else int(str(t)[:4])
            for t in tarr
        ])
    
    # find ensemble boundaries given a time array (returns starts, ends)
    def ensemble_bounds(tarr):
        yrs = years_from_times(tarr)
        starts = np.insert(np.where(np.diff(yrs) < 0)[0] + 1, 0, 0)
        ends = np.append(starts[1:] - 1, len(tarr) - 1)
        return starts, ends
    
    # compute bounds for the full time axis and for the HW times
    starts_full, ends_full = ensemble_bounds(times)
    starts_hw, ends_hw = ensemble_bounds(hw_times_raw)
    
    n_ens_full = len(starts_full)
    n_ens_hw = len(starts_hw)
    
    print(f"ensembles in full time axis: {n_ens_full}, ensembles in HW times: {n_ens_hw}")
    
    # If the numbers don't match, use the smaller number and warn.
    if n_ens_full != n_ens_hw:
        print("Warning: ensemble counts differ between full time axis and HW times.")
    
    n_ens = min(n_ens_full, n_ens_hw)

    # Convert both arrays to pandas Timestamps (string conversion handles cftime -> pandas)
    time_array = pd.to_datetime([str(t) for t in times])
    hw_time_array = pd.to_datetime([str(t) for t in hw_times_raw])
    
    # Prepare final mask
    mask = np.zeros(len(time_array), dtype=bool)
    
    # Process ensemble-by-ensemble
    for i in range(n_ens):
        i0_full, i1_full = starts_full[i], ends_full[i]
        i0_hw, i1_hw = starts_hw[i], ends_hw[i]
    
        # slice views
        full_slice = time_array[i0_full:i1_full+1]
        hw_slice = hw_time_array[i0_hw:i1_hw+1]
    
        # If there are no HW times in this ensemble, continue
        if len(hw_slice) == 0:
            # nothing to flag in this ensemble
            continue
    
        # create mask for this ensemble: mark True for any positions in full_slice that match hw_slice
        # This marks every repeated occurrence inside the ensemble (if full_slice contains repeats)
        ensemble_mask = np.isin(full_slice, hw_slice)
    
        # put ensemble_mask into overall mask
        mask[i0_full:i1_full+1] = ensemble_mask

    if verbose == True:
        print("Final mask length:", len(mask))
        print("Total True (flagged times):", mask.sum())
        print("Should be = total occurrences of HW times across ensembles:", len(hw_time_array))

    if mask.sum() != len(hw_time_array):
        print("Number of Trues in the mask does NOT equal number of HW days")

    return mask, starts_full

if len(sys.argv) > 1:
    model = sys.argv[1]
else:
    model = "error"  # default

ERA5_som = load_som('/home/users/tildah/Internship/CIRCULATION/SOMs/ERA5_som_paper_order.pkl')
ERA5_nodes = np.load('/home/users/tildah/Internship/CIRCULATION/SOMs/ERA5_best_nodes_paper_order.npy')
ERA5_zg500_anom = xr.open_dataset('/home/users/tildah/Internship/CIRCULATION/ERA5/ERA5_coords.nc')

experiments = ['hist-GHG', 'hist-aer', 'historical']

for exp in experiments:      
    print(model, exp)

    # -----------------------------
    # 1. Load SOM, node assignments, and coordinates
    # -----------------------------
    som = ERA5_som
    ERA5_coords = xr.open_dataset('/home/users/tildah/Internship/CIRCULATION/ERA5/ERA5_coords.nc')
    print(ERA5_coords)
    
    model_coords_ds = xr.open_dataset(f'~/Internship/CIRCULATION/data_train/{model}_{exp}coords.nc')
    times = model_coords_ds['time'].values
    HW_zg500_anom = model_coords_ds
    
    zg_daily_flat = np.load(f'/gws/nopw/j04/aopp/tildah/projection/{model}_{exp}_zg_daily_flat.npy')
    zg500_anom = xr.open_dataset(f"//gws/nopw/j04/aopp/tildah/projection/{model}_{exp}_coords.nc")

    # -----------------------------
    # CREATES HW MASK ... NOT NEEDED FOR CORE CODE. JUST FOR CHECKING THINGS ARE WORKING.
    # Also returns the starting index of each ensemble member for the all JJA days zg dataset.
    # -----------------------------
    hw_mask, starts = get_HW_mask(zg500_anom, HW_zg500_anom, verbose = False)
    print(starts)

    # Extract SOM patterns from MiniSom
    som_weights = som.get_weights()  # shape: (nx, ny, lat, lon)
    print(som_weights.shape)
    nx, ny, space = som_weights.shape
    som_flat = som_weights.reshape(nx*ny, space)  # shape: (n_nodes, space)
    print(som_flat.shape)
    
    # Function to compute Pearson correlation (numpy arrays only)
    def pattern_correlation(daily, nodes):
        n_time = daily.shape[0]
        n_nodes = nodes.shape[0]
        corr = np.empty((n_time, n_nodes))
         
        for t in range(n_time):
            for n in range(n_nodes):
                corr[t, n], _ = pearsonr(daily[t], nodes[n])  # ignore p-value  
        
        return corr

    print(zg_daily_flat.shape)
    
    # Compute correlations
    correlations = pattern_correlation(zg_daily_flat, som_flat)
    print("Correlations shape:", correlations.shape)  # (time, n_nodes)
    #print(correlations[hw_mask])
    
    best_node_idx = np.argmax(correlations, axis=1)           # shape: (time,)
    best_node_corr = correlations[np.arange(correlations.shape[0]), best_node_idx]


    # -----------------------------
    # SANITY CHECK
    # -----------------------------

    # SAVE RESULTS HERE!
    # --- 1. Load or define data arrays ---
    times = zg500_anom['time'].values
    times = pd.to_datetime([str(t) for t in times])
    
    N_total = best_node_idx.size
    if times.size != N_total:
        raise ValueError(f"Length mismatch: times ({times.size}) vs arrays ({N_total}).")
    
    starts = np.asarray(starts, dtype=int)
    n_members = len(starts)
    
    # Compute lengths for each member
    lengths = np.empty(n_members, dtype=int)
    for i in range(n_members - 1):
        lengths[i] = starts[i+1] - starts[i]
    lengths[-1] = N_total - starts[-1]
    
    # --- 2. Build member_of_sample array ---
    # Create an array of length N_total with member numbers (1-indexed)
    member_of_sample = np.empty(N_total, dtype=int)
    for m, s in enumerate(starts):
        member_of_sample[s:s + lengths[m]] = m + 1  # +1 so members start from 1 not 0
    
    # --- 3. Build the xarray Dataset ---
    tds = xr.Dataset(
        data_vars=dict(
            best_node_idx      = ("time", best_node_idx.astype(np.int32)),
            best_node_corr     = ("time", best_node_corr.astype(np.float32))
        ),
        coords=dict(
            time=("time", times),
            member=("time", member_of_sample),
        ),
        attrs=dict(
            description=f"{model} / {exp}: SOM assignments & distances for each ensemble member concatenated along time.",
        )
    )
    
    mask = tds['member'] == 1
    print(tds.where(mask, drop=True))
    
    # --- 4. Save to NetCDF ---
    out_nc = f"{model}_{exp}_node_assignments.nc"
    tds.to_netcdf(out_nc)
    print(f"✅ Saved dataset to {out_nc}")