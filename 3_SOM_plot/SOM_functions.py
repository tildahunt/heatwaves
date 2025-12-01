# functions for SOMs - Should be kept up to date and so can easily be carried accross models etc.
import numpy as np
import minisom
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr, bootstrap
import pickle
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment
from skimage.measure import block_reduce
from scipy.stats import theilslopes
import pymannkendall as mk

"""all functions assume 
zg500_anom: an xarray dataset which contains the coordinates for the heatwave days in europe
x, y: node shapes. Generally x = 1 #Columns, y = 4  # Rows
best_nodes_flat: has dimensions of time, says which node each day of the training data has been assigned to
"""

"""Uses minisom to train SOM"""
def train_som(training, x, y, input_len, sigma, learning_rate, num_iteration):
    som = minisom.MiniSom(
        y,
        x,
        input_len = input_len,
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function='gaussian',
        activation_distance='euclidean',
        sigma_decay_function = 'asymptotic_decay',
        decay_function = 'asymptotic_decay',
        )

    # Initialize SOM weights
    som.random_weights_init(training)

    # Train SOM
    som.train(training, num_iteration=num_iteration, random_order=True, verbose=True)

    return som

"""WEIGHTS DATA BY sqrt LAT"""

def weight_data(data_train, zg500_anom):
    latitudes = zg500_anom['lat'].values  # Extract latitude array
    latitudes_stacked = np.repeat(latitudes, len(zg500_anom['lon']))
    latitudes_expanded = np.tile(latitudes_stacked, (data_train.shape[0], 1))
    lat_weights = np.sqrt(np.cos(np.deg2rad(latitudes_expanded)))
    weighted_data = data_train * lat_weights
    return weighted_data


"""OPENS the SOMs which are saved as pickle files"""


def load_som(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

"""Weighted mean pattern correlation calculation
"""
def get_WMPC(som, best_nodes_flat, data_train, y, x):
    som_patterns = som.get_weights().reshape(y * x, -1)  # Flatten to (num_nodes, input_length)
    # Compute Pearson correlation for each day
    correlations = np.array([
        pearsonr(data_train[i], som_patterns[best_nodes_flat[i]])[0] for i in range(len(data_train))
    ])
    # Compute frequencies of each SOM node
    unique_nodes, counts = np.unique(best_nodes_flat, return_counts=True)
    node_frequencies = counts / len(data_train)  # Normalize to get weights
    # Create weights array: match each day with its node's frequency
    weights = np.array([node_frequencies[unique_nodes == node] for node in best_nodes_flat]).flatten()
    # Compute weighted mean correlation
    weighted_correlation = np.average(correlations, weights=weights)
    return weighted_correlation


"""Plots SOM using axes info, SOM, and number of rows of nodes (assumes x = 1) 
gives the percentage of days assigned to each node"""

def plot_som_simple(som, best_nodes_flat, zg500_anom, y):
    lat_size = len(zg500_anom.lat)
    lon_size = len(zg500_anom.lon)
    
    # Count occurrences of each SOM node across all days
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100  # Convert to percentage
    
    # Create figure with y rows, 1 column for the SOM nodes
    fig, axes = plt.subplots(y, 1, figsize=(8, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    
    # Ensure axes is iterable when y=1
    if y == 1:
        axes = [axes]
    
    # Loop through SOM nodes
    for i, ax in enumerate(axes):
        # Reshape the SOM weights into the (lat, lon) grid
        som_pattern = som._weights[i, 0].reshape(lat_size, lon_size)
    
        # Define the symmetric color range for contour plot centered around zero
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
    
        # Plot SOM pattern
        im = ax.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern, cmap="RdBu_r", 
                         transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
         # Add coastlines and country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    
        # Get frequencies for the current node, default to 0 if missing
        total_day_freq = day_freq.get(i, 0)
    
        # Update title with heatwave frequencies
        ax.set_title(
            f"SOM Node {i + 1} - {total_day_freq:.1f}% of Total Heatwave Days",
            fontsize=12
        )
    
    # Adjust figure layout to make space for the colorbar
    fig.subplots_adjust(bottom=0.15)
    
    # Add a horizontal colorbar at the bottom
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('z500 hPa Geopotential Height Anomaly')
    
    plt.show()


"""Calculates the time series for the whole concatenated time series for 1940-2015 for all models"""

def hw_time_series(zg500_anom, best_nodes_flat): 
    df = pd.DataFrame({
        "time": zg500_anom.time.values,
        "node": best_nodes_flat,
    })

    # Convert time to standard calendar and extract the year
    zg500_anom = zg500_anom.convert_calendar("standard", dim="time", align_on="year")
    df["time"] = pd.to_datetime(zg500_anom.time.values)
    df["year"] = df["time"].dt.year

    # Count occurrences of heatwave days per SOM node per year
    hw_node_counts = df.groupby(["year", "node"]).size().unstack(fill_value=0)

    # Ensure all years (including no-heatwave years) are included
    all_years = list(range(1940, 2015))
    hw_node_counts = hw_node_counts.reindex(all_years, fill_value=0)  # Fill missing years with 0
    node_freq = hw_node_counts.div(hw_node_counts.sum(axis=1), axis=0) * 100  # Convert to %
    hw_node_freq = node_freq.fillna(0)

    return hw_node_counts, hw_node_freq


def time_series(zg500_anom, zg500_anom_HW, best_nodes_flat): 
    df = pd.DataFrame({
        "time": zg500_anom.time.values,
        "node": best_nodes_flat,
    })

    zg500_anom = zg500_anom.convert_calendar("standard", dim="time", align_on="year")
    df["time"] = pd.to_datetime(zg500_anom.time.values)
    df["year"] = df["time"].dt.year

    # --- ALL days ---
    node_counts = df.groupby(["year", "node"]).size().unstack(fill_value=0)

    # --- HEATWAVE days ---
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = df["time"].isin(hw_times)  # boolean mask for heatwave days
    df_hw = df[hw_mask]  # only keep heatwave days

    # Count occurrences per node per year
    hw_node_counts = df_hw.groupby(["year", "node"]).size().unstack(fill_value=0)

    # Ensure all years (including years with zero assignments) are included
    all_years = list(range(1940, 2015))
    node_counts = node_counts.reindex(all_years, fill_value=0)
    hw_node_counts = hw_node_counts.reindex(all_years, fill_value=0)

    return node_counts, hw_node_counts


"""Calculates individual node counts and freqs for each model within the concatenated time series for 1940-2015.
"""

def hw_time_series_per_model(zg500_anom, best_nodes_flat):
    # Convert calendar and get years
    zg500_anom = zg500_anom.convert_calendar("standard", dim="time", align_on="year")
    times = pd.to_datetime(zg500_anom.time.values)
    years = times.year.values

    # Detect model boundaries where year decreases (assuming concatenation order)
    model_start_indices = np.where(years[:-1] > years[1:])[0] + 1
    model_start_indices = np.insert(model_start_indices, 0, 0)
    model_start_indices = list(model_start_indices) + [len(times)]  # add end boundary

    results = []

    for i in range(len(model_start_indices)-1):
        start_idx = model_start_indices[i]
        end_idx = model_start_indices[i+1]

        # Slice times and best_nodes_flat for current model
        model_times = times[start_idx:end_idx]
        model_nodes = best_nodes_flat[start_idx:end_idx]

        # Create DataFrame for this model
        df = pd.DataFrame({
            "time": model_times,
            "node": model_nodes,
        })
        df["year"] = df["time"].dt.year

        # Count heatwave days per node per year
        hw_node_counts = df.groupby(["year", "node"]).size().unstack(fill_value=0)

        # Define the years to cover for this model (min to max in data)
        all_years = list(range(1940, 2015))
        hw_node_counts = hw_node_counts.reindex(all_years, fill_value=0)

        # Convert counts to percentages
        node_freq = hw_node_counts.div(hw_node_counts.sum(axis=1), axis=0) * 100
        hw_node_freq = node_freq.fillna(0)

        results.append((hw_node_counts, hw_node_freq))


""" Uses hw_time_series to plot a time series for the ensemble average on the RHS of the plot
Not really ideal for understanding model spread).
"""

def plot_complex(som, best_nodes_flat, zg500_anom, y):
    # get number of ensemble members
    zg500_anom_years = zg500_anom.convert_calendar("standard", dim="time", align_on="year")
    times = pd.to_datetime(zg500_anom_years.time.values)
    years = times.year.values
    model_start_indices = np.where(years[:-1] > years[1:])[0] + 1
    model_start_indices = np.insert(model_start_indices, 0, 0)
    n_members = len(model_start_indices)

    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100  # Convert to percentage
    time_series, time_series_percents = hw_time_series(zg500_anom, best_nodes_flat)
    time_series = time_series / n_members 
    lat_size = len(zg500_anom.lat)
    lon_size = len(zg500_anom.lon)
    num_som_nodes = y 

    
    # Define the figure with the desired overall size
    fig = plt.figure(figsize=(16, 12))
    
    # Define gridspec for the layout: 1 column for SOM plots and 1 column for time series
    gs = gridspec.GridSpec(nrows=num_som_nodes, ncols=2, width_ratios=[0.3, 0.2], height_ratios=[1]*num_som_nodes)
    
    # Create the axes for the SOM plots and time series plots
    som_axes = []
    time_series_axes = []
    
    for i in range(num_som_nodes):
        # SOM plots in the first column
        ax_som = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        som_axes.append(ax_som)
        
        # Time series plots in the second column
        ax_ts = fig.add_subplot(gs[i, 1])
        time_series_axes.append(ax_ts)
    
    # Plot the SOM patterns and time series
    for i, ax in enumerate(som_axes):
        # Reshape the SOM weights into the (lat, lon) grid and plot them (same as before)
        som_pattern = som._weights[i, 0].reshape(lat_size, lon_size)
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
        im = ax.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern, cmap="RdBu_r", 
                         transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
        ax.set_title(
            f"SOM Node {i+1}: "
            f"HW in EU: {day_freq[i]:.2f}%",
            fontsize=10
        )
    
        # Add colorbar for SOM plots
        if i == 0:
            cax = fig.add_axes([0.16, 0.002, 0.3, 0.015])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label('z500 hPa Geopotential Height Anomaly')

    
    y_max = time_series.max().max()
    pad = 0.05 * y_max
    years = np.arange(1940, 2015)
    for i, ax in enumerate(time_series_axes):
        ax.plot(years, time_series.iloc[:, i], label=f'Total JJA', color='tab:blue', linestyle='-')
        ax.set_title(f'Number of HW Days for SOM {i+1}')
        ax.set_ylim(0, y_max + pad)
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        ax.grid(True)
    
    plt.tight_layout()
    #plt.savefig('ERA5_som_SMALL.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()


"""
Reorders SOMs: 
    Reorder a SOM so that its nodes best match a reference SOM using a cost matrix.

    Parameters
    ----------
    som : trained MiniSom object
        The SOM to reorder.
    ref_som : trained MiniSom object
        The reference SOM.
    lat, lon : 1D arrays
        Latitude and longitude of som grid.
    ref_lat, ref_lon : 1D arrays
        Latitude and longitude of reference som grid.

    Returns
    -------
    reordered_weights : np.ndarray
        Reordered SOM weights with the same shape as som.get_weights().
    order : list
        List of indices giving the mapping from som → ref_som.
    corr matrix: matrix of pearson pattern correlations
    matcehd_corrs: list of pattern correlations for matched nodes 
        (should be in order of the reference som nodes
NOTE THAT COL_IND IS THE OPPOSITE TO ERA5 NODE ORDER OMG. 
COL_IND [3 1 0 2] MEANS THAT 
ORIGINAL NODE 0 IS MAPPED TO NODE 3
ORIFINAL NODE 2 IS MAPPED TO NODE 0 ETC

"""


def reorder_som_to_reference(som, ref_som, lat, lon, ref_lat, ref_lon):
    # Extract weights
    weights_new = som.get_weights()   # shape (n_nodes, 1, lat*lon)
    weights_ref = ref_som.get_weights()

    n_nodes = weights_new.shape[0]

    # Reshape to spatial grids
    patterns_new = weights_new.reshape((n_nodes, len(lat), len(lon)))
    patterns_ref = weights_ref.reshape((n_nodes, len(ref_lat), len(ref_lon)))

    # Interpolate both onto a common grid (choose reference grid)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ref_lon_grid, ref_lat_grid = np.meshgrid(ref_lon, ref_lat)

    common_lon, common_lat = lon, lat
    common_lon_grid, common_lat_grid = np.meshgrid(common_lon, common_lat)
    points_ref = np.column_stack((ref_lon_grid.ravel(), ref_lat_grid.ravel()))
    points_new = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    points_common = np.column_stack((common_lon_grid.ravel(), common_lat_grid.ravel()))

    interp_new = []
    for i in range(n_nodes):
        interp_field = griddata(points_new, patterns_new[i].ravel(),
                                points_common, method='linear')
        interp_new.append(interp_field.reshape(len(lat), len(lon)))
    interp_new = np.array(interp_new)

    interp_ref = []
    for i in range(n_nodes):
        interp_field = griddata(points_ref, patterns_ref[i].ravel(),
                                points_common, method='linear')
        interp_ref.append(interp_field.reshape(len(lat), len(lon)))
    interp_ref = np.array(interp_ref)

    # Build cost matrix (RMSE) and correlation matrix
    cost_matrix = np.zeros((n_nodes, n_nodes))
    corr_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            f_new = interp_new[i].ravel()
            f_ref = interp_ref[j].ravel()

            # mask NaNs for safe comparison
            mask = ~np.isnan(f_new) & ~np.isnan(f_ref)
            if np.any(mask):
                diff = f_new[mask] - f_ref[mask]
                cost_matrix[i, j] = np.sqrt(np.mean(diff**2))
                corr_matrix[i, j] = pearsonr(f_new[mask], f_ref[mask])[0]
            else:
                cost_matrix[i, j] = np.inf
                corr_matrix[i, j] = np.nan

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder weights
    reordered_weights = weights_new[col_ind]

    matched_corrs = corr_matrix[row_ind, col_ind]

    return reordered_weights, col_ind, corr_matrix, matched_corrs


def plot_larger_som_grid(som, best_nodes_flat, zg500_anom, zg500_anom_HW, x, y):
    """
    Plot SOM patterns on a grid of size (y rows, x cols).
    
    Parameters:
        som : trained SOM object (with attribute _weights)
        best_nodes_flat : array of BMU indices for each sample
        zg500_anom : xarray DataArray with dimensions (lat, lon)
        x : number of columns in SOM grid
        y : number of rows in SOM grid
    """
    
    lat_size = len(zg500_anom.lat)
    lon_size = len(zg500_anom.lon)
    
    # Count occurrences of each SOM node across all days
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100  # percentage

    time_array = pd.to_datetime(zg500_anom.time.values)
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = np.isin(time_array, hw_times)
    best_nodes_hw = best_nodes_flat[hw_mask]
    hw_counts = pd.Series(best_nodes_hw).value_counts().sort_index()
    hw_freq = (hw_counts / hw_counts.sum()) * 100
    top_hw_nodes = hw_freq.sort_values(ascending=False).head(4).index.tolist()
    
    # Create grid of plots
    fig, axes = plt.subplots(y, x, figsize=(4*x, 2.5*y), 
                             subplot_kw={"projection": ccrs.PlateCarree()})
    
    # Loop through SOM nodes
    for node in range(x*y):
        row, col = divmod(node, x)
        ax = axes[row, col]
        
        # Reshape SOM weights into the (lat, lon) grid
        som_pattern = som._weights[row, col].reshape(lat_size, lon_size)
        
        # Symmetric color range around zero
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
        
        # Plot SOM pattern
        im = ax.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern,
                         cmap="RdBu_r", transform=ccrs.PlateCarree(),
                         vmin=vmin, vmax=vmax)
        
        # Add coastlines and borders
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

        if node in top_hw_nodes:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        
        # Frequency for this node
        total_day_freq = day_freq.get(node, 0)
        hw_day_freq = hw_freq.get(node, 0) 
        
        # Title
        ax.set_title(f"Node {node+1}\n JJA days: {total_day_freq:.1f}%\nHW days: {hw_day_freq:.1f}%", fontsize=10)
    
    # Adjust layout and add a colorbar
    fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.25, wspace=0.15)
    cax = fig.add_axes([0.2, 0, 0.6, 0.02])  
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('z500 hPa Geopotential Height Anomaly')
    
    fig.suptitle("SOM Patterns", fontsize=14)
    plt.tight_layout()
    plt.show()


"""also for a larger grid but this time with time series"""

def plot_som_grid_with_timeseries(som, best_nodes_flat, zg500_anom, zg500_anom_HW, x, y):

    lat_size = len(zg500_anom.lat)
    lon_size = len(zg500_anom.lon)

    # --- Calculate node percentages ---
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100
    

    time_array = pd.to_datetime(zg500_anom.time.values)
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = np.isin(time_array, hw_times)
    best_nodes_hw = best_nodes_flat[hw_mask]
    hw_counts = pd.Series(best_nodes_hw).value_counts().sort_index()
    hw_freq = (hw_counts / hw_counts.sum()) * 100
    
    
    # Highlight top 4 HW nodes
    top_hw_nodes = hw_freq.sort_values(ascending=False).head(4).index.tolist()

    # --- Calculate annual time series ---
    node_counts, hw_node_counts = time_series(zg500_anom, zg500_anom_HW, best_nodes_flat)
    max_days = max(node_counts.max().max(), hw_node_counts.max().max())
    
    # --- Set up figure ---
    fig = plt.figure(figsize=(5*x, 5*y))  # extra height for time series

    # Loop through SOM nodes
    for node in range(x*y):
        row_idx, col_idx = divmod(node, x)
        gs_top = plt.GridSpec(y*2, x, figure=fig)  # 2 rows per SOM row

        # --- SOM pattern subplot ---
        ax_map = fig.add_subplot(gs_top[row_idx*2, col_idx], projection=ccrs.PlateCarree())
        som_pattern = som._weights[row_idx, col_idx].reshape(lat_size, lon_size)
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
        im = ax_map.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern,
                             cmap="RdBu_r", transform=ccrs.PlateCarree(),
                             vmin=vmin, vmax=vmax)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)

        total_day_freq = day_freq.get(node, 0)
        hw_day_freq = hw_freq.get(node, 0)
        ax_map.set_title(f"Node {node+1}, JJA days: {total_day_freq:.1f}%, HW days: {hw_day_freq:.1f}%", fontsize=10)

        # Highlight top 4 HW nodes
        if node in top_hw_nodes:
            for spine in ax_map.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

        # --- Time series subplot ---
        ax_ts = fig.add_subplot(gs_top[row_idx*2+1, col_idx])
        ax_ts.plot(node_counts.index, node_counts.get(node, 0), color='blue', label='All days')
        ax_ts.plot(hw_node_counts.index, hw_node_counts.get(node, 0), color='red', label='HW days')
        ax_ts.set_ylim(0, max_days)
        ax_ts.set_xlabel('Year', fontsize=8)
        ax_ts.set_ylabel('Days', fontsize=8)
        ax_ts.tick_params(axis='both', labelsize=7)
        if row_idx == 0 and col_idx == 0:
            ax_ts.legend(fontsize=7)

    # Add a single colorbar for all SOM plots
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='z500 hPa Geopotential Height Anomaly')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.suptitle("SOM Patterns with Annual Node Counts", fontsize=14)
    plt.show()

"""for a larger grid, same inputs as above, but plots ONLY the 4 nodes with the largest percentage of HW days"""

def plot_top4_hw_nodes(som, best_nodes_flat, zg500_anom, zg500_anom_HW):

    # --- Calculate node percentages ---
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100

    time_array = pd.to_datetime(zg500_anom.time.values)
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = np.isin(time_array, hw_times)
    best_nodes_hw = best_nodes_flat[hw_mask]
    hw_counts = pd.Series(best_nodes_hw).value_counts().sort_index()
    hw_freq = (hw_counts / hw_counts.sum()) * 100

    # Top 4 HW nodes
    top_hw_nodes = hw_freq.sort_values(ascending=False).head(4).index.tolist()

    # --- Annual time series ---
    node_counts, hw_node_counts = time_series(zg500_anom, zg500_anom_HW, best_nodes_flat)
    max_days = max(node_counts.max().max(), hw_node_counts.max().max())

    # --- Figure setup ---
    fig = plt.figure(figsize=(12, 10)) 
    

    # --- Loop through top 4 nodes ---
    for i, node in enumerate(top_hw_nodes):
        row_idx = i

        # --- SOM pattern subplot ---
        ax_map = fig.add_subplot(4, 2, i*2 + 1, projection=ccrs.PlateCarree())
        # Convert flat node index to grid indices
        grid_x, grid_y = som._weights.shape[:2]
        row_node, col_node = divmod(node, grid_y)
        som_pattern = som._weights[row_node, col_node].reshape(len(zg500_anom.lat), len(zg500_anom.lon))
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
        im = ax_map.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern,
                             cmap="RdBu_r", transform=ccrs.PlateCarree(),
                             vmin=vmin, vmax=vmax)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        total_day_freq = day_freq.get(node, 0)
        hw_day_freq = hw_freq.get(node, 0)
        ax_map.set_title(f"Node {node+1}, JJA days: {total_day_freq:.1f}%, HW days: {hw_day_freq:.1f}%", fontsize=10)

        # --- Time series subplot ---
        ax_ts = fig.add_subplot(4, 2, i*2 + 2)
        ax_ts.plot(node_counts.index, node_counts.get(node, 0), color='blue', label='All days')
        ax_ts.plot(hw_node_counts.index, hw_node_counts.get(node, 0), color='red', label='HW days')
        ax_ts.set_xlabel('Year', fontsize=9)
        ax_ts.set_ylabel('Days', fontsize=9)
        ax_ts.set_ylim(0, max_days)
        ax_ts.tick_params(axis='both', labelsize=8)
        if row_idx == 0:
            ax_ts.legend(fontsize=8)

    # Colorbar for SOM maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='z500 hPa Geopotential Height Anomaly')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    #fig.suptitle("Top 4 HW SOM Nodes and Annual Node Counts", fontsize=14)
    plt.show()


"""calculates the thiel sen trends with mann kendall significance for the large som"""

def large_som_trends(som, best_nodes_flat, zg500_anom, zg500_anom_HW, rows, cols, plotting = True):
    # --- Calculate node percentages ---
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100

    time_array = pd.to_datetime(zg500_anom.time.values)
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = np.isin(time_array, hw_times)
    best_nodes_hw = best_nodes_flat[hw_mask]
    hw_counts = pd.Series(best_nodes_hw).value_counts().sort_index()
    hw_freq = (hw_counts / hw_counts.sum()) * 100

    # Top 4 HW nodes
    top_hw_nodes = hw_freq.sort_values(ascending=False).head(4).index.tolist()

    # --- Annual time series ---
    node_counts, hw_node_counts = time_series(zg500_anom, zg500_anom_HW, best_nodes_flat)
    years = node_counts.index

    # Define periods
    periods = {
        "1940-1979": (1940, 1979),
        "1980-2014": (1980, 2014)
    }

    # Prepare results dictionary
    trends = {}

    for period_name, (start_year, end_year) in periods.items():
        trends[period_name] = {"node_counts": {}, "hw_node_counts": {}}

        mask = (years >= start_year) & (years <= end_year)

        for node in range(node_counts.shape[1]):
            # Define 5-year running average
            y = node_counts.loc[mask, node].rolling(window=5, center=True, min_periods=1).mean().values
            x = years[mask].values
            
            if np.any(~np.isnan(y)):
                slope, intercept, lo_slope, hi_slope = theilslopes(y, x, 0.95)
                mk_result = mk.original_test(y)
            else:
                slope, lo_slope, hi_slope = np.nan, np.nan, np.nan
                mk_result = None
            
            trends[period_name]["node_counts"][node] = {
                "theil_sen_slope": slope,
                "theil_sen_lo": lo_slope,
                "theil_sen_hi": hi_slope,
                "mk_p": mk_result.p if mk_result else np.nan,
                "mk_trend": mk_result.trend if mk_result else None
            }
            
            # Node HW days, 5-year running mean
            y_hw = hw_node_counts.loc[mask, node].rolling(window=5, center=True, min_periods=1).mean().values
            
            if np.any(~np.isnan(y_hw)):
                slope, intercept, lo_slope, hi_slope = theilslopes(y_hw, x, 0.95)
                mk_result = mk.original_test(y_hw)
            else:
                slope, lo_slope, hi_slope = np.nan, np.nan, np.nan
                mk_result = None
            
            trends[period_name]["hw_node_counts"][node] = {
                "theil_sen_slope": slope,
                "theil_sen_lo": lo_slope,
                "theil_sen_hi": hi_slope,
                "mk_p": mk_result.p if mk_result else np.nan,
                "mk_trend": mk_result.trend if mk_result else None
            }

    if plotting == True:
        nodes = range(node_counts.shape[1])  # 2x4 SOM
        periods = ["1940-1979", "1980-2014"]
        width = 0.35  # width of a single bar
    
        fig, axes = plt.subplots(rows, cols, figsize=(18, 8), sharey=True)
        axes = axes.flatten()
    
        for node in nodes:
            ax = axes[node]
    
            # Gather values
            node_vals = [trends[p]["node_counts"][node]["theil_sen_slope"] for p in periods]
            hw_vals = [trends[p]["hw_node_counts"][node]["theil_sen_slope"] for p in periods]
    
            # Gather significance marks
            node_sig = ["*" if trends[p]["node_counts"][node]["mk_p"] is not None and trends[p]["node_counts"][node]["mk_p"] < 0.05 else "" for p in periods]
            hw_sig = ["*" if trends[p]["hw_node_counts"][node]["mk_p"] is not None and trends[p]["hw_node_counts"][node]["mk_p"] < 0.05 else "" for p in periods]
    
            # X locations
            x = np.arange(len(periods))
            
            bars1 = ax.bar(x - width/2, node_vals, width, label="Node counts", color="skyblue")
            bars2 = ax.bar(x + width/2, hw_vals, width, label="HW node counts", color="lightgreen")
    
            # Add * for significance
            for i, bar in enumerate(bars1):
                if node_sig[i]:
                    height = bar.get_height()
                    # if slope < 0, put star at a fixed distance above zero
                    y_pos = height + 0.01*np.max([*node_vals,*hw_vals]) if height >= 0 else 0.01*np.max([*node_vals,*hw_vals])
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos, "*",
                            ha='center', va='bottom', fontsize=14, color='red')
            
            for i, bar in enumerate(bars2):
                if hw_sig[i]:
                    height = bar.get_height()
                    y_pos = height + 0.01*np.max([*node_vals,*hw_vals]) if height >= 0 else 0.01*np.max([*node_vals,*hw_vals])
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos, "*",
                            ha='center', va='bottom', fontsize=14, color='red')

            #highlight top ondes
            if node in top_hw_nodes:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
    
            ax.set_title(f"Node {node+1}")
            ax.set_xticks(x)
            ax.set_xticklabels(periods)
            ax.set_ylabel("Theil-Sen slope (days/year)")
            ax.axhline(0, color='black', linestyle='-', linewidth=1)

        # Inside your plot function, after plotting all bars:

        # Grab the first set of bars for handles
        handles = [bars1[0], bars2[0]]  # one bar from each set
        labels = ["Node counts", "HW node counts"]
        
        fig.legend(handles, labels, loc="upper right", fontsize=12)

        plt.suptitle("Theil-Sen trends per node: total vs HW days with MK significance (5 year running average)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    return trends 

def plot_top4_hw_nodes_with_trends(som, best_nodes_flat, zg500_anom, zg500_anom_HW, rows, cols):
    # --- Calculate node percentages ---
    day_counts = pd.Series(best_nodes_flat).value_counts().sort_index()
    day_freq = (day_counts / day_counts.sum()) * 100

    time_array = pd.to_datetime(zg500_anom.time.values)
    hw_times = pd.to_datetime(zg500_anom_HW.time.values)
    hw_mask = np.isin(time_array, hw_times)
    best_nodes_hw = best_nodes_flat[hw_mask]
    hw_counts = pd.Series(best_nodes_hw).value_counts().sort_index()
    hw_freq = (hw_counts / hw_counts.sum()) * 100

    # Top 4 HW nodes
    top_hw_nodes = hw_freq.sort_values(ascending=False).head(4).index.tolist()

    # --- Annual time series ---
    node_counts, hw_node_counts = time_series(zg500_anom, zg500_anom_HW, best_nodes_flat)
    max_days = max(node_counts.max().max(), hw_node_counts.max().max())

    # --- Calculate trends (5-year running average) ---
    trends = large_som_trends(som, best_nodes_flat, zg500_anom, zg500_anom_HW, rows, cols, plotting = False)

    # --- Figure setup ---
    fig = plt.figure(figsize=(18, 12))  # wider to accommodate 3 columns
        
    all_vals = []
    for node in top_hw_nodes:
        periods = ["1940-1979", "1980-2014"]
        for p in periods:
            all_vals.append(trends[p]["node_counts"][node]["theil_sen_slope"])
            all_vals.append(trends[p]["hw_node_counts"][node]["theil_sen_slope"])
    global_min = min(all_vals)
    global_max = max(all_vals)
    
    # Optional padding (e.g., 10% of range)
    padding = 0.1 * (global_max - global_min)
    ymin = global_min - padding
    ymax = global_max + padding


    # --- Loop through top 4 nodes ---
    for i, node in enumerate(top_hw_nodes):
        row_idx = i

        # --- SOM pattern subplot ---
        ax_map = fig.add_subplot(4, 3, i*3 + 1, projection=ccrs.PlateCarree())
        grid_x, grid_y = som._weights.shape[:2]
        row_node, col_node = divmod(node, grid_y)
        som_pattern = som._weights[row_node, col_node].reshape(len(zg500_anom.lat), len(zg500_anom.lon))
        vmin = -abs(som_pattern).max()
        vmax = abs(som_pattern).max()
        im = ax_map.contourf(zg500_anom.lon, zg500_anom.lat, som_pattern,
                             cmap="RdBu_r", transform=ccrs.PlateCarree(),
                             vmin=vmin, vmax=vmax)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        total_day_freq = day_freq.get(node, 0)
        hw_day_freq = hw_freq.get(node, 0)
        ax_map.set_title(f"Node {node+1}, JJA days: {total_day_freq:.1f}%, HW days: {hw_day_freq:.1f}%", fontsize=10)

        # --- Time series subplot ---
        ax_ts = fig.add_subplot(4, 3, i*3 + 2)
        ax_ts.plot(node_counts.index, node_counts.get(node, 0), color='blue', label='All days')
        ax_ts.plot(hw_node_counts.index, hw_node_counts.get(node, 0), color='red', label='HW days')
        ax_ts.set_xlabel('Year', fontsize=9)
        ax_ts.set_ylabel('Days', fontsize=9)
        ax_ts.set_ylim(0, max_days)
        ax_ts.tick_params(axis='both', labelsize=8)
        if row_idx == 0:
            ax_ts.legend(fontsize=8)

        # --- Trend bar chart subplot ---
        ax_bar = fig.add_subplot(4, 3, i*3 + 3)
        # Gather values for 1940-1979 and 1980-2014
        periods = ["1940-1979", "1980-2014"]
        node_vals = [trends[p]["node_counts"][node]["theil_sen_slope"] for p in periods]
        hw_vals = [trends[p]["hw_node_counts"][node]["theil_sen_slope"] for p in periods]
        node_sig = ["*" if trends[p]["node_counts"][node]["mk_p"] < 0.05 else "" for p in periods]
        hw_sig = ["*" if trends[p]["hw_node_counts"][node]["mk_p"] < 0.05 else "" for p in periods]

        width = 0.35
        x = np.arange(len(periods))
        bars1 = ax_bar.bar(x - width/2, node_vals, width, color='skyblue', label='Node counts')
        bars2 = ax_bar.bar(x + width/2, hw_vals, width, color='lightgreen', label='HW node counts')

        # Add * for significance with slope check
        for i_b, bar in enumerate(bars1):
            if node_sig[i_b]:
                height = bar.get_height()
                y_pos = height + 0.02*np.max([*node_vals,*hw_vals]) if height >= 0 else 0.02*np.max([*node_vals,*hw_vals])
                ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos, "*", ha='center', va='bottom', fontsize=12, color='red')
        for i_b, bar in enumerate(bars2):
            if hw_sig[i_b]:
                height = bar.get_height()
                y_pos = height + 0.02*np.max([*node_vals,*hw_vals]) if height >= 0 else 0.02*np.max([*node_vals,*hw_vals])
                ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos, "*", ha='center', va='bottom', fontsize=12, color='red')

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(periods, rotation=45)
        ax_bar.set_ylabel("Theil-Sen slope (days/yr)")
        ax_bar.set_ylim(ymin, ymax)
        if row_idx == 0:
            ax_bar.legend(fontsize=8)

        ax_bar.axhline(0, color='black', linestyle='-', linewidth=1)

    # Colorbar for SOM maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='z500 hPa Geopotential Height Anomaly')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

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
