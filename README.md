# heatwaves

Description of each file given in this repository

0 Getting HWF: 
- ERA5 HWF: produces a netcdf file of HWF for ERA5 data (given downloaded tasmax data of a particular format)
- Example Model HWF: produces a netcdf file of HWF for each member of NorESM (given experiment, and using the data downloaded to JASMIN). Slight tweaks are required for the different models/ experiments (ie when the historical experiment is extended using SP245, or to account for the different calendars/ grids used), but the overall code structure is identical for all models: ACCESS-ESM1-5, CanESM5, CMCC-CM2-SR5, HadGEM3-GC31-LL, MIROC6, MPI-ESM1-2-LR, NorESM2-LM. 

Plot 1: This plot shows ERA5 spatially / annually averaged HWF and TX time series and maps of ERA5 and MMM HWF trends.
- why
- d
