# heatwaves

Description of each file given in this repository

0 Getting HWF: 
- ERA5 HWF: produces a netcdf file of HWF for ERA5 data (given downloaded tasmax data of a particular format)
- Example Model HWF: produces a netcdf file of HWF for each member of NorESM (given experiment, and using the data downloaded to JASMIN). Slight tweaks are required for the different models/ experiments (ie when the historical experiment is extended using SP245, or to account for the different calendars/ grids used), but the overall code structure is identical for all models: ACCESS-ESM1-5, CanESM5, CMCC-CM2-SR5, HadGEM3-GC31-LL, MIROC6, MPI-ESM1-2-LR, NorESM2-LM. 

Plot 1: This plot shows ERA5 spatially / annually averaged HWF and TX time series and maps of ERA5 and MMM HWF trends.
- get trends: takes in HWF files and gets spatial trends and mk significance for trends for periods (1940, 1979), (1980, 2020), (1940,2020). Does not do sea masking or any regridding. The same code is used for ERA5.
- plotting code for plot 1

Plot 2: Attribution of trends to single forcings
- trends then average: using the HWF files, calculates trends at each grid point and then averages regionally saves to a csv for each member. This is used in the (b) part of the plot.
- plotting code: Generates plot 2 and related numbers. Alternate method of calculating the regional HWF and then the trend is included as a method (alternative csv) but was not ultimately used.

Plot 3: Self Organising Maps
- f

Plot 4: Projection of all JJA days onto the heatwave circulation patterns
- CanESM5 data train: flattens daily zg patterns for all JJA days so they can be compared to the SOM patterns (same method for other models and ERA5).
- 
- 
