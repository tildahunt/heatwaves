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
SOM_functions.py gives functions used in the rest of the SOM code.
Prepping the training data:
- ERA5_heatwaves_in: finds the heatwave days (10% of JJA days (those with the greatest area of heatwaves over Europe)
- hw_in_CanESM5_hist-aer.py: the same but for CanESM5 hist-aer (all ensemble members concatenated). All other model experiments follow the same structure.
- ERA5_data_train: creates the training data required to train the SOMs (flattened zg500 daily patterns for heatwave days and associated file with coordinates)
- data_train_CanESM5_hist-aer.py: the same but for CanESM5 hist-aer (all ensemble members concatenated). All other model experiments follow the same structure.
Getting/ processing the SOMS:
- SOM_size_choice: uses ERA5 training data and calculates distinctness and weighted mean pattern correlations to choose the 4x1 grid.
- get_SOM: trains and saves the SOMs (and best_node time series) for ERA5 and the model data. Includes a bayesian optimisation to find training parameters and uses functions from SOM_functions.py for plotting.
- SOM_processing_and_plot: Uses the SOMs saved from get_SOMs for ERA5 and the model experiments. Reorders the model SOMs to the ERA5 order using cost matrix. Calculates 5-year running average Theil-Sen trends and Mann Kendall p-values, and ensemble and multi-model means. Puts ensemble members with insufficent pattern correlation to their ERA5 counterpart to zero. Plots.

Plot 4: Projection of all JJA days onto the heatwave circulation patterns
- CanESM5 data train: flattens daily zg patterns for all JJA days so they can be compared to the SOM patterns (same method for other models and ERA5).
- ERA5_projection: creates ERA5_node_assignments.nc which is an assignment of every ERA5 JJA day to the ERA5 SOM nodes. It also includes some analysis of this data.
- projection_models_to_ERA5: compares each JJA day of each model experiment to the ERA5 nodes using pattern correlations.
- projection_analysis: Analysis for the node assignments for ERA5 and the model data. Allows you to create a csv which, given a threshold, gives the trends and significance for each node for each period.
- plot_4_plotting_code: uses CSV from above to create figure 4 (calcuating MMMs)
