""" Script for finding beaver disturbances in wetlands
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd
import yaml

# import YATSM functions
import yatsm
from yatsm.io import read_line
from yatsm.utils import csvfile_to_dataframe, get_image_IDs
# from yatsm.config_parser import convert_config, parse_config_file
import yatsm._cyprep as cyprep

# Define image reading function
def read_image(f):
    ds = gdal.Open(f, gdal.GA_ReadOnly)
    return ds.GetRasterBand(1).ReadAsArray()


# State/condition enumeration
CONDITION_OPEN_WATER = 1
CONDITION_DIST_VEG = 2
CONDITION_UNDIST_VEG = 3

NDV = -9999

# SPECIFY disturbance parameters
T_TCB_diff = -500   # change in annual mean TCB
T_TCB_sum = -500	# cumulative difference in annual mean TCB

T_TCG_veg = 0.70    # % reduction in TCG amplitude (relative to TS start)
T_TCG_open = 750	# TCG amplitude threshold for open water conditions

# Read in example image for dimensions, map creation
example_img_fn = '/projectnb/landsat/projects/Massachusetts/Broadmoor_medium/images/example_img'
example_img = read_image(example_img_fn)
py_dim = example_img.shape[0]
px_dim = example_img.shape[1]
print('Shape of example image:')
print(example_img.shape)

# Up front -- declare hard coded dataset attributes (for now)
BAND_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'therm', 'tcb', 'tcg', 'tcw']
n_band = len(BAND_NAMES)
dtype = np.int16

## SPECIFY YATSM CONFIG FILE
config_file = '/projectnb/landsat/projects/Massachusetts/Broadmoor_medium/Broadmoor_config_pixel.yaml'

# Read in and parse config file
cfg = yaml.load(open(config_file))
# List to np.ndarray so it works with cyprep.get_valid_mask
cfg['dataset']['min_values'] = np.asarray(cfg['dataset']['min_values'])
cfg['dataset']['max_values'] = np.asarray(cfg['dataset']['max_values'])

# Get files list
df = csvfile_to_dataframe(cfg['dataset']['input_file'], \
                          date_format=cfg['dataset']['date_format'])


# Initialize disturbance and vegetation condition arrays
# Number of years in time series currently hardcoded as 29 (1985-2014)
disturbances = np.zeros((py_dim, px_dim, 29))
veg_cond = np.zeros((py_dim, px_dim, 29))

# Get dates for image stack
df['image_ID'] = get_image_IDs(df['filename']) 
df['x'] = df['date'] 
dates = df['date'].values

# Loop over rows and columns, read and mask time series, find disturbance events
for py in range(0, py_dim): # row iterator
    print('Working on row {py}'.format(py=py))
    Y_row = read_line(py, df['filename'], df['image_ID'], cfg['dataset'],
                      px_dim, n_band + 1, dtype,  # +1 for now for Fmask
                      read_cache=True, write_cache=False,
                      validate_cache=False)
    #print('Read in the data...')

    for px in range(0, px_dim): # column iterator
        Y = Y_row.take(px, axis=2)
        #import pdb; pdb.set_trace()
        
        if (Y[0:6] == NDV).mean() > 0.3:
            continue
        else: # process time series for disturbance events
            
            # Mask based on physical constraints and Fmask 
            valid = cyprep.get_valid_mask( \
                        Y, \
                        cfg['dataset']['min_values'], \
                        cfg['dataset']['max_values']).astype(bool)

            # Apply mask band
            idx_mask = cfg['dataset']['mask_band'] - 1
            valid *= np.in1d(Y.take(idx_mask, axis=0), \
                                     cfg['dataset']['mask_values'], \
                                     invert=True).astype(np.bool)

            # Mask time series using fmask result
            Y_fmask = np.delete(Y, idx_mask, axis=0)[:, valid]
            dates_fmask = dates[valid]

            # Apply multitemporal mask - original time series (no fmask)
            # Multi-temp only TS used for TCG range - preserves more winter obs
            # Step 1. mask where green > 3 stddev
            multitemp1 = np.where(Y[1] < (np.mean(Y_fmask[1])+np.std(Y_fmask[1])*3))
            dates_multi = dates[multitemp1[0]] 
            Y_multi = Y[:, multitemp1[0]]
            # Step 2. mask where swir < 3 std dev
            multitemp2 = np.where(Y_multi[4] > (np.mean(Y_fmask[4])-np.std(Y_fmask[4])*3))
            dates_multi = dates_multi[multitemp2[0]] 
            Y_multi = Y_multi[:, multitemp2[0]]

            # Apply multi-temporal mask - fmask time series
            # Fully masked TS used for TCB mean
            # Step 1. mask where green > 3 stddev
            multitemp1_fmask = np.where(Y_fmask[1] < (np.mean(Y_fmask[1])+np.std(Y_fmask[1])*3))
            dates_fmask = dates_fmask[multitemp1_fmask[0]] 
            Y_fmask = Y_fmask[:, multitemp1_fmask[0]]
            # Step 2. mask where swir < 3 std dev
            multitemp2_fmask = np.where(Y_fmask[4] > (np.mean(Y_fmask[4])-np.std(Y_fmask[4])*3))
            dates_fmask = dates_fmask[multitemp2_fmask[0]] 
            Y_fmask = Y_fmask[:, multitemp2_fmask[0]]

            # convert time from ordinal to dates
            dt_dates_multi = np.array([dt.datetime.fromordinal(d) for d in dates_multi])
            dt_dates_fmask = np.array([dt.datetime.fromordinal(d) for d in dates_fmask])

            # Create dataframes for analysis
            # Fmasked + multitemporal masked data (for TCB TS)
            # Step 1. reshape data
            shp_ = dt_dates_fmask.shape[0]
            dt_dates_fmask_csv = dt_dates_fmask.reshape(shp_, 1)
            Y_fmask_csv = np.transpose(Y_fmask)
            data_fmask = np.concatenate([dt_dates_fmask_csv, Y_fmask_csv], axis=1)
            col_names = ['date', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'therm', 'tcb', 'tcg', 'tcw']
            # Step 2. create dataframe
            data_fmask_df = pd.DataFrame(data_fmask, columns=col_names)
            # convert reflectance to int
            data_fmask_df[BAND_NAMES] = data_fmask_df[BAND_NAMES].astype(int)
            # Step 3. group by year to generate annual TS
            year_group_fmask = data_fmask_df.groupby(data_fmask_df.date.dt.year)
            # get years in time series 
            years_fmask = np.asarray(year_group_fmask.groups.keys()) 
            years_fmask = years_fmask.astype(int)

            # Multitemporal only masked data (for TCG TS)
            # Step 1. reshape data
            shp_ = dt_dates_multi.shape[0]
            dt_dates_multi_csv = dt_dates_multi.reshape(shp_, 1)
            Y_multi_csv = np.transpose(Y_multi)
            data_multi = np.concatenate([dt_dates_multi_csv, Y_multi_csv], axis=1)
            col_names = ['date', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'therm', 'tcb', 'tcg', 'tcw', 'fmask']
            # Step 2. create dataframe
            data_multi_df = pd.DataFrame(data_multi, columns=col_names)
            # convert reflectance to int
            data_multi_df[BAND_NAMES] = data_multi_df[BAND_NAMES].astype(int) # convert reflectance to int
            # Step 3. group by year to generate annual TS
            year_group_multi = data_multi_df.groupby(data_multi_df.date.dt.year) # annual time series
            # get years in time series 
            years_multi = np.asarray(year_group_multi.groups.keys()) # years in time series
            years_multi = years_multi.astype(int)

            # TC Brightness Change Detection - Flood
            # Calculate mean annual TCB
            TCB_mean = year_group_fmask['tcb'].mean()
            #import pdb; pdb.set_trace()
            # Calculate year-to-year difference in mean TCB
            TCB_mean_diff = np.diff(TCB_mean)
            # Cumulative sum of annual difference in TCB
            TCB_mean_sum = np.cumsum(TCB_mean_diff)

            # TC Greenness Change Detection - Vegetation
            # Find annual min TCG
            TCG_min = year_group_multi['tcg'].min()
            # Find annual max TCG
            TCG_max = year_group_multi['tcg'].max()  
            # Calculate annual range TCG       
            TCG_amp = np.asarray(TCG_max - TCG_min)
            # Normalize to first year's value (assuming forest)
            TCG_amp_adj = TCG_amp.astype(float) / TCG_amp[0].astype(float) 

            # Detect disturbance based on TCB/TCG thresholds
            # Record year of disturbance
            length = (years_fmask.size - 1)
            for index, year in enumerate(years_fmask): 
                if index < length:
                    if ((((TCB_mean_diff[index] < T_TCB_diff) and (TCG_amp_adj[index+1] < T_TCG_veg)) or \
                    ((TCB_mean_sum[index] < T_TCB_sum) and (TCG_amp_adj[index+1] < T_TCG_veg))) and \
                    (np.mean(TCB_mean)<3000)):
                        disturbances[py, px, index] = year+1
                        #print('{row}, {col} - {year}'.format(row=py, col=px, year=year))
                else:
                    continue

            # Record vegetation condition 
            #(IN TESTING - NOT WORKING RIGHT YET!)            
            for index, year in enumerate(years_fmask):
                if index < length:
                    if (TCG_amp[index] < T_TCG_open): 
                        veg_cond[py, px, index] = CONDITION_OPEN_WATER
                    elif (TCG_amp[index] > T_TCG_open and TCG_amp_adj[index] < T_TCG_veg):   
                        veg_cond[py, px, index] = CONDITION_DIST_VEG
                    elif TCG_amp_adj[index] > T_TCG_veg: 
                        veg_cond[py, px, index] = CONDITION_UNDIST_VEG
                else:
                    continue          
print('Analysis of disturbance complete!')

# Output disturbance map for each year
in_ds = gdal.Open(example_img_fn, gdal.GA_ReadOnly)
for index, year in enumerate(years_fmask):
    if index < length:
        disturbance_fn = './{year}_disturbance.tif'.format(year=year+1)

        out_driver = gdal.GetDriverByName("GTiff")
        out_ds = out_driver.Create(disturbance_fn, 
                                   example_img.shape[1],  # x size
                                   example_img.shape[0],  # y size
                                   1,  # number of bands
                                   gdal.GDT_UInt32)
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        out_ds.GetRasterBand(1).WriteArray(disturbances[:, :, index])
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        out_ds.GetRasterBand(1).SetDescription('Beaver Disturbances')
        out_ds = None

for index, year in enumerate(years_fmask):
    if index < length:
        condition_fn = './{year}_condition.tif'.format(year=year)

        out_driver = gdal.GetDriverByName("GTiff")
        out_ds = out_driver.Create(condition_fn, 
                                   example_img.shape[1],  # x size
                                   example_img.shape[0],  # y size
                                   1,  # number of bands
                                   gdal.GDT_UInt32)
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        out_ds.GetRasterBand(1).WriteArray(veg_cond[:, :, index])
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        out_ds.GetRasterBand(1).SetDescription('Vegetation Condition')
        out_ds = None        
print('Mapped results complete!')
