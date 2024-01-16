'''
postprocessing.py
Series of functions to aid the postprocessing of ERUO processed outputs.

Copyright (C) 2021  Alfonso Ferrone

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
import copy
import datetime
import scipy
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4 as nc
import astropy.convolution
from configparser import ConfigParser

from algo import plotting, reconstruction

# --------------------------------------------------------------------------------------------------
# DEBUGGING PARAMETERS
'''
To display the intermediate steps of the processing, just set the following flag to True or any
non-zero number (e.g. 1). To disable the plots, set the flag to False or 0.
For a fine-tuning of the plots, the codes and parameters are available in "plotting.py".
NOTE: SET BOTH TO FALSE BEFORE PROCESSING MORE THAN ONE FILE!
'''
PLOT_INTERF_POSTPROC = 0
PLOT_FINAL_RESULT = 0

ANY_PLOT = PLOT_INTERF_POSTPROC or PLOT_FINAL_RESULT

if ANY_PLOT:
    # matplotlib necessary only if the debugging plots are enabled
    import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# LOADING CONSTANTS AND PARAMETERS
config_fname = "config.ini"
with open(config_fname) as fp:
    config_object = ConfigParser()
    config_object.read_file(fp)


# Postprocessing parameters
postprocessing_info = config_object['POSTPROCESSING_PARAMETERS']

# SNR threshold
MIN_SNR_POSTPROC = float(postprocessing_info['MIN_SNR_POSTPROC'])

# Interference removal
REMOVE_INTERF_POSTPROC = bool(int(postprocessing_info['REMOVE_INTERF_POSTPROC']))
MIN_TIME_FRACTION_INTERF_POSTPROC = float(postprocessing_info['MIN_TIME_FRACTION_INTERF_POSTPROC'])
WINDOW_POSTPROCESS_T = int(postprocessing_info['WINDOW_POSTPROCESS_T'])
WINDOW_POSTPROCESS_R = int(postprocessing_info['WINDOW_POSTPROCESS_R'])
MIN_HALF_FRACTION = float(postprocessing_info['MIN_HALF_FRACTION'])
MIN_RATIO_H_V = float(postprocessing_info['MIN_RATIO_H_V'])
MIN_INTERF_FLAG = int(postprocessing_info['MIN_INTERF_FLAG'])
# Computing an auxiliary quantity: STD of the weight function when looking for interferences
INTEREF_KERNEL_STD = WINDOW_POSTPROCESS_T / 8.0

# Isolated artifacts (noise) removal
REMOVE_NOISE_POSTPROC = bool(int(postprocessing_info['REMOVE_NOISE_POSTPROC']))
MIN_SLICE_LENGHT_NOISE_REMOVAL = int(postprocessing_info['MIN_SLICE_LENGHT_NOISE_REMOVAL'])
MIN_NUM_PIXEL_NOISE_REMOVAL = int(postprocessing_info['MIN_NUM_PIXEL_NOISE_REMOVAL'])

# Debugging parameters
debugging_info = config_object['DEBUGGING_PARAMETERS']
VERBOSE = bool(int(debugging_info['VERBOSE']))
IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
if IGNORE_WARNINGS:
    import warnings
    warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------------


def load_mrr_variables(fpath):
    '''
    Simple function to load the variables we need for postprocessing.

    The processed netCDF file is opened and the necessary variables are retrieved and returned.
    '''
    with nc.Dataset(fpath) as f:
        r = np.array(f.variables['range'])
        t = np.array(f.variables['time'])
        zea = np.array(f.variables['Zea'])
        vel = np.array(f.variables['VEL'])
        s_w = np.array(f.variables['WIDTH'])
        snr = np.array(f.variables['SNR'])
        noise_level = np.array(f.variables['noise_level'])
        noise_floor = np.array(f.variables['noise_floor'])
    return r, t, zea, vel, s_w, snr, noise_level, noise_floor


def identify_interference_lines(zea):
    '''
    Find interference lines in processed files.

    The function identifies suspiciously elongated features in the processed reflectivity matrix.
    The range gates investigated are the ones with anomalously high power, identified during
    the preprocessing.

    Parameters
    ----------
    interf_range_array : numpy array (1D)
        Numpy array (1D) highlighting range with not-empty interference mask.
    '''

    # Preparing some half window quantities in advance
    min_t_len = MIN_HALF_FRACTION * WINDOW_POSTPROCESS_T
    hw_t = int(WINDOW_POSTPROCESS_T / 2.)
    hw_r = int(WINDOW_POSTPROCESS_R / 2.)

    # The indices of possible interferences 
    investigated_r_idx = np.arange(zea.shape[1])[np.sum(np.isfinite(zea), axis=0) > \
                                                 MIN_TIME_FRACTION_INTERF_POSTPROC * zea.shape[0]]
    # We define the extend of the running window (r and t) in advance:
    # - the window width is always constant
    # - when we are near the border, it just becomes uncentered
    t_min_array = np.max(np.stack([np.arange(zea.shape[0], dtype=int) - hw_t, 
                         np.zeros(zea.shape[0], dtype=int)], axis=1), axis=1)
    t_max_array = t_min_array + WINDOW_POSTPROCESS_T
    t_min_array[t_max_array > zea.shape[0]] = zea.shape[0] - WINDOW_POSTPROCESS_T
    t_max_array[t_max_array > zea.shape[0]] = zea.shape[0]

    r_low_array = np.max(np.stack([investigated_r_idx - hw_r,
                         np.zeros(investigated_r_idx.shape, dtype=int)], axis=1), axis=1)
    r_top_array = r_low_array + WINDOW_POSTPROCESS_R
    r_low_array[r_top_array > zea.shape[1]] = zea.shape[1] - WINDOW_POSTPROCESS_R
    r_top_array[r_top_array > zea.shape[1]] = zea.shape[1]

    # Container for the possible interference lines
    interf_flag = np.zeros(zea.shape, dtype=float)

    # How to assign values in the suspect lines
    sus_weights = WINDOW_POSTPROCESS_T * np.array(astropy.convolution.Gaussian1DKernel(
                                                                       INTEREF_KERNEL_STD,
                                                                       x_size=WINDOW_POSTPROCESS_T)) 

    # Running window in range and time to look for suspiciously elongated measure patches
    for i_t in range(zea.shape[0]):
        for i_r in range(investigated_r_idx.shape[0]):
            t_valid = np.sum(np.isfinite(zea[t_min_array[i_t]:t_max_array[i_t],
                                             investigated_r_idx[i_r]]))
            
            if t_valid > min_t_len:
                r_valid = max(1, np.sum(np.isfinite(zea[i_t, r_low_array[i_r]:r_top_array[i_r]])))
                if (t_valid/r_valid) > MIN_RATIO_H_V:
                    closest_to_min_t = t_max_array[i_t] - i_t - hw_t
                    if not closest_to_min_t:
                        interf_flag[t_min_array[i_t]:t_max_array[i_t], investigated_r_idx[i_r]] += \
                                                                                             sus_weights
                    elif closest_to_min_t > 0:
                        interf_flag[t_min_array[i_t]:t_max_array[i_t]-closest_to_min_t,
                                    investigated_r_idx[i_r]] += sus_weights[closest_to_min_t:]
                    else:
                        interf_flag[t_min_array[i_t]-closest_to_min_t:t_max_array[i_t],
                                    investigated_r_idx[i_r]] += sus_weights[:closest_to_min_t]
                
    return interf_flag


def identify_isolated_artifacts_1d(zea_post, to_remove):
    '''
    1D (range) approach to the removal of noise/isolated artifacts
    '''
    for i_t in range(zea_post.shape[0]):
        slices_zea = reconstruction.slice_at_nan(zea_post[i_t, :])
        if len(slices_zea):
            for sl_list in slices_zea:
                if sl_list[1].shape[0] < MIN_SLICE_LENGHT_NOISE_REMOVAL:
                    print(sl_list[0])
                    to_remove[i_t, sl_list[0]] = True

    return to_remove


def identify_isolated_artifacts_2d(zea_post, to_remove):
    '''
    2D (time, range) approach to the removal of noise/isolated artifacts
    '''
    label, num_features = scipy.ndimage.label(np.isfinite(zea_post))

    if num_features:
        # Looping over all contiguous region of the image
        for i_feat in range(1,num_features+1):
                curr_region = (label == i_feat)
                if np.sum(curr_region) < MIN_NUM_PIXEL_NOISE_REMOVAL:
                    to_remove[curr_region] = True

    return to_remove


def postprocess_file(in_fpath, dir_postproc_netcdf, out_fname_suffix):
    '''
    Function to postprocess a single "raw" netCDF MRR file.


    Parameters
    ----------
    in_fpath : str
        Full path to the "raw" netCDF file to process.
    '''
    # Load netCDF file
    r, t, zea, vel, s_w, snr, noise_level, noise_floor = load_mrr_variables(in_fpath)

    # Creating a mask for pixels to remove
    to_remove = np.zeros(zea.shape, dtype='bool')

    # And a copy of a variable to use when looking for pixels to remove
    zea_post = np.full(zea.shape, np.nan)

    # Enforcing SNR threshold before postprocessing
    to_remove[snr < MIN_SNR_POSTPROC] = True
    zea_post[np.logical_not(to_remove)] = zea[np.logical_not(to_remove)]

    # 1. Interference lines removal
    if REMOVE_INTERF_POSTPROC:
        interf_flag = identify_interference_lines(zea_post)

        to_remove[interf_flag > MIN_INTERF_FLAG] = True
        zea_post[to_remove] = np.nan

    # Optional: plotting result of first step
    if PLOT_INTERF_POSTPROC:
        fig, axes = plotting.plot_postprocessing(t, r, zea, zea_post,
                                                 postprocessing_stage='interf. removal')

    # 2. Removal of isolated artifacts (noise)
    if REMOVE_NOISE_POSTPROC:
        to_remove = identify_isolated_artifacts_2d(zea_post, to_remove)

    # 3. Applying the filter
    Zea_post = copy.deepcopy(zea)
    VEL_post = copy.deepcopy(vel)
    WIDTH_post = copy.deepcopy(s_w)
    SNR_post = copy.deepcopy(snr)
    noise_level_post = copy.deepcopy(noise_level)
    noise_floor_post = copy.deepcopy(noise_floor)

    Zea_post[to_remove] = np.nan
    VEL_post[to_remove] = np.nan
    WIDTH_post[to_remove] = np.nan
    SNR_post[to_remove] = np.nan
    noise_level_post[to_remove] = np.nan
    noise_floor_post[to_remove] = np.nan

    # Optional: plotting final results before saving
    if PLOT_FINAL_RESULT:
        fig, axes = plotting.plot_postprocessing(t, r, zea, Zea_post,
                                                 postprocessing_stage='whole postprocessing')

    # Creating a dictionary of filtered variables to save them to the postprocessed netCDF file
    new_vars_dic = {'Zea':Zea_post, 'VEL':VEL_post, 'WIDTH':WIDTH_post, 'SNR':SNR_post,
                    'noise_level':noise_level_post, 'noise_floor':noise_floor_post}

    # ===================
    # Exporting to netCDF
    # ===================
    # 1. Read processed attributes and coordinates
    with xr.open_dataset(in_fpath) as ds_ini:
        coords = ds_ini.coords
        attrs = ds_ini.attrs
        raw_spectrum_units = ds_ini.variables['noise_level'].attrs['units']

    # 2. Add the new information
    attrs['field_names'] = ','.join(new_vars_dic.keys())
    attrs['title'] = attrs['title'] + ' - Post-processed with ERUO'
    attrs['history'] = 'Re-processed with ERUO on %s UTC' % \
                        datetime.datetime.utcnow().strftime('%d/%m/%Y %H:%M:%S')
    
    # 3. Create the new dataset
    units_dataset = {'Zea': 'dBZ',
                     'VEL': 'm/s',
                     'WIDTH': 'm/s',
                     'SNR': 'dB',
                     'noise_level': raw_spectrum_units,
                     'noise_floor': 'dBZ'}
    
    data_vars = {}
    for k in new_vars_dic.keys():
        data_vars[k] = (['time', 'range'], new_vars_dic[k])
    
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # 4. Adding units to variables
    for k in new_vars_dic.keys():
        ds.variables[k].attrs['units'] = units_dataset[k]

    # 5. Saving to file
    # Defining output filepath
    out_fname = os.path.basename(in_fpath).split('.')[0] + out_fname_suffix + '.nc'

    out_fdir = os.path.join(dir_postproc_netcdf, os.sep.join(in_fpath.split(os.sep)[-3:-1]))
    if not os.path.exists(out_fdir):
        os.makedirs(out_fdir)

    out_fpath = os.path.join(out_fdir, out_fname)
    
    if ANY_PLOT:
        plt.show()

    # Exporting to netCDF4
    ds.to_netcdf(out_fpath, mode='w', unlimited_dims='time')
