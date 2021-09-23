'''
reconstruction.py
Series of functions to aid the reconstruction of the spectra affected by interference.

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

import copy
import scipy
import numpy as np
import astropy.convolution
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve
from configparser import ConfigParser
from algo import plotting

# --------------------------------------------------------------------------------------------------
# DEBUGGING PARAMETERS
'''
To display the intermediate steps of the processing, just set the following flag to True or any
non-zero number (e.g. 1). To disable the plots, set the flag to False or 0.
For a fine-tuning of the plots, the codes and parameters are available in "plotting.py".
NOTE: SET PLOT_RECONSTRUCTION TO FALSE BEFORE PROCESSING MORE THAN ONE FILE!
'''
PLOT_RECONSTRUCTION = 0
# The example time step to plot
IDX_T_PLOT = 60

if PLOT_RECONSTRUCTION:
    # matplotlib necessary only if the debugging plots are enabled
    import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------------------
# LOADING CONSTANTS AND PARAMETERS
# This section access directly the "config.ini" file.
config_fname = "config.ini"
with open(config_fname) as fp:
    config_object = ConfigParser()
    config_object.read_file(fp)

spectrum_reconstruction_parameters = config_object['SPECTRUM_RECONSTRUCTION_PARAMETERS']

MARGIN_SMALL_INTERF_DETECTION = \
                        int(spectrum_reconstruction_parameters['MARGIN_SMALL_INTERF_DETECTION'])
MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF = \
                    int(spectrum_reconstruction_parameters['MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF'])
FRACTION_VEL_LINE_INTERFERENCE = \
                        float(spectrum_reconstruction_parameters['FRACTION_VEL_LINE_INTERFERENCE'])
ADIACIENTIA_WIDTH = int(spectrum_reconstruction_parameters['ADIACIENTIA_WIDTH'])
EXCEPTIONAL_ANOMALY_THREHSOLD = \
               float(spectrum_reconstruction_parameters['EXCEPTIONAL_ANOMALY_THREHSOLD'])
HORIZONTAL_TOL = \
               int(spectrum_reconstruction_parameters['HORIZONTAL_TOL_SPECTRUM_RECONSTRUCTION'])
MIN_WIN_RECONSTRUCTION = int(spectrum_reconstruction_parameters['MIN_WIN_RECONSTRUCTION'])
KERNEL_SCALE_FACTOR = float(spectrum_reconstruction_parameters['KERNEL_SCALE_FACTOR'])
NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION = \
               int(spectrum_reconstruction_parameters['NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION'])
MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED = \
               float(spectrum_reconstruction_parameters['MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED'])

# Debugging parameters
debugging_info = config_object['DEBUGGING_PARAMETERS']
VERBOSE = bool(int(debugging_info['VERBOSE']))
IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
if IGNORE_WARNINGS:
    import warnings
    warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------------


def slice_at_nan(a):
    '''
    Simple function to split array in slices of contiguous non-NaN values.

    The function has been inspired by one of the answers at:
    https://stackoverflow.com/questions/14605734/numpy-split-1d-array-of-chunks-separated-by-nans-into-a-list-of-the-chunks

    Parameters
    ----------
    a : numpy.array (1D)
        The input array to split at NaN locations

    Returns
    -------
    [...] : list
        A list of list, the latter containing two elements: a slice, with the indexes of non-Nan
        entries, and a list of the contiguous non-NaN values.
    '''
    return [[s, a[s]] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def define_reficiendo(spectrum_3d, median_line_tiled, interference_mask_2d):
    '''
    Identifies the region of the spectrum, at each time step, to be reconstructed.

    The anomaly is defined as the difference between the 3D spectrum and the "smoothed median line",
    which is one of the preprocessing product. In this function, the spectrum and anomaly are 3D
    because they are the matrix containing all the spectra/anomalies from the file that is
    currently being processed.
    The interference mask, on the other hand, is 2D, since it is the same at every time step, so
    we decided to keep only the range and velocity dimensions.
    The section of the anomaly that are flagged "True" by the interference mask and above a certain 
    threshold, will be flagged in "reficiendo" and will be reconstructed. This threshold is fixed in
    "config.ini", as "MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED".
    The resulting "reficiendo" 3D matrix is finally returned.
    
    Parameters
    ----------
    spectrum_3d : numpy.array (3D, float or equivalent)
        All the spectra from a file, concatenated in a single 3D matrix. If the border correction
        has been enabled, the function assumes that it has been added to the spectrum before
        providing it in input to this function. (dimensions: time, range, velocity)
    median_line_tiled : numpy.array (3D, float or equivalent)
        The estimated median noise level at each range for the whole dataset (one of the
        preprocessing output), converted from 1D to 3D. The conversion is performed by repeating
        the profile "m" times across the velocity dimension, and "num_t" across the temporal one.
        (dimension: time, range, velocity)
    interference_mask_2d : numpy.array (2D, bool)
        The two dimensional matrix, flagging with "True" the regions likely to contain and 
        interference, and with "False" all other regions of the spectrum. It is a preprocessing
        product. (dimensions: range, velocity)


    Returns
    -------
    anomaly_3d : numpy.array (3D, float or equivalent)
        The difference between the spectra from a MRR-PRO file and the median smoothed noise floor
        for the dataset. (dimensions: time, range, velocity)
    reficiendo_3d : numpy.array (3D, float or equivalent)
        The region of the anomaly_3d to be reconstructed. (dimensions: time, range, velocity)

    '''
    # Auxiliary params
    num_t = spectrum_3d.shape[0]
    r_idx = np.arange(spectrum_3d.shape[1])
    
    # Casting the 2d Interference mask to 3d
    interference_mask_3d = np.tile(interference_mask_2d, (num_t, 1, 1))

    # Computing the anomaly
    anomaly_3d = np.array(spectrum_3d - median_line_tiled)

    # The region to reconstruct:
    # 1) Starting from a simple thresholding
    reficiendo_3d_raw = np.logical_and(interference_mask_3d,
                                       anomaly_3d > MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED)

    reficiendo_3d_v2 = np.zeros(reficiendo_3d_raw.shape, dtype=bool)

    # 2) Choosing either small interferences, or the ones spanning the whole spectrum
    for i_t in range(reficiendo_3d_v2.shape[0]):
        label, num_features = scipy.ndimage.label(reficiendo_3d_raw[i_t, :, :])

        if num_features:
            for i_feat in range(1,num_features+1):
                curr_masked = (label == i_feat)
                curr_adiacentia = np.logical_xor(curr_masked,
                                                 scipy.ndimage.binary_dilation(curr_masked,
                                                    iterations=MARGIN_SMALL_INTERF_DETECTION))
                curr_adiacentia[np.logical_xor(curr_masked, reficiendo_3d_raw[i_t, :, :])] = False
                if np.sum(anomaly_3d[i_t,:,:][curr_adiacentia] > \
                         MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED) < MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF:
                    # We now look for small isolated interferences
                    reficiendo_3d_v2[i_t, :, :] += curr_masked
                else:
                    # If not isolated, let's see if it's spanning the whole spectum
                    num_masked_in_range = np.sum(curr_masked, axis=1)
                    affected_by_line_interf = r_idx[num_masked_in_range > \
                                                    FRACTION_VEL_LINE_INTERFERENCE*spectrum_3d.shape[2]]
                    reficiendo_3d_v2[i_t, affected_by_line_interf, :] += curr_masked[affected_by_line_interf]
    reficiendo_3d = reficiendo_3d_v2 > 0

    gates_reficiendi = np.sum(reficiendo_3d, axis=2) > 0

    # We keep the peak when we are surrounded by significant precipitation
    for i_t in range(reficiendo_3d.shape[0]):
        for i_r in r_idx[gates_reficiendi[i_t,:]]:
            curr_anom_max = np.nanmax(anomaly_3d[i_t, i_r, :])
            if curr_anom_max > EXCEPTIONAL_ANOMALY_THREHSOLD:
            
                valid_below = np.where(np.logical_and(~gates_reficiendi[i_t,:], r_idx < i_r))[0]
                valid_above = np.where(np.logical_and(~gates_reficiendi[i_t,:], r_idx > i_r))[0]

                closest_valid_below = valid_below[0-min(valid_below.shape[0], ADIACIENTIA_WIDTH):]
                closest_valid_above = valid_above[0:min(valid_above.shape[0], ADIACIENTIA_WIDTH)]

                if closest_valid_below.shape[0] > 3:
                    closest_valid_below = closest_valid_below[:-1][np.diff(closest_valid_below)<2]
                    valid_anom_below = anomaly_3d[i_t, closest_valid_below, :]
                    max_anom_below = np.nanmedian(np.nanmax(valid_anom_below, axis=1))
                    max_anom_pos_below = np.nanmedian(np.nanargmax(valid_anom_below, axis=1))
                else:
                    max_anom_below = 0.
                    max_anom_pos_below = -999

                if closest_valid_above.shape[0] > 3:
                    closest_valid_above = closest_valid_above[1:][np.diff(closest_valid_above)<2]
                    valid_anom_above = anomaly_3d[i_t, closest_valid_above, :]
                    max_anom_above = np.nanmedian(np.nanmax(valid_anom_above, axis=1))
                    max_anom_pos_above = np.nanmedian(np.nanargmax(valid_anom_above, axis=1))
                else:
                    max_anom_above = 0.
                    max_anom_pos_above = -999

                if ((max_anom_below > EXCEPTIONAL_ANOMALY_THREHSOLD) or \
                                    (max_anom_above > EXCEPTIONAL_ANOMALY_THREHSOLD)):
                    curr_anom_max_pos = np.nanargmax(anomaly_3d[i_t, i_r, :])

                    if np.abs(curr_anom_max_pos - max_anom_pos_below) < HORIZONTAL_TOL or \
                                    np.abs(curr_anom_max_pos - max_anom_pos_above) < HORIZONTAL_TOL:
                        reficiendo_3d[i_t, i_r, curr_anom_max_pos] = False

    if PLOT_RECONSTRUCTION:
        plotting.plot_spectrum_reconstruction(anomaly_3d[IDX_T_PLOT,:,:], reficiendo_3d[IDX_T_PLOT,:,:])
        plt.show()

    return anomaly_3d, reficiendo_3d


def reconstruct_anomaly(anomaly, reficiendo):
    '''
    Reconstruct the "anomaly" of a spectrum from a single time step.

    The anomaly is defined as the difference between a spectrum and the "smoothed median noise
    floor", which is one of the preprocessing product.
    The anomaly used in this function comes from a single spectrum, therefore is from a single
    time step of a file (anmd this is the reason why it is 2D).
    The reficiendo matrix is the output of the function "define_reficiendo", and it is defined as 
    the section of the anomaly that are flagged "True" by the interference mask and above a certain 
    threshold. This threshold is fixed in "config.ini".
    The recostructed anomaly is finally returned by the function.
    
    Parameters
    ----------
    anomaly : numpy.array (2D, float or equivalent)
        The difference between a single spectrum, from a spectific time step, and the median
        smoothed noise floor for the dataset. (dimensions: range, velocity)
    reficiendo : numpy.array (2D, bool)
        The two dimensional matrix, flagging with "True" the regions to be reconstructed, and with
        "False" all other regions of the spectrum. (dimensions: range, velocity)


    Returns
    -------
    reconstructed_anomaly : numpy.array (2D, float or equivalent)
        The anomaly obtained by re-constructing the section of the input anomaly below the
        interference mask and above a fixed threshold. (dimensions: range, velocity)

    '''
    # Auxiliary params
    m = anomaly.shape[1]

    # Deciding the size of the kernel for the reconstruction
    has_at_least_one_masked = ~np.any(reficiendo, axis=1)
    masked_tmp = np.ma.masked_array(has_at_least_one_masked, mask=has_at_least_one_masked)
    sections_masked = np.ma.clump_unmasked(masked_tmp)
    # If the section is too small, we use a default value, defined in config.ini
    if len(sections_masked):
        longest_masked_section = MIN_WIN_RECONSTRUCTION
        for sl in sections_masked:
            if masked_tmp[sl].shape[0] > longest_masked_section:
                longest_masked_section = masked_tmp[sl].shape[0]
    else:
        longest_masked_section = MIN_WIN_RECONSTRUCTION

    # Standard deviation for reconstruction
    y_std = int(np.ceil((longest_masked_section)/KERNEL_SCALE_FACTOR))

    # Preparation of the matrix for the reconstruction
    valid_spectrum_extend = np.logical_or(np.sum(np.isfinite(anomaly), axis=1) > 0, 
                                           np.sum(reficiendo, axis=1) > 0)
    # SKipping lowest part of the spectrum (it causes artifacts in reconstruction otherwise)
    valid_spectrum_extend[0:NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION] = False

    valid_anomaly = anomaly[valid_spectrum_extend, :]
    valid_anomaly[reficiendo[valid_spectrum_extend, :]] = np.nan

    # A margin is added at the extremes of the range
    vertical_dim_reconstr = np.sum(valid_spectrum_extend) + (2 * longest_masked_section)
    img_for_reconstruction = np.zeros((vertical_dim_reconstr, anomaly.shape[1]))

    img_for_reconstruction[longest_masked_section:-longest_masked_section, :] = valid_anomaly
    # The margin is filled with the average of the two gates closest to the extremes
    to_fill_bot = np.nanmean(np.stack([valid_anomaly[0,:],
                                       valid_anomaly[1,:]], axis=1), axis=1)

    to_fill_top = np.nanmean(np.stack([valid_anomaly[-1,:],
                                       valid_anomaly[-2,:]], axis=1), axis=1)
    for i in range(longest_masked_section):
        img_for_reconstruction[i, :] = to_fill_bot
        img_for_reconstruction[-i, :] = to_fill_top

    # Reconstructing the anomaly
    kernel = Gaussian2DKernel(x_stddev=1, y_stddev=y_std)
    fixed_img = interpolate_replace_nans(img_for_reconstruction, kernel,
                                         convolve=convolve, boundary='wrap')

    # From the artificially extended image to the anomaly
    fixed_anomaly = copy.deepcopy(anomaly)
    fixed_anomaly[valid_spectrum_extend, :] = \
                                        fixed_img[longest_masked_section:-longest_masked_section, :]

    return fixed_anomaly
