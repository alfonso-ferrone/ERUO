'''
processing.py
Series of functions to aid the processing of a dataset of MRR-PRO measurements.

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
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4 as nc
import scipy
import scipy.signal
import astropy.convolution
from configparser import ConfigParser
from scipy.optimize import curve_fit
from algo import plotting, reconstruction

# --------------------------------------------------------------------------------------------------
# DEBUGGING PARAMETERS
'''
To display the intermediate steps of the processing, just set the following flag to True or any
non-zero number (e.g. 1). To disable the plots, set the flag to False or 0.
For a fine-tuning of the plots, the codes and parameters are available in "plotting.py".
NOTE: SET ALL TO FALSE BEFORE PROCESSING MORE THAN ONE FILE!
'''
PLOT_RAW_PEAKS = 0
PLOT_RAW_LINES = 0
PLOT_LINES_WITHOUT_DUPLICATES = 0
PLOT_ACCEPTED_LINES = 0
PLOT_SPECTRUM_AROUND_PEAKS = 0
PLOT_NOISE_MASKED_SPECTRUM = 0
PLOT_PRODUCTS_LINEAR = 0
PLOT_FINAL_PRODUCTS = 0

ANY_PLOT = PLOT_RAW_PEAKS or PLOT_RAW_LINES or PLOT_LINES_WITHOUT_DUPLICATES or \
           PLOT_ACCEPTED_LINES or  PLOT_SPECTRUM_AROUND_PEAKS or PLOT_NOISE_MASKED_SPECTRUM or \
           PLOT_PRODUCTS_LINEAR or PLOT_FINAL_PRODUCTS

if ANY_PLOT:
    # matplotlib necessary only if the debugging plots are enabled
    import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# LOADING CONSTANTS AND PARAMETERS
# This section access directly the "config.ini" file.
config_fname = "config.ini"
with open(config_fname) as fp:
    config_object = ConfigParser()
    config_object.read_file(fp)

# Constants (and derived quantities)
fixed_params = config_object['FIXED_PARAMETERS']
f_s = float(fixed_params['f_s'])
lam = float(fixed_params['lam'])
c = float(fixed_params['c'])
k2 = float(fixed_params['k2'])
# Derived quantity: constant for Z calculation
const_z_calc = ((10.**18) * (lam**4) * k2) / (np.pi**5)

# Name of variables in input NetCDF (needed to automatically load correct thresholds)
var_names_info = config_object['INPUT_NECDF_VARIABLE_NAMES']
spectrum_varname = var_names_info['SPECTRUM_VARNAME']

# Parameters for transfer function problems handling
transfer_function_parameters = config_object['TRANSFER_FUNCTION_PARAMETERS']
USE_EXTERNAL_TRANSFER_FUNCTION = bool(int(transfer_function_parameters['USE_EXTERNAL_TRANSFER_FUNCTION']))
EXTERNAL_TRANSFER_FUNCTION_PATH = transfer_function_parameters['EXTERNAL_TRANSFER_FUNCTION_PATH']
if USE_EXTERNAL_TRANSFER_FUNCTION:
    # The transfer function reconstruction is mutually exclusive with the usage of the external one
    RECONSTRUCT_TRANSFER_FUNCTION = False
else:
    RECONSTRUCT_TRANSFER_FUNCTION = bool(int(transfer_function_parameters['RECONSTRUCT_TRANSFER_FUNCTION']))

# Parameters for spectrum reconstruction
spectrum_reconstruction_parameters = config_object['SPECTRUM_RECONSTRUCTION_PARAMETERS']
RECONSTRUCT_SPECTRUM = bool(int(spectrum_reconstruction_parameters['RECONSTRUCT_SPECTRUM']))

# Parametersor the processing
processing_parameters_info = config_object['SPECTRUM_PROCESSING_PARAMETERS']

# Peak identification
if spectrum_varname == 'spectrum_reflectivity':
    PROMINENCE_THRESHOLD = float(processing_parameters_info['PROMINENCE_THRESHOLD_REFLECTIVITY'])
else:
    PROMINENCE_THRESHOLD = float(processing_parameters_info['PROMINENCE_THRESHOLD_RAW_SPECTRUM'])
RELATIVE_PROMINENCE_THRESHOLD = float(processing_parameters_info['RELATIVE_PROMINENCE_THRESHOLD'])
MAX_NUM_PEAKS_AT_R = int(processing_parameters_info['MAX_NUM_PEAKS_AT_R'])

# Connecting peaks in lines
WINDOW_R = float(processing_parameters_info['WINDOW_R'])
WINDOW_V = float(processing_parameters_info['WINDOW_V'])
MIN_NUM_PEAKS_IN_LINE = int(processing_parameters_info['MIN_NUM_PEAKS_IN_LINE'])

# Cleaning and noise floor identification
VEL_TOL = float(processing_parameters_info['VEL_TOL'])
DA_THRESHOLD = float(processing_parameters_info['DA_THRESHOLD'])

# Cleaning noise floor
NOISE_STD_FACTOR = float(processing_parameters_info['NOISE_STD_FACTOR'])
CORRECT_NOISE_LVL = bool(int(processing_parameters_info['CORRECT_NOISE_LVL']))
NOISE_CORR_WINDOW = float(processing_parameters_info['NOISE_CORR_WINDOW'])
MAX_DIFF_NOISE_LVL = float(processing_parameters_info['MAX_DIFF_NOISE_LVL'])

# Computing moments
CALIB_CONST_FACTOR = float(processing_parameters_info['CALIB_CONST_FACTOR'])

# Removal of isolated peaks from spectrum
REMOVE_ISOLATED_PEAK_SPECTRUM = bool(int(processing_parameters_info['REMOVE_ISOLATED_PEAK_SPECTRUM']))

# Debugging parameters
debugging_info = config_object['DEBUGGING_PARAMETERS']
VERBOSE = bool(int(debugging_info['VERBOSE']))
IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
if IGNORE_WARNINGS:
    import warnings
    warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------------------


def compute_additional_mrr_parameters(N, m, T_i, d_r, verbose=False):
    '''
    Function to get some basic info
    '''
    # Dependent
    I = (f_s * T_i) / (2. * N * m)   # Number of incoherently averaged spectra
    f_ny = f_s / (2. * N)             # Nyquist frequency range
    v_ny = (lam * f_s) / (4. * N)     # Nyquist velocity range
    d_f = f_s / (2. * m * N)         # Frequency resolution
    d_t = m * f_ny                     # Time resolution of one spectrum (single measurement)
    d_v = (lam * f_s) / (4. * N * m) # Velocity resolution
    H = N * d_r                         # Height range

    # Velocity bins
    v_0 = np.arange(0., d_v * m, d_v)

    # Creating a dictionary with all the info
    info_dic = {'I': I, 'f_ny': f_ny, 'v_ny': v_ny, 'd_f': d_f, 'd_t': d_t, 'd_v': d_v, 'H': H,
                'v_0': v_0}

    if verbose:
        print('----------\nConfigurables')
        print('N: ', N)
        print('m: ', m)
        print('T_i: ', T_i)
        print('d_r: ', d_r)
        print('----------\nDependent')
        print('I: ', I)
        print('f_ny: ', f_ny)
        print('v_ny: ', v_ny)
        print('d_f: ', d_f)
        print('d_t: ', d_t)
        print('d_v: ', d_v)
        print('H: ', H)
        print('----------\nVelocity bins')
        # print('v_0: ', v_0)
        print('shape v0: ', v_0.shape)
        print('v_min, v_max: ', v_0.min(), v_0.max())

    return info_dic


def transfer_fun_exp(x, a, c, d):
    '''
    The function used in fitting the top part of the transfer function.

    The function used is a simple exponential with 3 parameters.
    '''
    return a * np.exp(-c * x) + d


def reconstruct_transfer_function_old(transfer_function, r, max_value_tranfer_fun=9.e9):
    '''
    Old version of the transfer function reconstruction
    '''
    # Finding if there is a problem to solve, otherwise just return the transfer function
    if ~np.any(transfer_function > max_value_tranfer_fun):
        return transfer_function
    # Condition to select acceptable range gates
    cond_acceptable = transfer_function < max_value_tranfer_fun

    # Creating a new transfer function, copying already the acceptable part
    new_transfer_fun = np.full(transfer_function.shape, np.nan)
    new_transfer_fun[cond_acceptable] = transfer_function[cond_acceptable]

    # That's where the transfer fun. start to decrease with height
    descent_point = np.where(np.gradient(new_transfer_fun) < 0)[0][0]
    descent_cond = np.logical_and(np.arange(r.shape[0]) > descent_point,
                                      np.isfinite(new_transfer_fun))

    # Splitting this descending part in half
    second_half_split = int(np.where(descent_cond)[0][0] + \
                            0.5*(np.where(descent_cond)[0][-1] - \
                                 np.where(descent_cond)[0][0]))
    # Chosing the second half of the descending part for the fit
    second_half_cond = np.logical_and(np.arange(r.shape[0]) > second_half_split,
                                      np.isfinite(new_transfer_fun))
    # And the missing part outside the acceptable condition as the one to recreate
    to_regenerate_cond = np.logical_and(np.arange(r.shape[0]) > second_half_split,
                                        np.isnan(new_transfer_fun))
    
    # Fitting an exponential
    popt, pcov = curve_fit(transfer_fun_exp, r[second_half_cond], transfer_function[second_half_cond],
                           p0=(1., 1e-3, 0.1))

    # Reconstructing final part
    fit_results =  transfer_fun_exp(r[to_regenerate_cond], *popt)
    # Temporary placeholder
    raw_merged_transfer_function = copy.deepcopy(new_transfer_fun)
    raw_merged_transfer_function[to_regenerate_cond] = fit_results

    # Smoothing all together for a cleaner transition
    kernel = astropy.convolution.Box1DKernel(width=10)
    smoothed_transfer_fun = astropy.convolution.convolve(raw_merged_transfer_function, kernel,
                                                         boundary='extend',
                                                         nan_treatment='interpolate',
                                                         preserve_nan=False)

    # Using the smooth version for the missing part of the new transfer fun.
    new_transfer_fun[to_regenerate_cond] = smoothed_transfer_fun[to_regenerate_cond]
    return new_transfer_fun


def reconstruct_transfer_function(transfer_function, max_value_tranfer_fun=9.e9):
    '''
    Stretch the transfer function to cover missing gates.

    If the transfer function at the upper range gates is above a certain threshold
    (normally it should be between 0 and 1), we stretch the part not above that threshold
    to cover the full range.
    '''
    # Finding if there is a problem to solve, otherwise just return the transfer function
    if ~np.any(transfer_function > max_value_tranfer_fun):
        return transfer_function

    # Condition to select acceptable range gates
    cond_acceptable = transfer_function < max_value_tranfer_fun

    # Creating a new transfer function, stretchingthe acceptable part
    new_transfer_fun = scipy.signal.resample(transfer_function[cond_acceptable],
                                             transfer_function.shape[0])

    # Making sure that the new transfer function does not have a higher peak than the old one
    new_transfer_fun *= np.nanmax(transfer_function[cond_acceptable])/np.nanmax(new_transfer_fun)
    return new_transfer_fun


def repeat_spectra(all_spectra, transfer_function):
    '''
    Function that opens a netCDF file from the MRR and returns all the spectra in a file.
    '''
    # Repeating the spectrum 3 times
    spectrum_before = np.full(all_spectra.shape, np.nan)
    spectrum_after = np.full(all_spectra.shape, np.nan)

    spectrum_before[:, :-1, :] = all_spectra[:, 1:, :]
    spectrum_after[:, 1:, :] = all_spectra[:, :-1, :]

    tiled_spectra = np.concatenate([spectrum_before, all_spectra, spectrum_after], axis=2)
    all_spectra_x3_lin = np.power(10., tiled_spectra / 10.)

    # Now re-shaping the transfer function to apply it later
    m_x3 = all_spectra_x3_lin.shape[2]
    transfer_function_x3 = np.tile(transfer_function, (m_x3, 1)).T

    return all_spectra_x3_lin, transfer_function_x3


def find_raw_peaks(spec, N, m, max_num_peaks_at_r=6):
    '''
    Function to find peaks
    '''
    # Coordinates of peak
    r_idx_peaks_list = []
    v_idx_peaks_list = []
    # Left and right side of peak
    v_l_idx_peaks_list = []
    v_r_idx_peaks_list = []

    for i_r in range(N):
        peaks, properties = scipy.signal.find_peaks(spec[i_r,:],
                                                    prominence=PROMINENCE_THRESHOLD,
                                                    height=0.)
        
        if len(peaks):
            if len(peaks) > max_num_peaks_at_r:
                peak_heights = properties['peak_heights']
                peak_order = np.argsort(peak_heights)
                peaks = peaks[peak_order][-max_num_peaks_at_r:]
                for k in properties.keys():
                    properties[k] = properties[k][peak_order][-max_num_peaks_at_r:]

            # Only secondary peaks high enough relative to the first one are kept
            accepted = properties['prominences'] > RELATIVE_PROMINENCE_THRESHOLD * \
                                                        np.max(properties['prominences'])

            r_idx_peaks_list.append((np.ones(np.sum(accepted))*i_r))
            v_idx_peaks_list.append(peaks[accepted])
            v_l_idx_peaks_list.append(properties['left_bases'][accepted])
            v_r_idx_peaks_list.append(properties['right_bases'][accepted])

    if len(r_idx_peaks_list):
        # Creating unique numpy array
        r_idx_peaks = np.concatenate(r_idx_peaks_list).astype(int)
        v_idx_peaks = np.concatenate(v_idx_peaks_list).astype(int)
        v_l_idx_peaks = np.concatenate(v_l_idx_peaks_list).astype(int)
        v_r_idx_peaks = np.concatenate(v_r_idx_peaks_list).astype(int)
        # An integer index, unique for each peak, used later in the analysis
        idx_peaks = np.arange(r_idx_peaks.shape[0], dtype='int')

        return r_idx_peaks, v_idx_peaks, v_l_idx_peaks, v_r_idx_peaks, idx_peaks
    else:
        return [], [], [], [], []


def find_raw_lines(spec, v_0_3, r, r_idx_peaks, v_idx_peaks, idx_peaks):
    '''
    Function to unite closeby peaks in lines, and get properties of these lines
    '''
    # --------------------------------------------------------------------------------------------
    # 1. Connecting peaks in lines
    lines = [[]]

    for i_peak in idx_peaks:
        curr_r = r_idx_peaks[i_peak]
        curr_v = v_idx_peaks[i_peak]
        # Only peaks in the window (parameters at beginning of code)
        elegible = np.logical_and(np.logical_and(np.abs(curr_r - r_idx_peaks) < WINDOW_R,
                                                  np.abs(curr_v - v_idx_peaks) < WINDOW_V),
                                  idx_peaks > i_peak)
        # If any 
        if np.sum(elegible):
            elegible_r = r_idx_peaks[elegible]
            elegible_v = v_idx_peaks[elegible]
            elegible_idx = idx_peaks[elegible]
            # The distance always favor peaks in next range gate, if in the window
            distance2 = (1 + WINDOW_V**2) * np.square(curr_r - elegible_r) + np.square(curr_v - elegible_v) 
            closest_idx = elegible_idx[np.argmin(distance2)]

            for l in lines:
                if i_peak in l:
                    # If the peak is in the line, its neighbor goes to the same line
                    l.append(closest_idx)
                    break
            else:
                # Otherwise peak and neighbor create a new line
                lines.append([i_peak, closest_idx])

    # --------------------------------------------------------------------------------------------
    # 2. Connecting lines properties
    line_v_idx = []
    line_r_idx = []

    line_v = []
    line_r = []
    line_pow_lin = []

    line_min_r = []
    line_max_r = []
    line_median_v = []
    line_median_pow_lin = []            

    lines_array = []
    for l in lines:
        # We keep only lines with MIN_NUM_PEAKS_IN_LINE
        if len(l) >= MIN_NUM_PEAKS_IN_LINE:
            l_array = np.array(l, dtype=int)
            lines_array.append(l_array)

            line_v_idx.append(v_idx_peaks[l_array])
            line_r_idx.append(r_idx_peaks[l_array])

            line_v.append(v_0_3[v_idx_peaks[l_array]])
            line_r.append(r[r_idx_peaks[l_array]])
            line_pow_lin.append(spec[r_idx_peaks[l_array], v_idx_peaks[l_array]])

            line_min_r.append(np.nanmin(line_r[-1]))
            line_max_r.append(np.nanmax(line_r[-1]))

            # For the median v, we use only the top half of the line
            idx_half_line_v = int(np.floor(len(line_v[-1])/2.))
            line_median_v.append(np.nanmedian(line_v[-1][idx_half_line_v:]))
            line_median_pow_lin.append(np.nanmedian(line_pow_lin[-1]))

    # That's a lot of return...maybe we can reduce it later
    return lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r, \
            line_max_r, line_median_v, line_median_pow_lin


def exclude_duplicate_lines(v_ny, lines_array, line_v_idx, line_r_idx, line_v, line_r,
                            line_pow_lin, line_min_r, line_max_r, line_median_v):
    '''
    Removing peaks repaeted at approximately v_ny, choosing the one in the line closest to 0 m/s
    '''
    # 1. Identifying when there are potential conflicts between lines
    # To numpy array
    array_min_r = np.array(line_min_r)
    array_max_r = np.array(line_max_r)
    array_median_v = np.array(line_median_v)

    matrix_min_r_1, matrix_min_r_2 = np.meshgrid(array_min_r, array_min_r)
    matrix_max_r_1, matrix_max_r_2 = np.meshgrid(array_max_r, array_max_r)
    matrix_median_v_1, matrix_median_v_2 = np.meshgrid(array_median_v, array_median_v)
    
    # Looking for lines with overlapping range
    cond1 = matrix_min_r_1 <= matrix_max_r_2
    cond2 = matrix_min_r_2 <= matrix_max_r_1
    cond_r = np.logical_and(cond1, cond2)

    # And lines at +/- multiples of v_ny (with tolerance)
    cond_v = np.zeros(cond_r.shape, dtype=bool)
    for i in range(len(lines_array)-1):
        for j in range(i+1, len(lines_array)):
            lines_intersect, comm1, comm2 = np.intersect1d(line_r_idx[i], line_r_idx[j],
                                                           return_indices=True, assume_unique=False)
            if len(lines_intersect):
                diff = np.abs(np.nanmedian(line_v[i][comm1] - line_v[j][comm2]))
                pow_diff = np.abs(np.nanmedian(line_pow_lin[i][comm1] - line_pow_lin[j][comm2]))

                if np.isclose(diff, v_ny, atol=VEL_TOL) or \
                    np.isclose(diff, 2.*v_ny, atol=VEL_TOL) or \
                    np.isclose(diff, 3.*v_ny, atol=VEL_TOL):  # and np.isclose(pow_diff, 0., atol=1.e3)
                    cond_v[i,j] = True
                    cond_v[j,i] = True

    # Combining conditions
    cond = np.logical_not(np.logical_and(cond_r, cond_v))

    cond[np.logical_and(np.tile(np.any(np.logical_not(cond), axis=0), (cond.shape[0], 1)),
                        np.identity(cond.shape[0], dtype=bool))] = False


    # Some tricks to vectorize the check without doing loops
    v_investigated = ma.masked_array(np.abs(matrix_median_v_1), mask=cond)

    no_conflict = np.all(cond, axis=1)
    idx_no_conflict = np.arange(no_conflict.shape[0])[no_conflict]

    idx_conflict_all = np.argmin(v_investigated, axis=1)[np.logical_not(no_conflict)]
    y_conflict_all = np.arange(no_conflict.shape[0])[np.logical_not(no_conflict)]

    idx_conflict = np.unique(np.intersect1d(idx_conflict_all, y_conflict_all))

    # 2. Resolving conflicts and selecting the accepted lines
    accepted_lines = []

    accepted_lines_v_idx = []
    accepted_lines_r_idx = []

    accepted_lines_v = []
    accepted_lines_r = []

    accepted_lines_min_r = []
    accepted_lines_max_r = []

    # These last two list shall be converted to array during return
    accepted_lines_v_med = []
    accepted_lines_pow_lin_max = []

    for i_idx, curr_idx in enumerate(idx_no_conflict):
        accepted_lines.append(lines_array[curr_idx])

        accepted_lines_v_idx.append(line_v_idx[curr_idx])
        accepted_lines_r_idx.append(line_r_idx[curr_idx])

        accepted_lines_v.append(line_v[curr_idx])
        accepted_lines_r.append(line_r[curr_idx])

        accepted_lines_min_r.append(line_min_r[curr_idx])
        accepted_lines_max_r.append(line_max_r[curr_idx])

        # For the median v, we use only the top half of the line
        idx_half_line_v = int(np.floor(len(line_v[curr_idx])/2.))
        accepted_lines_v_med.append(np.nanmedian(line_v[curr_idx][idx_half_line_v:]))
        accepted_lines_pow_lin_max.append(np.nanmax(line_pow_lin[curr_idx]))

    for i_idx, curr_idx in enumerate(idx_conflict):
        # The best candidate among the current overlapping ones
        curr_best_line_idx = np.argmin(v_investigated[curr_idx,:])

        if not curr_best_line_idx == curr_idx:
            # If the chosen one is not the best among all lines overlapping with it
            # we remove from the line range gates that are already in the "best line"
            r_idx_to_keep = np.setdiff1d(line_r_idx[curr_idx], line_r_idx[curr_best_line_idx],
                                         assume_unique=True)
            if len(r_idx_to_keep):
                mask_valid = np.in1d(line_r_idx[curr_idx], r_idx_to_keep)

                accepted_lines.append(lines_array[curr_idx][mask_valid])

                accepted_lines_v_idx.append(line_v_idx[curr_idx][mask_valid])
                accepted_lines_r_idx.append(line_r_idx[curr_idx][mask_valid])

                accepted_lines_v.append(line_v[curr_idx][mask_valid])
                accepted_lines_r.append(line_r[curr_idx][mask_valid])

                accepted_lines_min_r.append(np.min(line_r[curr_idx][mask_valid]))
                accepted_lines_max_r.append(np.max(line_r[curr_idx][mask_valid]))

                # For the median v, we use only the top half of the (remaining part of the) line
                idx_half_line_v = int(np.floor(len(line_v[curr_idx][mask_valid])/2.))
                accepted_lines_v_med.append(
                                       np.nanmedian(line_v[curr_idx][mask_valid][idx_half_line_v:]))
                accepted_lines_pow_lin_max.append(np.nanmax(line_pow_lin[curr_idx][mask_valid]))
        else:
            # Otherwise we proceed as usual (export whole line)
            accepted_lines.append(lines_array[curr_idx])

            accepted_lines_v_idx.append(line_v_idx[curr_idx])
            accepted_lines_r_idx.append(line_r_idx[curr_idx])

            accepted_lines_v.append(line_v[curr_idx])
            accepted_lines_r.append(line_r[curr_idx])

            accepted_lines_min_r.append(line_min_r[curr_idx])
            accepted_lines_max_r.append(line_max_r[curr_idx])

            # For the median v, we use only the top half of the line
            idx_half_line_v = int(np.floor(len(line_v[curr_idx])/2.))
            accepted_lines_v_med.append(np.nanmedian(line_v[curr_idx][idx_half_line_v:]))
            accepted_lines_pow_lin_max.append(np.nanmax(line_pow_lin[curr_idx]))

    
    return accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx, accepted_lines_v, \
           accepted_lines_r, accepted_lines_min_r, accepted_lines_max_r, \
           np.array(accepted_lines_v_med), np.array(accepted_lines_pow_lin_max)


def exclude_lines_far_from_main_one(v_ny, accepted_lines, accepted_lines_v_idx, accepted_lines_r_idx,
                                    accepted_lines_v, accepted_lines_r, accepted_lines_min_r,
                                    accepted_lines_max_r, accepted_lines_v_med_array,
                                    accepted_lines_pow_lin_max_array):
    '''
    Excludes the lines too far from the one with highest maximum power
    '''
    # Choosing main line
    idx_main_line = np.argmax(np.array(accepted_lines_max_r) - np.array(accepted_lines_min_r))
    # Selecting idx of lines not too far from it
    dist_v_from_main_line = np.abs(accepted_lines_v_med_array - accepted_lines_v_med_array[idx_main_line])
    accepted_idx_dist = dist_v_from_main_line < v_ny

    # Re-defining all lists
    accepted_lines_v2 = []

    accepted_lines_v_idx_v2 = []
    accepted_lines_r_idx_v2 = []

    accepted_lines_v_v2 = []
    accepted_lines_r_v2 = []

    for i_idx in np.arange(len(accepted_lines))[accepted_idx_dist]:
        accepted_lines_v2.append(accepted_lines[i_idx])

        accepted_lines_v_idx_v2.append(accepted_lines_v_idx[i_idx])
        accepted_lines_r_idx_v2.append(accepted_lines_r_idx[i_idx])

        accepted_lines_v_v2.append(accepted_lines_v[i_idx])
        accepted_lines_r_v2.append(accepted_lines_r[i_idx])
        
    return accepted_lines_v2, accepted_lines_v_idx_v2, accepted_lines_r_idx_v2, accepted_lines_v_v2, accepted_lines_r_v2


def extract_spectrum_around_peaks(spec, m, r_idx_peaks, v_idx_peaks, v_l_idx_peaks, v_r_idx_peaks, accepted_lines_v2):
    '''
    Function to extract exactly "m" (=num. lines in spectrum from MRR config file) velocity bins
    around the accepted peaks.
    Peaks are sorted by power and the higest ones is favored (its left/right borders are added first).
    If adding a secondary peak makes the spectrum at a cerain "r_i" too wide, we exclude that secondary peak.
    '''
    mask_spec = np.ones(spec.shape, dtype=bool)
    peak_spectrum_masked_dic = {}
    indexes_v = np.arange(spec.shape[1], dtype=int)
    
    # Masking the peak position
    for l in accepted_lines_v2:
        curr_peak_r = r_idx_peaks[l]
        curr_peal_v = v_idx_peaks[l]
        mask_spec[curr_peak_r, curr_peal_v] = False

        # Dictionary used in compute_noise_lvl_std
        for i_r_idx, r_idx in enumerate(curr_peak_r):
            if r_idx in peak_spectrum_masked_dic.keys():
                peak_spectrum_masked_dic[r_idx].append(v_idx_peaks[l][i_r_idx])
            else:
                peak_spectrum_masked_dic[r_idx] = [v_idx_peaks[l][i_r_idx]]

    masked_spectrum = np.ma.masked_array(spec, mask=mask_spec)
    
    # Add vel. bins until we have m around the peak 
    for i_r in np.where(np.logical_and(np.sum(np.logical_not(mask_spec), axis=1) < m,
                                       np.sum(np.logical_not(mask_spec), axis=1) > 0))[0]:
        num_gates_to_add = m - np.sum(1-mask_spec[i_r,:])
        while num_gates_to_add > 0:
            # We use erosion to expand the mask arounf the valid spectrum
            # Note: origin=num_gates_to_add%2 allows to cycle between the two sides
            erosion = scipy.ndimage.binary_erosion(mask_spec[i_r,:], border_value=1)

            # And then we do a xor between the result and the old mask to get the eroded pixels
            candidates_to_add = indexes_v[np.logical_xor(erosion, mask_spec[i_r,:])]

            to_add = candidates_to_add[np.argmax(spec[i_r,:][candidates_to_add])]
            mask_spec[i_r, to_add] = False
            num_gates_to_add = m - np.sum(1-mask_spec[i_r,:])

    masked_spectrum = np.ma.masked_array(spec, mask=mask_spec)
    
    return masked_spectrum, peak_spectrum_masked_dic


def compute_noise_lvl_std(r, masked_spectrum, peak_spectrum_masked_dic):
    '''
    Function to compute noise level and std using DA method
    '''
    mask_spec = masked_spectrum.mask
    
    indexes_v = np.arange(mask_spec.shape[1], dtype=int)
    noise_mask = np.ones(mask_spec.shape, dtype=bool)

    noise_lvl = np.zeros(r.shape[0])
    noise_std = np.zeros(r.shape[0])

    for i_r in np.arange(r.shape[0], dtype=int)[~np.all(mask_spec, axis=1)]:
        # The spectrum at the current range gate
        curr_spec = masked_spectrum[i_r,:]

        unmasked_part = ~curr_spec.mask
        x = curr_spec[unmasked_part]
        idx_array = indexes_v[unmasked_part]

        # Sorting peaks by power
        curr_valid_peaks = np.intersect1d(peak_spectrum_masked_dic[i_r],
                                          indexes_v[~mask_spec[i_r,:]])
        curr_valid_peaks = curr_valid_peaks[np.argsort(masked_spectrum[i_r,:][curr_valid_peaks])]
        curr_ave_pow = np.nanmean(masked_spectrum[i_r,:])

        # Looping over peaks
        curr_mask_sum = np.zeros(idx_array.shape, dtype='bool')
        for peak in curr_valid_peaks:
            mask = idx_array == peak

            old_mean = np.mean(x)
            new_mean = np.mean(np.ma.masked_array(x, mask=mask))

            i = 1
            while np.sum(~mask):
                old_mean = new_mean
                candidates = np.logical_xor(mask, scipy.ndimage.binary_dilation(mask))    
                mask[np.where(candidates)[0][np.argmax(x[candidates])]] = True
                idx = np.where(candidates)[0][np.argmax(x[candidates])]
                new_mean = np.mean(np.ma.masked_array(x, mask=mask))

                i+=1
                if old_mean - new_mean < DA_THRESHOLD:
                    break
            # Summing all masks, to keep counting all peaks
            curr_mask_sum += mask

        # And we assign it to the mask for the whole spectra
        noise_mask[i_r, unmasked_part] = np.logical_not(curr_mask_sum)

        # Current noise array
        noise = np.ma.masked_array(x.data, mask=curr_mask_sum)
        signal = np.ma.masked_array(x.data, mask=~curr_mask_sum)

        # Assigning noise level/std info for that gate
        if (~noise.mask).sum():
            noise_lvl[i_r] = np.nanmean(noise)
            noise_std[i_r] = np.nanstd(noise)
        else:
            noise_lvl[i_r] = np.nanmin(signal)
            
    return noise_mask, noise_lvl, noise_std


def correct_noise_lvl(noise_lvl_raw, standard_noise_lvl, noise_corr_window, max_diff):
    '''
    Function to correct noise level, adjusting anomalous peak to the "median noise" of the dataset.
    '''
    # Defining the box used for smoothing
    kernel = astropy.convolution.Box1DKernel(width=noise_corr_window)

    # Condition for the reasonably clean part of the noise floor
    condition_lvl = np.logical_and(noise_lvl_raw > 0.,
                                   np.abs(np.subtract(noise_lvl_raw, standard_noise_lvl)) < max_diff)

    # Smoothing noise level
    noise_lvl_tmp = np.full(noise_lvl_raw.shape, np.nan)
    noise_lvl_tmp[condition_lvl] = noise_lvl_raw[condition_lvl]
    noise_lvl_tmp[np.logical_not(condition_lvl)] = standard_noise_lvl[np.logical_not(condition_lvl)]

    noise_lvl = astropy.convolution.convolve(noise_lvl_tmp, kernel,
                                             boundary='fill', fill_value=0.,
                                             nan_treatment='interpolate', preserve_nan=True)
    noise_lvl_nans = astropy.convolution.convolve(noise_lvl_tmp, kernel, boundary='fill', fill_value=0.,
                                             nan_treatment='interpolate')
    noise_lvl_nans[np.isnan(noise_lvl_nans)] = 0.

    return noise_lvl, noise_lvl_nans, noise_lvl_tmp


def convert_spectrum_to_reflectivity(raw_spec, noise_lvl, noise_std, d_r, transfer_function,
                                     calibration_constant,  noise_std_factor=0.,
                                     remove_isolated_peals=True):
    '''
    Refining spectrum, by removing noise and converting to spectral reflectivity
    '''
    # 1. Preparation of noise and signal array
    # Adding a buffer on the noise level
    if noise_std_factor > 0.:
        noise_lvl += (noise_std * noise_std_factor)

    # Subtract the noise from the power
    spec_out = raw_spec - noise_lvl[:,None]
    spec_out.mask[spec_out < 0] = True # Noise-removed power (by masking)
    if noise_std_factor > 0.:
        spec_out += (noise_std * noise_std_factor)[:,None]

    if remove_isolated_peals:
        img = (spec_out.mask == False)
        eroded_img = scipy.ndimage.binary_erosion(img)
        label, num_features = scipy.ndimage.label(img)
        if num_features:
            # Looping over all contiguous region of the spectrum
            for i_feat in range(1,num_features+1):
                curr_region = (label == i_feat)
                if not np.sum(eroded_img[curr_region]):
                    spec_out.mask[curr_region] = True

    # Condition on where there is signal (for later noise floor computation)
    ncondi = np.sum(spec_out > 0., axis = 1)

    # 2. Convert to dBZ (Calibrated, attenuated reflectivity spectra)
    # Auxiliary stuff
    N = raw_spec.shape[0]
    m_x3 = raw_spec.shape[1]
    n_square_mat = np.square(np.tile(np.arange(1,N+1), (m_x3, 1)).T)

    # Conversion
    spec_out = np.divide(spec_out * calibration_constant, transfer_function) * n_square_mat * d_r
    noise_lvl = np.divide(noise_lvl * calibration_constant, transfer_function[:,0]) * \
                                                                            n_square_mat[:,0] * d_r
    noise_std = np.divide(noise_std * calibration_constant, transfer_function[:,0]) * \
                                                                            n_square_mat[:,0] * d_r

    # 3. Get Noise floor: the noise level integrated over the area where the signal is.
    noise_floor = int(m_x3 / 3) * noise_lvl

    return spec_out, noise_lvl, noise_std, noise_floor


def compute_spectra_parameters(spec_refined, vel_array, noise_floor, spectrum_varname):
    '''
    Computes the moments of the signal from a Doppler spectrum.

    Since the noise has already been subtracted from the spectrum in input, the function simply
    computes the moments of the input spectrum, together with few additional parameters.
    Moments and parameters are retured in a dictionary, called "params".
    Given the correct name of the spectrum variable (spectrum_varname), the function is able to
    function with both MRR-PRO raw spectra and reflectivity spectra.
    '''

    # Get information about inputs
    n_fft = vel_array.shape[0]
    if spec_refined.shape[1] != n_fft:
        raise ValueError('Dimension mismatch between spec_refined and vel_array')

    # The input spectrum has already the noise subtracted
    power = np.nansum(spec_refined, axis=1)
    if spectrum_varname == 'spectrum_raw':
        z = const_z_calc * power
    else:
        # No need to convert to reflectivity, if it is already reflectivity from the start
        z = power

    # Noise floor also converted
    if spectrum_varname == 'spectrum_raw':
        noise_floor_z = const_z_calc * noise_floor
    else:
        noise_floor_z = noise_floor
    
    # Computing SNR in two ways: power/noise or (power+noise)/noise
    snr = 10 * np.log10(power/noise_floor) # [-]

    # Auxiliary quantities for moments computation
    weights = spec_refined / power[:,None]

    # Moments
    with np.errstate(divide='ignore'): # Ignore divide by zero warnings
        # M1 [m/s]
        m1_dop = np.sum(vel_array * weights, axis=1)
        # M2 [m/s] sigma
        m2_dop = np.sqrt(np.sum(weights * (vel_array - m1_dop[:,None])**2, axis=1))
        # M3 [-] skewness
        m3_dop = ((np.sum(weights * (vel_array - m1_dop[:,None])**3, axis=1))
                        / m2_dop ** 3)
        # M4 [-] kurtosis
        m4_dop = ((np.sum(weights * (vel_array - m1_dop[:,None]**4), axis=1))
                        / m2_dop**4)

    # Create output structure
    params = {'z':z, 'm1_dop':m1_dop, 'm2_dop':m2_dop, 'm3_dop':m3_dop, 'm4_dop':m4_dop,
              'noise_floor_z':noise_floor_z, 'snr':snr}

    return params


def convert_spectrum_parameters_to_dBZ(noise_masked_spectrum, noise_lvl,
                                       spectrum_params):
    '''
    Conversion to dBZ and preparation of an output dictionary for the final netCDF.
    Unsing names as similar as possible as original NetCDF products form Metek software.
    '''
    spectrum_reflectivity = 10. * np.log10(noise_masked_spectrum)
    Zea = 10. * np.log10(spectrum_params['z'])
    VEL = spectrum_params['m1_dop']
    WIDTH = spectrum_params['m2_dop']
    SNR = spectrum_params['snr'] # SNR already in dB
    noise_level = 10. * np.log10(noise_lvl)
    noise_floor = 10. * np.log10(spectrum_params['noise_floor_z'])

    output_dic = {'spectrum_reflectivity':spectrum_reflectivity, 'Zea':Zea, 'VEL':VEL,
                  'WIDTH':WIDTH, 'SNR':SNR, 'noise_level':noise_level, 'noise_floor':noise_floor}

    return output_dic


def process_single_spectrum(spec, v_0_3, r, m, v_ny, d_r, transfer_function_x3, 
                            calibration_constant, standard_noise_lvl):
    '''
    Processing of the spectrum at a single time step
    '''
    # 1. Getting the "raw" peaks
    N = r.shape[0]
    r_idx_peaks, v_idx_peaks, v_l_idx_peaks, v_r_idx_peaks, idx_peaks = find_raw_peaks(spec, N, m,
                                                                                MAX_NUM_PEAKS_AT_R)
    if not len(r_idx_peaks):
        return {}

    if PLOT_RAW_PEAKS:
        # To check raw peaks
        fig, ax = plotting.plot_spectrum(spec, v_0_3, r)
        ax.scatter(v_0_3[v_idx_peaks], r[r_idx_peaks]/1000., marker='x', c='g', s=10.)
        #ax.scatter(v_0_3[v_l_idx_peaks], r[r_idx_peaks]/1000., marker='o', c='r', s=10.)
        #ax.scatter(v_0_3[v_r_idx_peaks], r[r_idx_peaks]/1000., marker='^', c='orange', s=10.)

    # 2. Connecting them in lines
    lines_array, line_v_idx, line_r_idx, line_v, line_r, line_pow_lin, line_min_r, line_max_r, \
    line_median_v, line_median_pow_lin = find_raw_lines(spec, v_0_3, r,
                                                                r_idx_peaks, v_idx_peaks, idx_peaks)
    if not len(line_median_v):
        return {}
    if PLOT_RAW_LINES:
        fig, ax = plotting.plot_spectrum(spec, v_0_3, r)
        for l in lines_array:
            ax.plot(v_0_3[v_idx_peaks[l]], r[r_idx_peaks[l]]/1000., ls=':',
                    marker='.',markersize=10, markeredgecolor='k', alpha=1.)


    # 3. Removing duplicate lines at +/- v_ny
    accepted_lines, accepted_lines_v_idx, \
    accepted_lines_r_idx, accepted_lines_v, \
    accepted_lines_r, accepted_lines_min_r, \
    accepted_lines_max_r, accepted_lines_v_med_array, \
    accepted_lines_pow_lin_max_array = exclude_duplicate_lines(v_ny, lines_array, line_v_idx,
                                                               line_r_idx, line_v, line_r,
                                                               line_pow_lin, line_min_r,
                                                               line_max_r, line_median_v)
    if PLOT_LINES_WITHOUT_DUPLICATES:
        fig, ax = plotting.plot_spectrum(spec, v_0_3, r)
        for i_l, l in enumerate(accepted_lines):
            ax.plot(v_0_3[v_idx_peaks[l]], r[r_idx_peaks[l]]/1000., ls=':',
                    marker='.',markersize=10, markeredgecolor='k', alpha=1.)

    # 4. Removing lines too far from the "main" (higest peak power) one
    accepted_lines_v2, accepted_lines_v_idx_v2, \
    accepted_lines_r_idx_v2, accepted_lines_v_v2, \
    accepted_lines_r_v2 = exclude_lines_far_from_main_one(v_ny, accepted_lines, accepted_lines_v_idx,
                                                          accepted_lines_r_idx, accepted_lines_v, 
                                                          accepted_lines_r, accepted_lines_min_r,
                                                          accepted_lines_max_r,
                                                          accepted_lines_v_med_array,
                                                          accepted_lines_pow_lin_max_array)
    if PLOT_ACCEPTED_LINES:
        fig, ax = plotting.plot_spectrum(spec, v_0_3, r)
        for i_l, l in enumerate(accepted_lines_v2):
            ax.plot(v_0_3[v_idx_peaks[l]], r[r_idx_peaks[l]]/1000., ls=':',
                    marker='.',markersize=10, markeredgecolor='k', alpha=1.)
    # 5. Leaving unmasked only "m" vel. bins at each range gate
    masked_spectrum, peak_spectrum_masked_dic = extract_spectrum_around_peaks(spec, m, r_idx_peaks,
                                                                              v_idx_peaks, 
                                                                              v_l_idx_peaks,
                                                                              v_r_idx_peaks, 
                                                                              accepted_lines_v2)
    if PLOT_SPECTRUM_AROUND_PEAKS:
        fig, ax = plotting.plot_spectrum(masked_spectrum, v_0_3, r)

    # 6. Computing noise lvl and std
    noise_mask, noise_lvl_raw, \
            noise_std = compute_noise_lvl_std(r, masked_spectrum, peak_spectrum_masked_dic)


    # And masking the spectrum where the noise is
    noise_masked_spectrum = np.ma.masked_array(spec, mask=noise_mask)
    # In case we want to smooth the noise level
    if CORRECT_NOISE_LVL:
        noise_lvl, noise_lvl_nans, noise_lvl_tmp = correct_noise_lvl(noise_lvl_raw,
                                                            standard_noise_lvl = standard_noise_lvl,
                                                            noise_corr_window=NOISE_CORR_WINDOW,
                                                            max_diff=MAX_DIFF_NOISE_LVL)
    else:
        noise_lvl = noise_lvl_raw
        noise_lvl_nans = noise_lvl_raw

    if PLOT_NOISE_MASKED_SPECTRUM:
        # Plot 1: the spectrum
        fig, ax = plotting.plot_spectrum(noise_masked_spectrum, v_0_3, r)

        # Plot 2: the noise floor
        fig2, axes2 = plotting.plot_noise_smoothed(r, noise_lvl_raw, noise_lvl, noise_lvl_nans,
                                                   noise_lvl_tmp, standard_noise_lvl, noise_std)

    # 8. Final adjustments to the spectrum and noise:
    # Converting to calibrated, attenuated spectral reflectivity
    noise_masked_spectrum_cal, noise_lvl_cal, noise_std_cal, \
        noise_floor_cal = convert_spectrum_to_reflectivity(noise_masked_spectrum, noise_lvl,
                                                           noise_std, d_r, transfer_function_x3,
                                                           calibration_constant,
                                                           noise_std_factor=NOISE_STD_FACTOR,
                                                           remove_isolated_peals=REMOVE_ISOLATED_PEAK_SPECTRUM)

    # 9. Computing moments of the signal
    spectrum_params = compute_spectra_parameters(noise_masked_spectrum_cal, v_0_3, noise_lvl_cal,
                                                 spectrum_varname=spectrum_varname)

    if PLOT_PRODUCTS_LINEAR:
        fig, ax = plotting.plot_parameters_before_dBZ_conversion(v_0_3, r, spectrum_params)

    '''
    # TO CHECK MASKED SPECTRUM WITH REMOVAL OF NOISE
    plot_spectrum_masked_and_moments(noise_masked_spectrum, v_0_3, r, spectrum_params,
                                     noise_lvl, noise_std)
    '''
    # 10. And converting to dBZ
    spectrum_params_dBZ = convert_spectrum_parameters_to_dBZ(noise_masked_spectrum_cal,
                                                             noise_floor_cal, spectrum_params)

    if PLOT_FINAL_PRODUCTS:
        fig, ax = plotting.plot_spectrum_dBZ(v_0_3, r, spectrum_params_dBZ)

    if ANY_PLOT:
        plt.show()

    return spectrum_params_dBZ


def process_file(in_fpath, border_correction, interference_mask, spectrum_varname, dir_proc_netcdf,
                 smooth_median_spec, out_fname_prefix=None,
                 max_num_spectra_to_process=None, verbose=False):
    '''
    Function to process a single "raw" netCDF MRR file.

    The file is opened, the necessary variables are extracted, some accessory parameters are
    computed, and finally each spectrum is processed separately by calling another function:
    "process_single_spectrum".
    The processed spectrum and its derived variables are then exported to a new netCDF file.
    The export is performed using xarray, by creating a "xarray.dataset" as intermediate step and 
    calling the function "to_netcdf".

    Parameters
    ----------
    in_fpath : str
        Full path to the "raw" netCDF file to process.
    border_correction : numpy array (2D, float or equivalent)
        Numpy array (2D) containing the correction for the values at the min and max velocity limits
        of the spectrum. (dimensions: range, velocity)
    interference_mask : numpy array (2D, bool)
        Numpy array (2D) flagging which sections of the spectrum are most likely to contain
        interference lines. (dimensions: range, velocity)
    spectrum_varname : str
        String containing the name of the spectrum variable inside the "raw" netCDF file.
    dir_proc_netcdf : str
        Full path to the directory in which the processed netCDF files will be saved.
    smooth_median_spec : numpy array (1D, float or equivalent)
        Numpy array (1D) containing the median at each range gate of the median spectrum for the
        campaign. Used for smoothing of the noise floor. (dimension: range)
    out_fname_prefix : str or None
        Prefix to append to the name of processed files. If None, the output files will have the
        same name as the input ones.
    max_num_spectra_to_process : int or None
        Number of spectra in file to process. "None" means "all spectra will be processed".
    verbose : bool
        Flag to specify whether extra information should be printed out during processing.

    '''
    # Loading "raw" netCDF file 
    with nc.Dataset(in_fpath) as ncfile:
        # Basîc quantities
        r = np.array(ncfile.variables['range'])
        t = np.array(ncfile.variables['time'])

        # The raw spectra
        all_spectra_raw = np.array(ncfile.variables[spectrum_varname])
        Zea = np.array(ncfile.variables['Zea'])
        # And what we need to convert it to calibrated spectral reflectivity
        calibration_constant = ncfile.variables['calibration_constant'][0] / CALIB_CONST_FACTOR

        # Transfer function
        if USE_EXTERNAL_TRANSFER_FUNCTION:
            # Note that you should use the external transfer function only if the one in the
            # files has a problem. If you encounter no problem with the transfer function,
            # set USE_EXTERNAL_TRANSFER_FUNCTION and RECONSTRUCT_TRANSFER_FUNCTION to 0
            # in the "config.ini" file. 
            transfer_function_all = np.loadtxt(EXTERNAL_TRANSFER_FUNCTION_PATH, delimiter=',')
            # We need to convert the transfer function to the correct lenght
            tf_shrink = int(transfer_function_all.shape[0] / r.shape[0])
            transfer_function = transfer_function_all[::tf_shrink]
        else:
            transfer_function = np.array(ncfile.variables['transfer_function'])
            # In case ERUO needs to handle the reconstruction (we suggest not to use this function,
            # unless it is impossible to recover the correct transfer function by asking Metek)
            if RECONSTRUCT_TRANSFER_FUNCTION:
                transfer_function = reconstruct_transfer_function(transfer_function)

        # Configurables MRR Parameters:
        num_t = all_spectra_raw.shape[0]      # NUmber of time steps
        N = r.shape[0]                        # Number of range gates
        m = all_spectra_raw.shape[2]          # Number of lines in spectrum
        T_i = np.round(np.median(np.diff(t))) # Time of incoherent averaging
        d_r = np.round(np.median(np.diff(r))) # Range resolution

        # Creating an empty output for files without signal
        empty_var_dic = {}
        out_varnames = ['spectrum_reflectivity', 'Zea', 'VEL', 'WIDTH', 'SNR',
                        'noise_level', 'noise_floor']
        empty_var_dic = {}
        empty_var_dic[out_varnames[0]] = np.full((N, 3 * m), np.nan, dtype='float32')
        for varname in out_varnames[1:]:
            empty_var_dic[varname] = np.full(N, np.nan, dtype='float32')

        if verbose:
            print('----------\nBasic info:')
            print('Shape r: ', r.shape)
            print('Shape t: ', t.shape)
            print('Shape spectrum:', all_spectra_raw.shape)

    # Getting all info
    info_dic = compute_additional_mrr_parameters(N, m, T_i, d_r)
    I = info_dic['I']
    f_ny = info_dic['f_ny']
    v_ny = info_dic['v_ny']
    d_f = info_dic['d_f']
    d_t = info_dic['d_t']
    d_v = info_dic['d_v']
    H = info_dic['H']
    v_0 = info_dic['v_0']

    # Applying the correction for loss of power at the border of spectrum
    border_correction_3d = np.tile(border_correction, (num_t, 1, 1))
    all_spectra_raw += border_correction_3d

    # Spectrum reconstruction
    if RECONSTRUCT_SPECTRUM:
        # Casting the 1D median line to an appropriate 3D matrix
        median_line_tiled = np.moveaxis(np.tile(smooth_median_spec, (num_t, m, 1)), 1, 2)
        # We pass from spectrum to "anomaly" for the reconstruction
        anomaly_3d, reficiendo_3d = reconstruction.define_reficiendo(all_spectra_raw,
                                                                     median_line_tiled,
                                                                     interference_mask)

        all_anomalies = np.full(anomaly_3d.shape, np.nan)
        for i_t in range(num_t):
            all_anomalies[i_t, :, :] = reconstruction.reconstruct_anomaly(anomaly_3d[i_t, :, :],
                                                                          reficiendo_3d[i_t, :, :])
        # Converting back from anomaly to spectrum
        all_spectra = median_line_tiled + all_anomalies
    else:
        # Discouraged: it is possible to process a spectrum without reconstructing the interference
        all_spectra = all_spectra_raw

    # Getting the spectrum corrected and the transfer function repeated three times
    all_spectra_x3_lin, transfer_function_x3 = repeat_spectra(all_spectra, transfer_function)

    # Define velocity repeated three times
    v_0_3 = np.tile(v_0, 3)
    v_0_3[0:m] -= v_ny
    v_0_3[2*m:] += v_ny

    # Depending on the current spectrum variable, we will have to bring the standard noise floor
    # to linear units, otherwise we can leave as it is
    if spectrum_varname == 'spectrum_raw':
        smooth_median_spec_v2 = np.power(10., smooth_median_spec / 10.)
    else:
        smooth_median_spec_v2 = smooth_median_spec

    # Defining maximum number of spectra to process
    if not max_num_spectra_to_process:
        max_num_spectra_to_process = all_spectra_x3_lin.shape[0]
    
    # Processing all spectra
    new_vars_dic = {}
    for i_t in range(max_num_spectra_to_process):
        # Bulk of the processing: removing noise and computing moments
        output_dic = process_single_spectrum(all_spectra_x3_lin[i_t, :, :], v_0_3, r, m, v_ny,
                                             d_r, transfer_function_x3, calibration_constant,
                                             smooth_median_spec_v2)

        # If the processing did not identify any signal, we output matrixes full of "NaN"
        if not len(output_dic.keys()):
            output_dic = empty_var_dic
        # Adding the variables of the current spectrum to the dictionary
        for k in output_dic.keys():
            if k in new_vars_dic.keys():
                new_vars_dic[k].append(output_dic[k])
            else:
                new_vars_dic[k] = [output_dic[k]]
    # Concatenating variables (list --> array)
    concatenated_vars_dic = {}
    for k in new_vars_dic.keys():
        # Note: concatenation as masked array (not simply np.stack)
        concatenated_vars_dic[k] = np.ma.stack(new_vars_dic[k])
        # Unfortunately, we also have to put nans (for saving to file).
        # This aspect could be improved in the future (keeping values below mask).
        concatenated_vars_dic[k][concatenated_vars_dic[k].mask] = np.nan

    # Adding to the concatenated vars the information on specturm reconstruction
    if RECONSTRUCT_SPECTRUM:
        concatenated_vars_dic['flag_spectrum_reconstruction'] = reficiendo_3d
    else:
        concatenated_vars_dic['flag_spectrum_reconstruction'] = np.zeros(all_spectra.shape, dtype='bool')

    # ===================
    # Exporting to netCDF
    # ===================
    # 1. Read initial attributes and coordinates
    with xr.open_dataset(in_fpath) as ds_ini:
        coords = ds_ini.coords
        attrs = ds_ini.attrs
        raw_spectrum_units = ds_ini.variables[spectrum_varname].attrs['units']

    # 2. Add the new information
    attrs['field_names'] = ','.join(concatenated_vars_dic.keys())
    attrs['title'] = attrs['title'] + ' - Re-processed with ERUO'
    attrs['history'] = 'Re-processed with ERUO on %s UTC' % \
                        datetime.datetime.utcnow().strftime('%d/%m/%Y %H:%M:%S')
    
    # 3. Create the new dataset
    units_dataset = {'spectrum_reflectivity': 'dBZ/bin',
                     'Zea': 'dBZ',
                     'VEL': 'm/s',
                     'WIDTH': 'm/s',
                     'SNR': 'dB',
                     'noise_level': raw_spectrum_units,
                     'noise_floor': 'dBZ',
                     'flag_spectrum_reconstruction': '-'}
    if max_num_spectra_to_process is None:
        data_vars = {}
        for k in concatenated_vars_dic.keys():
            if 'spectrum' in k:
                if 'reconstruction' in k:
                    data_vars[k] = (['time', 'range', 'spectrum_n_samples'],
                                    concatenated_vars_dic[k])
                else:
                    data_vars[k] = (['time', 'range', 'spectrum_n_samples_extended'],
                                    concatenated_vars_dic[k])
            else:
                data_vars[k] = (['time', 'range'], concatenated_vars_dic[k])
    else:
        # If only a subset of spectra is processed (e.g. debugging), we have to modify the dataset
        # creation involving some temporary containers
        data_vars = {}
        for k in concatenated_vars_dic.keys():
            if 'spectrum' in k:
                tmp = np.full([coords['time'].shape[0], 
                               coords['range'].shape[0],
                               concatenated_vars_dic[k].shape[2]], np.nan)
                tmp[:max_num_spectra_to_process, :, :] = \
                                        concatenated_vars_dic[k][:max_num_spectra_to_process, :, :]
                if 'reconstruction' in k:
                    data_vars[k] = (['time', 'range', 'spectrum_n_samples'],
                                    tmp)
                else:
                    data_vars[k] = (['time', 'range', 'spectrum_n_samples_extended'],
                                    tmp)
            else:
                tmp = np.full([coords['time'].shape[0], coords['range'].shape[0]], np.nan)
                tmp[:max_num_spectra_to_process, :] = concatenated_vars_dic[k]
                data_vars[k] = (['time', 'range'], tmp)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # 4. Adding units to variables
    for k in concatenated_vars_dic.keys():
        ds.variables[k].attrs['units'] = units_dataset[k]

    # 5. Saving to file
    # Defining output filepath
    out_fname = out_fname_prefix + os.path.basename(in_fpath)

    out_fdir = os.path.join(dir_proc_netcdf, os.sep.join(in_fpath.split(os.sep)[-3:-1]))
    if not os.path.exists(out_fdir):
        os.makedirs(out_fdir)

    out_fpath = os.path.join(out_fdir, out_fname)

    # Exporting to netCDF4
    ds.to_netcdf(out_fpath, mode='w', unlimited_dims='time')

