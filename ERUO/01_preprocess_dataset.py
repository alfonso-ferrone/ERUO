'''
01_preprocess_dataset.py 
Script to perform the ERUO preprocessing over a dataset of MRR-PRO measurements.

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
import numpy as np
from configparser import ConfigParser
from algo import preprocessing

# --------------------------------------------------------------------------------------------------
# PLOTTING PARAMETERS
'''
To display the results of the pre-processing (interference mask and border correction), just set the
following two flags to "True".
Parameters such as figure size, DPI, and font size can be changed in the "rc parameters" as shown
below.
'''
PLOT_FINAL_PRODUCTS = False
PLOT_SECONDARY_PRODUCTS = False

if PLOT_FINAL_PRODUCTS or PLOT_SECONDARY_PRODUCTS:
    # matplotlib necessary only if the debugging plots are enabled 
    import matplotlib.pyplot as plt
    # Figure parameters
    plt.rcParams.update({'figure.dpi': 100})
    plt.rc('font', size=11)
    px = 1/plt.rcParams['figure.dpi']

    figsize_2panelS = (1000*px, 500*px)

    spectrum_cmap = 'inferno'
    mask_cmap = 'RdYlGn_r'
# --------------------------------------------------------------------------------------------------


def main():
    '''
    First step of the processing of a dataset.
    
    Creation of:
    - interference_mask,
    - border_corr,
    - corrected_reconstructed_median_line,
    - alternative_interf_mask.
    These quantities will be saved in the directory indicated in config.ini.

    Note:
    All files in the dataset will be loaded to compute the necessary statistics.
    This may require a lot of time, depending on the dataset and the computer used.
    In case of crashes or bugs, try to limit the dataset, saving only clear-sky files (ideally from
    different dates) in a different directory and providing that directory as "dir_input_netcdf" in
    the "config.ini" file.
    In case, don't forget to re-set "dir_input_netcdf" to its original value after the preprocessing!
    '''
    #Reading configuration
    config_fpath = "config.ini"
    with open(config_fpath) as fp:
        config_object = ConfigParser()
        config_object.read_file(fp)

    # Directories
    path_info = config_object['PATHS']
    dir_input_netcdf = path_info['dir_input_netcdf']
    dir_npy_out = path_info['dir_npy']
    fname_interference_mask = path_info['fname_interference_mask']
    fname_border_correction = path_info['fname_border_correction']
    fname_reconstructed_median = path_info['fname_reconstructed_median']
    fname_interf_isolated_peaks = path_info['fname_interf_isolated_peaks']

    # Wheter already computed quantities should be recomputed
    reprocessing_info = config_object['REPROCESSING_INFO']
    REGENERATE_QUANTILE_ARCHIVE = bool(int(reprocessing_info['REGENERATE_QUANTILE_ARCHIVE']))
    REGENERATE_PREPROCESSING_PRODUCTS = \
                                   bool(int(reprocessing_info['REGENERATE_PREPROCESSING_PRODUCTS']))

    # User spectified processing parameters
    limitation_info = config_object['LIMITATION']
    MAX_NUM_FILES_TO_PREPROCESS = limitation_info['MAX_NUM_FILES_TO_PREPROCESS'].upper()
    if 'ALL' in MAX_NUM_FILES_TO_PREPROCESS:
        MAX_NUM_FILES_TO_PREPROCESS = None
    else:
        MAX_NUM_FILES_TO_PREPROCESS = int(MAX_NUM_FILES_TO_PREPROCESS)

    # Name of variables in input NetCDF
    var_names_info = config_object['INPUT_NECDF_VARIABLE_NAMES']
    spectrum_varname = var_names_info['SPECTRUM_VARNAME']

    # Interference removal and border correction
    preprocessing_info = config_object['PREPROCESSING_PARAMETERS']
    QUANTILE_PREPROCESSING = float(preprocessing_info['QUANTILE_PREPROCESSING'])
    MAX_GRADIENT_MULTIPLIER_INTER_FIT = float(preprocessing_info['MAX_GRADIENT_MULTIPLIER_INTER_FIT'])
    CHOSEN_DEGREE_FIT_INTER_FIT = int(preprocessing_info['CHOSEN_DEGREE_FIT_INTER_FIT'])
    MIN_LEN_SLICES_INTERF_FIT = int(preprocessing_info['MIN_LEN_SLICES_INTERF_FIT'])
    PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM = \
                           float(preprocessing_info['PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM'])
    MAX_FRACTION_OF_NAN_AT_RANGE = float(preprocessing_info['MAX_FRACTION_OF_NAN_AT_RANGE'])
    NUM_ITERATIONS_INTERF_MASK_DILATION = int(preprocessing_info['NUM_ITERATIONS_INTERF_MASK_DILATION'])
    MARGIN_L_BORD_CORR = int(preprocessing_info['MARGIN_L_BORD_CORR'])
    MARGIN_R_BORD_CORR = int(preprocessing_info['MARGIN_R_BORD_CORR'])

    # Debugging parameters
    debugging_info = config_object['DEBUGGING_PARAMETERS']
    VERBOSE = bool(int(debugging_info['VERBOSE']))
    IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
    if IGNORE_WARNINGS:
        import warnings
        warnings.filterwarnings("ignore")


    if VERBOSE:
        print('---------------------------------------------------------------------------------')
        print('LOADED CONFIGURATION')

        print('\nPATHS')
        print('dir_input_netcdf: ', dir_input_netcdf)
        print('dir_npy_out: ', dir_npy_out)
        print('fname_interference_mask: ', fname_interference_mask)
        print('fname_border_correction: ', fname_border_correction)
        print('fname_reconstructed_median: ', fname_reconstructed_median)
        print('fname_interf_isolated_peaks: ', fname_interf_isolated_peaks)

        print('\nREPROCESSING_INFO')
        print('REGENERATE_QUANTILE_ARCHIVE: ', REGENERATE_QUANTILE_ARCHIVE, 
              '(type: ', type(REGENERATE_QUANTILE_ARCHIVE), ')')
        print('REGENERATE_PREPROCESSING_PRODUCTS: ', REGENERATE_PREPROCESSING_PRODUCTS,
              '(type: ', type(REGENERATE_PREPROCESSING_PRODUCTS), ')')


        print('\nLIMITATION_INFO')
        print('MAX_NUM_FILES_TO_PREPROCESS: ', MAX_NUM_FILES_TO_PREPROCESS, ' (None means "all")')

        print('\nINPUT_NECDF_VARIABLE_NAMES')
        print('spectrum_varname: ', spectrum_varname)

        print('\nPREPROCESSING_PARAMETERS')
        print('QUANTILE_PREPROCESSING: ', QUANTILE_PREPROCESSING)
        print('MAX_GRADIENT_MULTIPLIER_INTER_FIT: ', MAX_GRADIENT_MULTIPLIER_INTER_FIT)
        print('CHOSEN_DEGREE_FIT_INTER_FIT: ', CHOSEN_DEGREE_FIT_INTER_FIT)
        print('MIN_LEN_SLICES_INTERF_FIT: ', MIN_LEN_SLICES_INTERF_FIT)
        print('PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM: ',
                                                    PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM)
        print('MAX_FRACTION_OF_NAN_AT_RANGE: ', MAX_FRACTION_OF_NAN_AT_RANGE)
        print('MARGIN_L_BORD_CORR: ', MARGIN_L_BORD_CORR)
        print('MARGIN_R_BORD_CORR: ', MARGIN_R_BORD_CORR)

        print('\nDEBUGGING_PARAMETERS')
        print('VERBOSE: ', VERBOSE, '(type: ', type(VERBOSE), ')')
        print('IGNORE_WARNINGS: ', IGNORE_WARNINGS, '(type: ', type(IGNORE_WARNINGS), ')')
        print('---------------------------------------------------------------------------------\n')

    # ==============================================================================================
    # 1) Archives containing some quantiles of the spectra distribution in time
    # ==============================================================================================
    # Defining output file name
    quantiles_out_fname = 'all_spectra_quantile.npy'
    quantiles_out_fpath = os.path.join(dir_npy_out, quantiles_out_fname)

    # Checking if file already exists
    if (not os.path.isfile(quantiles_out_fpath)) or REGENERATE_QUANTILE_ARCHIVE:
        # Finding files to process
        all_files = preprocessing.load_dataset(dir_input_netcdf, verbose=VERBOSE)
        if not len(all_files):
            print('No files found, quitting preprocessing.')
            return 1
        # Loading the spectra and concatenating them on the time axis
        concatenated_spectra = preprocessing.concatenate_all_spectra(all_files,
                                                           spectrum_varname=spectrum_varname,
                                                           verbose=VERBOSE,
                                                           num_profiles=MAX_NUM_FILES_TO_PREPROCESS)
        # Saving quantiles for further processing

        preprocessing.save_dataset_stats(concatenated_spectra, quantiles_out_fpath,
                                         q=QUANTILE_PREPROCESSING)
    elif VERBOSE:
        print('Quantile archive already exiting at:\n%s' % quantiles_out_fpath)
        print('It will not be recomputed.\n')


    # ==============================================================================================
    # 2) Removal of interference line and border correction
    # ==============================================================================================
     # Defining output file names
    interf_mask_out_fpath = os.path.join(dir_npy_out, fname_interference_mask)
    border_corr_out_fpath = os.path.join(dir_npy_out, fname_border_correction)
    reconstructed_median_out_fpath = os.path.join(dir_npy_out, fname_reconstructed_median)
    interf_isolated_peaks_out_fpath = os.path.join(dir_npy_out, fname_interf_isolated_peaks)
    # Checking if file already exists
    if (not os.path.isfile(interf_mask_out_fpath)) or REGENERATE_PREPROCESSING_PRODUCTS:
        # The function returns additional quantities not used in the processing.
        # They can be used as alternative version of quantitiy normally included in the library.
        interference_mask, border_corr, corrected_reconstructed_median_line, \
            alternative_interf_mask = preprocessing.interference_mask_and_border_correction(\
                                        quantiles_out_fpath,
                                        interf_mask_out_fpath, border_corr_out_fpath,
                                        reconstructed_median_out_fpath,
                                        interf_isolated_peaks_out_fpath,
                                        max_gradient_multiplier=MAX_GRADIENT_MULTIPLIER_INTER_FIT,
                                        chosen_degree_fit=CHOSEN_DEGREE_FIT_INTER_FIT,
                                        min_len_slice=MIN_LEN_SLICES_INTERF_FIT,
                                        threhsold_prominence_interference=\
                                        PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM,
                                        max_fraction_of_nan_at_range=MAX_FRACTION_OF_NAN_AT_RANGE,
                                        num_iterations_interf_mask_dilation=\
                                        NUM_ITERATIONS_INTERF_MASK_DILATION,
                                        margin_l_bord_corr=MARGIN_L_BORD_CORR,
                                        margin_r_bord_corr=MARGIN_R_BORD_CORR)

        if PLOT_FINAL_PRODUCTS:
            m = interference_mask.shape[1]
            fig, axes = plt.subplots(1, 2, figsize=figsize_2panelS)

            ax1 = axes[0]
            ax2 = axes[1]
            mappable1 = ax1.pcolormesh(np.arange(m), np.arange(interference_mask.shape[0]), 
                                       interference_mask, cmap=mask_cmap, vmin=0, vmax=1)
            plt.sca(ax1)
            plt.colorbar(mappable1)
            ax1.set_title('Interference mask')


            mappable2 = ax2.pcolormesh(np.arange(m), np.arange(border_corr.shape[0]), 
                                       border_corr, cmap=spectrum_cmap)
            plt.sca(ax2)
            plt.colorbar(mappable2)
            ax2.set_title('Border correction')

            for ax in axes.flatten():
                ax.set_facecolor('gray')
                ax.set_xlabel('Velocity bin index')
                ax.set_ylabel('Range gate index')

            plt.tight_layout()
            plt.savefig(os.path.join(dir_npy_out, 'preprocessing_results.png'))

        if PLOT_SECONDARY_PRODUCTS:
            m = interference_mask.shape[1]
            fig, axes = plt.subplots(1, 2, figsize=figsize_2panelS)

            ax1 = axes[0]
            ax2 = axes[1]
            mappable1 = ax1.pcolormesh(np.arange(m), np.arange(alternative_interf_mask.shape[0]), 
                                       alternative_interf_mask, cmap=mask_cmap, vmin=0, vmax=1)
            plt.sca(ax1)
            plt.colorbar(mappable1)
            ax1.set_title('Interference mask - alternative')

            ax1.set_facecolor('gray')
            ax1.set_xlabel('Velocity bin index')
            ax1.set_ylabel('Range gate index')

            ax2.grid(ls=':', c='gray', alpha=0.5)
            ax2.plot(corrected_reconstructed_median_line, np.arange(border_corr.shape[0]),
                     marker='.', ls='-', lw=0.8, markersize=2.)

            ax2.set_title('Reconstructed median profile')

            ax2.set_xlabel('Median spectrum [spectrum units]')
            ax2.set_ylabel('Range gate index')

            plt.tight_layout()
            plt.savefig(os.path.join(dir_npy_out, 'preprocessing_secondary_results.png'))

    elif VERBOSE:
        print('Interference mask already exiting at:\n%s' % interf_mask_out_fpath)
        print('It will not be recomputed.\n')


if __name__ == '__main__':
    main()