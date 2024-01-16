'''
02_process_dataset.py 
Script to perform the ERUO processing over a dataset of MRR-PRO measurements.

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
from joblib import Parallel, delayed
from algo import preprocessing, processing


def main():
    '''
    Second step of the processing of a dataset.

    Processing of the "raw" netCDF files , obtained directly from the MRR-PRO.
    Processed netCDF files will be created in the directory indicated in the config.ini file. 
    '''
    #Reading configuration
    config_fpath = "config.ini"
    with open(config_fpath) as fp:
        config_object = ConfigParser()
        config_object.read_file(fp)

    # Directories
    path_info = config_object['PATHS']
    dir_input_netcdf = path_info['dir_input_netcdf']
    dir_npy = path_info['dir_npy']
    dir_proc_netcdf = path_info['dir_proc_netcdf']
    fname_border_correction = path_info['fname_border_correction']
    fname_interference_mask = path_info['fname_interference_mask']
    fname_reconstructed_median = path_info['fname_reconstructed_median']
    out_fname_prefix = path_info['OUT_FNAME_PREFIX']
    
    # User spectified processing parameters
    limitation_info = config_object['LIMITATION']
    MAX_NUM_FILES_TO_PROCESS = limitation_info['MAX_NUM_FILES_TO_PROCESS'].upper()
    if 'ALL' in MAX_NUM_FILES_TO_PROCESS:
        MAX_NUM_FILES_TO_PROCESS = None
    else:
        MAX_NUM_FILES_TO_PROCESS = int(MAX_NUM_FILES_TO_PROCESS)

    MAX_NUM_SPECTRA_TO_PROCESS = limitation_info['MAX_NUM_SPECTRA_TO_PROCESS'].upper()
    if 'ALL' in MAX_NUM_SPECTRA_TO_PROCESS:
        MAX_NUM_SPECTRA_TO_PROCESS = None
    else:
        MAX_NUM_SPECTRA_TO_PROCESS = int(MAX_NUM_SPECTRA_TO_PROCESS)
    
    PARALLELIZATION = bool(int(limitation_info['PARALLELIZATION']))
    NUM_JOBS = int(limitation_info['NUM_JOBS'])

    # Name of variables in input NetCDF
    var_names_info = config_object['INPUT_NECDF_VARIABLE_NAMES']
    spectrum_varname = var_names_info['SPECTRUM_VARNAME']

    # Debugging parameters
    debugging_info = config_object['DEBUGGING_PARAMETERS']
    VERBOSE = bool(int(debugging_info['VERBOSE']))
    PRINT_EVERY_N = int(debugging_info['PRINT_EVERY_N'])
    IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
    if IGNORE_WARNINGS:
        import warnings
        warnings.filterwarnings("ignore")

    if VERBOSE:
        # Loading also the parameters for the processing and postprocessing
        # They are not directly used in the code, but it's better to print them here than in the
        # "processing" library, so we can display their value at the beginning of the analysis
        # rather than at every iteration.

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
        RECONSTRUCT_SPECTRUM = bool(int(spectrum_reconstruction_parameters['RECONSTRUCT_SPECTRUM']))
        MIN_WIN_RECONSTRUCTION = spectrum_reconstruction_parameters['MIN_WIN_RECONSTRUCTION']
        KERNEL_SCALE_FACTOR = spectrum_reconstruction_parameters['KERNEL_SCALE_FACTOR']
        NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION = \
                    spectrum_reconstruction_parameters['NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION']
        MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED = \
                    spectrum_reconstruction_parameters['MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED']
        # Processing parameters:
        processing_parameters_info = config_object['SPECTRUM_PROCESSING_PARAMETERS']
        # Peak identification
        if spectrum_varname == 'spectrum_reflectivity':
            PROMINENCE_THRESHOLD = float(processing_parameters_info['PROMINENCE_THRESHOLD_REFLECTIVITY'])
        else:
            PROMINENCE_THRESHOLD = float(processing_parameters_info['PROMINENCE_THRESHOLD_RAW_SPECTRUM'])
        RELATIVE_PROMINENCE_THRESHOLD = float(processing_parameters_info['RELATIVE_PROMINENCE_THRESHOLD'])
        # Connecting peaks in lines
        WINDOW_R = float(processing_parameters_info['WINDOW_R'])
        WINDOW_V = float(processing_parameters_info['WINDOW_V'])
        MIN_NUM_PEAKS_IN_LINE = float(processing_parameters_info['MIN_NUM_PEAKS_IN_LINE'])
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

        print('---------------------------------------------------------------------------------')
        print('LOADED CONFIGURATION')

        print('\nPATHS')
        print('dir_input_netcdf: ', dir_input_netcdf)
        print('dir_npy: ', dir_npy)
        print('dir_proc_netcdf: ', dir_proc_netcdf)
        print('fname_border_correction: ', fname_border_correction)
        print('fname_interference_mask: ', fname_interference_mask)
        print('fname_reconstructed_median: ', fname_reconstructed_median)
        print('out_fname_prefix: ', out_fname_prefix)

        print('\nLIMITATIONS')
        print('MAX_NUM_FILES_TO_PROCESS: ', MAX_NUM_FILES_TO_PROCESS)
        print('MAX_NUM_SPECTRA_TO_PROCESS: ', MAX_NUM_SPECTRA_TO_PROCESS)
        print('PARALLELIZATION: ', PARALLELIZATION, '(type: ', type(PARALLELIZATION), ')')
        print('NUM_JOBS: ', NUM_JOBS)

        print('\nINPUT_NECDF_VARIABLE_NAMES')
        print('spectrum_varname: ', spectrum_varname)

        print('\nTRANSFER_FUNCTION_PARAMETERS')
        print('USE_EXTERNAL_TRANSFER_FUNCTION: ', USE_EXTERNAL_TRANSFER_FUNCTION,
              ' (type: ', type(USE_EXTERNAL_TRANSFER_FUNCTION), ')')
        print('EXTERNAL_TRANSFER_FUNCTION_PATH: ', EXTERNAL_TRANSFER_FUNCTION_PATH)
        print('RECONSTRUCT_TRANSFER_FUNCTION: ', RECONSTRUCT_TRANSFER_FUNCTION,
              ' (type: ', type(RECONSTRUCT_TRANSFER_FUNCTION), ')')

        print('\nSPECTRUM_RECONSTRUCTION_PARAMETERS')
        print('RECONSTRUCT_SPECTRUM: ',RECONSTRUCT_SPECTRUM,
              ' (type: ', type(RECONSTRUCT_SPECTRUM), ')')
        print('MARGIN_SMALL_INTERF_DETECTION: ', MARGIN_SMALL_INTERF_DETECTION)
        print('MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF: ', MAX_NUM_PEAKS_IN_MARGIN_SMALL_INTERF)
        print('FRACTION_VEL_LINE_INTERFERENCE: ', FRACTION_VEL_LINE_INTERFERENCE)
        print('ADIACIENTIA_WIDTH: ', ADIACIENTIA_WIDTH)
        print('EXCEPTIONAL_ANOMALY_THREHSOLD: ', EXCEPTIONAL_ANOMALY_THREHSOLD)
        print('HORIZONTAL_TOL: ', HORIZONTAL_TOL)
        print('MIN_WIN_RECONSTRUCTION: ', MIN_WIN_RECONSTRUCTION)
        print('KERNEL_SCALE_FACTOR: ', KERNEL_SCALE_FACTOR)
        print('NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION: ',
              NUM_BOTTOM_GATES_TO_SKIP_IN_RECONSTRUCTION)
        print('MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED: ', MIN_PROMINENCE_THRESHOLD_RECONSTRUCTED)

        print('\nSPECTRUM_PROCESSING_PARAMETERS')
        print('PROMINENCE_THRESHOLD: ', PROMINENCE_THRESHOLD)
        print('RELATIVE_PROMINENCE_THRESHOLD: ', RELATIVE_PROMINENCE_THRESHOLD)
        print('WINDOW_R: ', WINDOW_R)
        print('WINDOW_V: ', WINDOW_V)
        print('MIN_NUM_PEAKS_IN_LINE: ', MIN_NUM_PEAKS_IN_LINE)
        print('VEL_TOL: ', VEL_TOL)
        print('DA_THRESHOLD: ', DA_THRESHOLD)
        print('NOISE_STD_FACTOR: ', NOISE_STD_FACTOR)
        print('CORRECT_NOISE_LVL: ', CORRECT_NOISE_LVL, ' (type: ', type(CORRECT_NOISE_LVL), ')')
        print('NOISE_CORR_WINDOW: ', NOISE_CORR_WINDOW)
        print('CALIB_CONST_FACTOR: ', CALIB_CONST_FACTOR)
        print('MAX_DIFF_NOISE_LVL: ', MAX_DIFF_NOISE_LVL)

        print('\nDEBUGGING_PARAMETERS')
        print('VERBOSE: ', VERBOSE, '(type: ', type(VERBOSE), ')')
        print('IGNORE_WARNINGS: ', IGNORE_WARNINGS, '(type: ', type(IGNORE_WARNINGS), ')')
        print('---------------------------------------------------------------------------------\n')

    # Finding files to process
    all_files = preprocessing.load_dataset(dir_input_netcdf, verbose=VERBOSE)

    # Debugging help:
    # - In case you want to skip some files at the beginning (example: already processed ones)
    # all_files = all_files[304:]
    # - In case you want to process a specific file knowing its date/time
    # all_files = [f for f in all_files if '20210112_160000' in f]

    # Limiting the number (if required by user in config.ini)
    all_files = all_files[:MAX_NUM_FILES_TO_PROCESS]
    if VERBOSE:
        if MAX_NUM_FILES_TO_PROCESS is None:
            print('Processing all files.')
        else:
            print('Processing only %d files.' % MAX_NUM_FILES_TO_PROCESS)

    # Loading preprocessing products
    # Correction near [0, v_ny] m/s
    border_corr_fpath = os.path.join(dir_npy, fname_border_correction)
    border_corr = np.load(border_corr_fpath)
    # Interference mask
    interf_mask_fpath = os.path.join(dir_npy, fname_interference_mask)
    interf_mask = np.load(interf_mask_fpath)
    # Reconstructed median profile
    reconstructed_median_out_fpath = os.path.join(dir_npy, fname_reconstructed_median)
    reconstructed_median = np.load(reconstructed_median_out_fpath)

    if PARALLELIZATION:
        # Defining a function with only 1 input
        def process_file_par(in_fpath):
            return processing.process_file(in_fpath, border_corr, interf_mask, 
                                           spectrum_varname, dir_proc_netcdf,
                                           smooth_median_spec=reconstructed_median,
                                           out_fname_prefix=out_fname_prefix,
                                           max_num_spectra_to_process=MAX_NUM_SPECTRA_TO_PROCESS,
                                           verbose=False)

        # Using joblib for parallelization (can be further customized if needed)
        tmp = Parallel(n_jobs=NUM_JOBS)(delayed(process_file_par)(in_fpath) for in_fpath in all_files)
    else:
        for i_f, in_fpath in enumerate(all_files):
            if VERBOSE and not (i_f % 1):
                print('%d/%d: %s' % (i_f, len(all_files), os.path.basename(in_fpath)))
                
            # Launching the processing of a single file (which in turn contains several spectra)
            processing.process_file(in_fpath, border_corr, interf_mask,
                                    spectrum_varname, dir_proc_netcdf,
                                    smooth_median_spec=reconstructed_median,
                                    out_fname_prefix=out_fname_prefix,
                                    max_num_spectra_to_process=MAX_NUM_SPECTRA_TO_PROCESS,
                                    verbose=False)


if __name__ == '__main__':
    main()
