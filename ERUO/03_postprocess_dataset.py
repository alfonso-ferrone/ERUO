'''
03_postprocess_dataset.py 
Script to perform the ERUO postprocessing over a dataset of MRR-PRO measurements.

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
import netCDF4 as nc
import scipy.ndimage
from configparser import ConfigParser
from joblib import Parallel, delayed
from algo import preprocessing, postprocessing, plotting


def main():
    '''
    Step 3 of the processing of a dataset.

    Postprocessing of  netCDF files generated in step 2.
    Depending on user choices, the following actions can be performed:
    - removal of leftover interference lines
    - removal of random isolated noise
    '''
	#Reading configuration
    config_fpath = "config.ini"
    with open(config_fpath) as fp:
        config_object = ConfigParser()
        config_object.read_file(fp)

    # Directories
    path_info = config_object['PATHS']
    dir_proc_netcdf = path_info['dir_proc_netcdf']
    dir_postproc_netcdf = path_info['dir_postproc_netcdf']
    out_postprocessed_suffix = path_info['OUT_POSTPROCESSED_SUFFIX']

    # Debugging parameters
    debugging_info = config_object['DEBUGGING_PARAMETERS']
    VERBOSE = bool(int(debugging_info['VERBOSE']))
    PRINT_EVERY_N = int(debugging_info['PRINT_EVERY_N'])
    IGNORE_WARNINGS = bool(int(debugging_info['IGNORE_WARNINGS']))
    if IGNORE_WARNINGS:
        import warnings
        warnings.filterwarnings("ignore")

    if VERBOSE:
        # Loading postprocessing parameters only if we need to diplay them
        # (Otherwise, they are loaded directly in the "postprocessing" script)
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

        print('\nPATHS')
        print('dir_proc_netcdf: ', dir_proc_netcdf)
        print('dir_postproc_netcdf: ', dir_proc_netcdf)
        print('out_postprocessed_suffix: ', out_postprocessed_suffix)

        print('\nPOSTPROCESSING_PARAMETERS')
        print('MIN_SNR_POSTPROC: ', MIN_SNR_POSTPROC)
        print('REMOVE_INTERF_POSTPROC: ', REMOVE_INTERF_POSTPROC,
              ' (type ', type(REMOVE_INTERF_POSTPROC), ')')
        print('MIN_TIME_FRACTION_INTERF_POSTPROC: ', MIN_TIME_FRACTION_INTERF_POSTPROC)
        print('WINDOW_POSTPROCESS_T: ', WINDOW_POSTPROCESS_T)
        print('WINDOW_POSTPROCESS_R: ', WINDOW_POSTPROCESS_R)
        print('MIN_HALF_FRACTION: ', MIN_HALF_FRACTION)
        print('MIN_RATIO_H_V: ', MIN_RATIO_H_V)
        print('MIN_INTERF_FLAG: ', MIN_INTERF_FLAG)
        print('INTEREF_KERNEL_STD: ', INTEREF_KERNEL_STD)
        print('REMOVE_NOISE_POSTPROC: ', REMOVE_NOISE_POSTPROC,
              ' (type ', type(REMOVE_NOISE_POSTPROC), ')')
        print('MIN_SLICE_LENGHT_NOISE_REMOVAL: ', MIN_SLICE_LENGHT_NOISE_REMOVAL)
        print('MIN_NUM_PIXEL_NOISE_REMOVAL: ', MIN_NUM_PIXEL_NOISE_REMOVAL)

        print('\nDEBUGGING_PARAMETERS')
        print('VERBOSE: ', VERBOSE, '(type: ', type(VERBOSE), ')')
        print('IGNORE_WARNINGS: ', IGNORE_WARNINGS, '(type: ', type(IGNORE_WARNINGS), ')')
        print('---------------------------------------------------------------------------------\n')


    # Finding files to postprocess
    all_files = preprocessing.load_dataset(dir_proc_netcdf, verbose=VERBOSE)

    # Debugging help:
    # - In case you want to skip some files at the beginning (example: already processed ones)
    # all_files = all_files[304:]
    # - In case you want to process a specific file knowing its date/time
    # all_files = [f for f in all_files if '20191223_180000' in f]
    # all_files = [f for f in all_files if '20191214_150000' in f]

    for i_f, curr_fpath in enumerate(all_files):
        if VERBOSE and not (i_f % PRINT_EVERY_N):
            print('%d/%d: %s' % (i_f, len(all_files), os.path.basename(curr_fpath)))
        postprocessing.postprocess_file(curr_fpath, dir_postproc_netcdf, out_postprocessed_suffix)
  

if __name__ == '__main__':
    main()
