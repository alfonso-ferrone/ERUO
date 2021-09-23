'''
04_generate_quickplots.py 
Script to save the plots of Zea and VEL for the ERUO processed and postprocessed products.

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
import netCDF4 as nc
from configparser import ConfigParser
from algo import plotting, preprocessing


QUICKPLOT_PROCESSED = 1
QUICKPLOT_POSTPROCESSED = 1


def main():
    '''
    Step 4 of the processing of a dataset (optional).

    Produce quickplots of processed and postprocessed files.
    Each plot has 2 panels: one for Zea, one for VEL.
    '''
    # Reading configuration
    config_fpath = "config.ini"
    with open(config_fpath) as fp:
        config_object = ConfigParser()
        config_object.read_file(fp)

    # Directories
    path_info = config_object['PATHS']
    dir_input_netcdf = path_info['dir_input_netcdf']
    dir_proc_netcdf = path_info['dir_proc_netcdf']
    dir_postproc_netcdf = path_info['dir_postproc_netcdf']
    dir_quickplots = path_info['dir_quickplots']
    dir_quickplots_postproc = path_info['dir_quickplots_postproc']

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
        print('dir_proc_netcdf: ', dir_proc_netcdf)
        print('dir_postproc_netcdf: ', dir_postproc_netcdf)
        print('dir_quickplots: ', dir_quickplots)
        print('dir_quickplots_postproc: ', dir_quickplots_postproc)

        print('\nDEBUGGING_PARAMETERS')
        print('VERBOSE: ', VERBOSE, ' (type: ', type(VERBOSE), ')')
        print('IGNORE_WARNINGS: ', IGNORE_WARNINGS, '(type: ', type(IGNORE_WARNINGS), ')')
        print('---------------------------------------------------------------------------------\n')

    if QUICKPLOT_PROCESSED:
        # Finding files to process
        all_files_proc = preprocessing.load_dataset(dir_proc_netcdf, verbose=VERBOSE)

        for in_fpath in all_files_proc:
            out_quickplot_fname = os.path.basename(in_fpath).split('.')[0] + '.png'
            out_quickplot_fpath = os.path.join(dir_quickplots, out_quickplot_fname)
            plotting.plot_timeserie_one_file(in_fpath, out_quickplot_fpath, dpi=150)

    if QUICKPLOT_POSTPROCESSED:
        # Finding files to process
        all_files_postproc = preprocessing.load_dataset(dir_postproc_netcdf, verbose=VERBOSE)

        for in_fpath in all_files_postproc:
            out_quickplot_fname = os.path.basename(in_fpath).split('.')[0] + '.png'
            out_quickplot_fpath = os.path.join(dir_quickplots_postproc, out_quickplot_fname)
            plotting.plot_timeserie_one_file(in_fpath, out_quickplot_fpath, dpi=150)


if __name__ == '__main__':
    main()
