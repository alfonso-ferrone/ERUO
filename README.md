# ERUO : Enhancement and Reconstruction of the spectrUm for the MRR-PRO

Library for the processing of raw spectra collected by the MRR-PRO in snowfall conditions.

## Description

This library has a two-fold objective:
* improving the quality of measurements, by lowering the minimum detectable attenuated equivalent reflectivity and by performing a simple dealiasing;
* addressing some issues typical of the MRR-PRO measurements, such as interference lines, random noise in the measurements and power drop in the spectrum at the extremes of the Doppler velocity range.

For a more detailed explanation of the library and the issues addressed, refer to the publication detailing it (in submission phase at the moment.)

Note that the library will not function if the MRR-PRO has been set to acquire *spectrum_reflectivity* instead of *spectrum_raw*.
You can set the MRR-PRO to acquire *spectrum_raw* during the configuration before deployment, if you plan to use ERUO for the processing.

## Getting Started

### Dependencies

The library has been written in Python 3.7, and tested on Windows 10 and on Ubuntu 20.04.

The Python libraries required for a correct functioning are:
* Numpy
* Scipy
* xarray
* astropy
* netCDF4
* joblib (for parallelization, it may be bypassed by removing a couple of imports and setting all the parallelization-related flags to *False*)
* Matplotlib (for visualization, it may be bypassed by removing a couple of imports and setting all the plot-related flags to *False*)

### Installing

The library can be downloaded by Github, at the address:
https://github.com/alfonso-ferrone/ERUO_v1

The library is designed to be run without installation.
Simply download the Python code, place them in a directory of your choosing, and execute the operations detailed in the next paragraph.

### Executing program

Once the MRR-PRO original files have been placed in a folder, following the same folder-tree structure as given by the MRR-PRO ( *YYYYMM/YYYYMMDD/[MRR_ORIGINAL_FILE].nc* ), you can start to set the configuration parameters.
To do so, open the **config.ini** file and set the directory information under the [PATH] sections.
There, you can decide the path to the original files (the parent directory under which you saved the three of files mentioned before), directories for processed and postprocessed output files, and the auxiliary quantities computed during the preprocessing.
The other parameters can be left as they are for the first run of the algorithm, but we still encourage you to have a look at them and some options to your preference (e.g. parallelization options, verbose options, ...)

After making sure that the **config.ini** file contains the correct paths, you can simply open a terminal (or in the Anaconda Prompt if you are on Windows and using Anaconda), navigate to the directory containing the four numbered ERUO scripts, and start launching them in order.
To start the preprocessing, type:

python3 01_preprocessing.py

Then, repeat the same action for all the next step (processing), by replacing **01_preprocessing.py** with the script **02_process_dataset.py**.
We suggest to run the script **04_generate_quickplots.py**, by setting the flag *QUICKPLOT_PROCESSED* equal to *True* and *QUICKPLOT_POSTPROCESSED* equal to *False* as soon as the processing ends.
This script displays the equivalent attenuated reflectivity and Doppler velocity of the processed files.
However, this step is purely optional, and if you prefer not to visualize those products, you can safely skip it.
Finally, you can postprocess the first set of ERUO products by launching the **03_postprocess_dataset.py** script.
You can visualize the equivalent attenuated reflectivity and Doppler velocity of the postprocessed files by running the script **04_generate_quickplots.py**, and setting the flag *QUICKPLOT_PROCESSED* equal to *False* and *QUICKPLOT_POSTPROCESSED* equal to *True*.
Once again, the visualization step is purely optional.

## Help

Even though in the previous section we advised to leave all parameters of the **config.ini** file as they are, there may be some situations in which some of them need to be changed to better fit the dataset.

Right after the preprocessing, we advise to look at the figures produced alongside the numpy archives, and make sure that the algorithm is not masking a very large fraction of the spectrum.
In case the masked section occupy the majority of the 2-dimensional spectrum, then you may need to lower some of the thresholds associated with the preprocessing in the **config.ini** file, such as
*PROMINENCE_INTERFERENCE_REMOVAL_RAW_SPECTRUM* or *NUM_ITERATIONS_INTERF_MASK_DILATION*.

If some of the masked parts resemble a precipitation signal, it means that your dataset contains a particularly large fraction of precipitation.
In this case, you may need to run again the preprocessing, including only clear-sky data.

A similar operation is needed if your dataset is too large and your computer has problem handling it.

The postprocessing also may need some special attention: if your dataset contains precipitation signal that persist for long period of times and that occupies only  a handful of range gates, the algorithm may confuse them with interference.
In this case, we advise to lower the thresholds associated with the postprocessing in the **config.ini** file.

You may also encounter problems when the postprocessing tries to handle particulalry short files.
A typical exaple is when the MRR-PRO starts the aquisition only few minutes before the change of hour, and the first file created only contains these few minutes of measurements.
In this case, the postprocessing will fail (likely during the identification of leftover interference lines).
We suggest to move these few problematic processed files to a different directory before starting the postprocessing.

## Authors

The library and the associated scientific publication have been designed and written by:
* Alfonso Ferrone
* Anne-Claire Billault-Roux
* Alexis Berne

All authors are affiliated to EPFL - LTE.


## Version History

* 1.0
	* Library uploaded before submission

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.

All program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain conditions.

## Acknowledgments

Help with the designing of the algorithm and/or with the practical implementation:
* Josué Gehring
* Gionata Ghiggi
* Monika Feldmann

Processing of the MRR-PRO *raw_spectrum*:
* M. Maahn and P. Kollias, with their publication *Improved Micro Rain Radar snow measurements using Doppler spectra post-processing*, DOI: https://doi.org/10.5194/amt-5-2661-2012
* Metek Meteorologische Messtechnik GmbH (Metek), for their amazingly detailed information available in both the instrument manuals and on their website (https://metek.de/).

Inspirations and code snippets:
* Colorbar centering around custom value: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
* Function to split a Numpy array in slices of contiguous non-NaN values: https://stackoverflow.com/questions/14605734/numpy-split-1d-array-of-chunks-separated-by-nans-into-a-list-of-the-chunks
