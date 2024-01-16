'''
plotting.py
Series of functions to aid the plotting of ERUO products and some of the  intermediate steps.

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
import numpy as np
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from configparser import ConfigParser

# --------------------------------------------------------------------------------------------------
# Figure parameters
figsize_1panel = (5,4)
figsize_2panels = (7,4)
figsize_3panels = (8.5,4)
figsize_4panels = (10,4)
figsize_4panels_verical = (6,8)

# Colormap can be also set from a different library..
# An example is colorcet:
# import colorcet as cc
# spectrum_cmap = cc.cm.rainbow
spectrum_cmap = 'inferno'
zea_cmap = 'rainbow'
vel_cmap = 'coolwarm'
width_cmap = 'viridis'
snr_cmap = 'plasma'
binary_cmap = 'RdYlGn_r'

# Format of x axis for timeseries
timeformat = mdates.DateFormatter('%H:%M')

# Font size
matplotlib.rcParams.update({'font.size': 5})

# For saving figures
DPI = 150
# --------------------------------------------------------------------------------------------------


class MidpointNormalize(matplotlib.colors.Normalize):
    '''
    Class, used in this script to create colorbars with a central point specified by the user.

    Inspired by one of the answers to:
    https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    And the example:
    https://matplotlib.org/3.2.2/gallery/userdemo/colormap_normalizations_custom.html
    '''
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_spectrum_reconstruction(anomaly_2d, reficiendo_2d):
    '''
    Plotting the masking of a specturm before the reconstruction.
    '''
    fig, axes = plt.subplots(1,3, figsize=figsize_3panels)

    mappable = axes[0].pcolormesh(np.arange(32), np.arange(256), anomaly_2d,
                                  cmap=spectrum_cmap, shading='nearest')
    axes[0].set_title('Origianl anomaly')
    plt.sca(axes[0])
    plt.colorbar(mappable=mappable, label='Raw spectrum anomaly [S.U.]')

    mappable = axes[1].pcolormesh(np.arange(32), np.arange(256), reficiendo_2d,
                                  cmap=binary_cmap, vmin=0, vmax=1, shading='nearest')
    axes[1].set_title('Area to reconstruct')
    plt.sca(axes[1])
    cbar = plt.colorbar(mappable=mappable, label='Binary mask', ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['OK', '', 'Masked'])

    tmp = copy.deepcopy(anomaly_2d)
    tmp[reficiendo_2d] = np.nan
    mappable = axes[2].pcolormesh(np.arange(32), np.arange(256), tmp, cmap=spectrum_cmap, shading='nearest')
    axes[2].set_title('Masked anomaly')
    plt.sca(axes[2])
    plt.colorbar(mappable=mappable, label='Raw spectrum anomaly [S.U.]')

    for ax in axes:
        ax.set_xlabel('vel. bin [-]')
        ax.set_ylabel('range gate idx [-]')

    plt.tight_layout()

    return fig, axes



def plot_spectrum(spec, v_0_3, r):
    '''
    Function to plot a spectrum.

    The figure contains a single panel, showing the spectrum. Velocity is on the x axis, and range
    on the y axis. 
    '''
    # If needed, those quantities can help customize the norm of the colormap
    #bounds = np.arange(-5., 16., 0.2)
    #norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize_1panel)
    ax.set_facecolor('gray')
    # If using "norm", remove "vmin" and "vmax"
    mappable = ax.pcolormesh(v_0_3, r/1000., spec, cmap=spectrum_cmap, # norm=norm,
                             vmin=np.nanquantile(spec[np.isfinite(spec)], q=0.01),
                             vmax=np.nanquantile(spec[np.isfinite(spec)], q=0.99),
                             shading='nearest')
    # Bug: nanquantile returns nan for high quantiles (even if it shouldn't, nan values should
    # be ignored). So, I needed to add "np.isfinite(spec)" to ignore them
    plt.sca(ax)
    plt.colorbar(mappable)
    
    ax.set_xlabel('v [m/s]')
    ax.set_ylabel('r [km]')
    
    plt.tight_layout()
    return fig, ax


def plot_noise_smoothed(r, noise_lvl_raw, noise_lvl, noise_lvl_nans, noise_lvl_tmp, standard_noise_lvl, noise_std):
    '''
    Function to plot smoothed and not smoothed noise lvl/std
    '''
    fig2, axes2 = plt.subplots(1,3,figsize=figsize_2panels)
    # Noise lvl
    axes2[0].plot(noise_lvl_tmp, r/1000., c='gray', label='noise_lvl_tmp', marker='o',
                  markerfacecolor='None', lw=0.6, ms=1.6, alpha=0.6)
    axes2[0].plot(noise_lvl_raw, r/1000., c='k', label='noise_lvl_raw', marker='.', lw=0.6, ms=1.5)
    axes2[0].plot(noise_lvl_nans, r/1000., c='tab:cyan', ls=':', alpha=0.6, lw=0.6, label='noise_lvl_nans')
    axes2[0].plot(standard_noise_lvl, r/1000., marker='+', c='tab:green', alpha=0.6, lw=0.6,
                  ms=2.1, label='standard_noise_lvl')
    axes2[0].plot(noise_lvl, r/1000., c='tab:blue', label='noise_lvl', marker='.', lw=0.6, ms=1.5, alpha=0.8)
    axes2[0].legend()
    axes2[0].set_xlabel('Noise lvl [mW]')

    axes2[1].plot(np.abs(noise_lvl_raw-standard_noise_lvl), r/1000., c='darkred', label='noise_lvl_raw-standard_noise_lvl',
                  marker='.', lw=0.6, ms=1.5)
    axes2[1].plot(np.abs(noise_lvl-standard_noise_lvl), r/1000., c='orangered', label='noise_lvl-standard_noise_lvl',
                  marker='.', lw=0.6, ms=1.5)
    axes2[1].set_xlabel('Noise lvl difference [mW]')
    # axes2[1].set_xlim((0, 1.))
    axes2[1].set_ylim(axes2[0].get_ylim())
    axes2[1].legend()

    axes2[2].plot(noise_std, r/1000., c='tab:gray', label='smoothed', marker='.', lw=0.6, ms=1.5, alpha=0.8)
    axes2[2].set_xlabel('Noise std [mW]')
    # Some formatting
    for ax2 in axes2:
        ax2.grid(ls=':', c='gray', alpha=0.5)
        ax2.set_ylabel('r [km]')
    plt.tight_layout()

    return fig2, axes2


def plot_spectrum_masked_and_moments(noise_masked_spectrum, v_0_3, r, params, noise_lvl, cmap=spectrum_cmap):
    '''
    Function to plot spectrum and moments
    '''
    # Defining what to plot
    var_to_plot = [params['power'], params['m1_dop'], noise_lvl, params['snr'], params['alt_snr']]
    title_list = ['Power', 'Moment 1', 'Noise level', 'SNR', '(sig+noise)/noise']
    color_list = ['tab:red', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:blue']
    
    # Setting up plot layout
    width_ratios = np.ones(1+len(var_to_plot))
    width_ratios[0] = 3.
    fig, axes = plt.subplots(1, 1+len(var_to_plot), figsize=figsize_4panels, sharey=True,
                             gridspec_kw={'width_ratios':width_ratios})

    # Plotting masked spectrum
    ax0 = axes[0]
    mappable = ax0.pcolormesh(v_0_3, r/1000., noise_masked_spectrum, cmap=cmap, shading='nearest')
    ax0.set_ylabel('Height AMSL [km]')
    ax0.set_xlabel('Doppler vel. [m/s]')
    ax0.set_title('Spectrum masked - noise floor')
    plt.sca(ax0)
    plt.colorbar(mappable=mappable, label='Power [mW]')

    # Plotting al moments profiles
    for i, ax in enumerate(axes[1:]):
        ax.grid(ls=':', c='gray', alpha=0.4)
        ax.plot(var_to_plot[i], r/1000, marker='o', ls='--', lw=0.6, ms=2., c=color_list[i])
        ax.set_xlabel(title_list[i])
        ax.set_title(title_list[i])

    # Adjusting limit on y axis to part with valid moments
    finite_params = np.isfinite(params['power'])
    if np.sum(finite_params):
        for ax in axes:
            ax.set_ylim((r[finite_params].min()/1000.-0.02, r[finite_params].max()/1000.+0.05))

    plt.tight_layout()
    
    return fig, axes


def plot_parameters_before_dBZ_conversion(v_0_3, r, spectrum_params):
    '''
    Function to plot spectrum moments before conversion to dBZ
    '''
    # Defining what to plot (not all moments...)
    var_to_plot = [spectrum_params['z'], spectrum_params['m1_dop'], spectrum_params['m2_dop'],
                   spectrum_params['snr'], spectrum_params['noise_floor_z']]
    title_list = ['reflectivity (linear)', 'Moment 1', 'Moment 2', 'SNR', 'noise_floor_z']
    color_list = ['tab:red', 'tab:blue', 'k', 'gray', 'tab:purple']

    fig, axes = plt.subplots(1, len(var_to_plot), figsize=figsize_4panels, sharey=True)

    # Plotting all moments profiles
    for i in range(len(var_to_plot)):
        ax = axes[i]
        ax.grid(ls=':', c='gray', alpha=0.3)
        ax.plot(var_to_plot[i], r/1000, marker='o', ls='--', lw=0.6, ms=2., c=color_list[i])
        ax.set_xlabel(title_list[i])
        ax.set_title(title_list[i])

    # Adjusting limit on y axis to part with valid moments
    finite_params = np.isfinite(var_to_plot[1])
    if np.sum(finite_params):
        for ax in axes:
            ax.set_ylim((r[finite_params].min()/1000.-0.02, r[finite_params].max()/1000.+0.05))

    plt.tight_layout()
    
    return fig, axes


def plot_spectrum_dBZ(v_0_3, r, output_dic):
    '''
    Function to plot spectrum and moments
    '''
    # Defining what to plot
    var_to_plot = [output_dic['spectrum_reflectivity'], output_dic['Zea'], output_dic['VEL'],
                   output_dic['SNR'], output_dic['noise_level'], output_dic['noise_floor']]


    title_list = ['spectrum_reflectivity', 'Zea', 'VEL', 'SNR',
                  'noise_level', 'noise_floor']
    color_list = [None, 'tab:red', 'tab:blue', 'k', 'gray', 'tab:purple', 'tab:pink']
    
    # Setting up plot layout
    width_ratios = np.ones(len(var_to_plot))
    width_ratios[0] = 3. # Spectrum plotted larger
    fig, axes = plt.subplots(1, len(var_to_plot), figsize=figsize_4panels, sharey=True,
                             gridspec_kw={'width_ratios':width_ratios})

    # Plotting masked spectrum
    ax0 = axes[0]
    mappable = ax0.pcolormesh(v_0_3, r/1000., var_to_plot[0], cmap=spectrum_cmap, shading='nearest')
    ax0.set_ylabel('Height AMSL [km]')
    ax0.set_xlabel('Doppler vel. [m/s]')
    ax0.set_title('Spectrum dBZ')
    plt.sca(ax0)
    plt.colorbar(mappable=mappable, label='Spectral reflectivity [dBZ]')

    # Plotting all moments profiles
    for i in range(1,len(var_to_plot)):
        ax = axes[i]
        ax.grid(ls=':', c='gray', alpha=0.3)
        ax.plot(var_to_plot[i], r/1000, marker='o', ls='--', lw=0.6, ms=2., c=color_list[i])
        ax.set_xlabel(title_list[i])
        ax.set_title(title_list[i])

    # Adjusting limits on x axis for spectrum
    finite_spectrum = np.isfinite(var_to_plot[0])
    valid_v = v_0_3[np.any(np.isfinite(var_to_plot[0]), axis=0)]
    axes[0].set_xlim((np.nanmin(valid_v)-0.5, np.nanmax(valid_v)+0.5))

    # Adjusting limit on y axis to part with valid moments
    finite_params = np.isfinite(var_to_plot[1])
    if np.sum(finite_params):
        for ax in axes:
            ax.set_ylim((r[finite_params].min()/1000.-0.02, r[finite_params].max()/1000.+0.05))

    plt.tight_layout()
    
    return fig, axes


def plot_clean_spectrum_dBZ(v_0_3, r, output_dic):
    '''
    Function to plot spectrum and moments
    '''
    # Defining what to plot
    var_to_plot = [output_dic['spectrum_reflectivity'], output_dic['Zea_clean'], 
                   output_dic['VEL_clean'], output_dic['WIDTH_clean'], output_dic['SNR_clean']]
    var_to_plot_raw = [None, output_dic['Zea'], output_dic['VEL'],
                       output_dic['WIDTH'], output_dic['SNR']]

    title_list = ['spectrum_reflectivity', 'Zea', 'VEL', 'WIDTH', 'SNR']
    color_list = [None, 'firebrick', 'midnightblue', 'darkgreen', 'k']
    color_raw_list = [None, 'lightcoral', 'skyblue', 'lightgreen', 'gray']

    # Setting up plot layout
    width_ratios = np.ones(len(var_to_plot))
    width_ratios[0] = 3. # Spectrum plotted larger
    fig, axes = plt.subplots(1, len(var_to_plot), figsize=figsize_4panels, sharey=True,
                             gridspec_kw={'width_ratios':width_ratios})

    # Plotting masked spectrum
    ax0 = axes[0]
    mappable = ax0.pcolormesh(v_0_3, r/1000., var_to_plot[0], cmap=spectrum_cmap, shading='nearest')
    ax0.set_ylabel('Height AMSL [km]')
    ax0.set_xlabel('Doppler vel. [m/s]')
    ax0.set_title('Spectrum dBZ')
    plt.sca(ax0)
    plt.colorbar(mappable=mappable, label='Spectral reflectivity [dBZ]')

    # Plotting all moments profiles
    for i in range(1,len(var_to_plot)):
        ax = axes[i]
        ax.grid(ls=':', c='gray', alpha=0.3)
        ax.plot(var_to_plot_raw[i], r/1000, marker='o', ls='--', lw=0.6, ms=2., c=color_raw_list[i])
        ax.plot(var_to_plot[i], r/1000, marker='o', ls='-', lw=0.6, ms=2., c=color_list[i])
        ax.set_xlabel(title_list[i])
        ax.set_title(title_list[i])

    # Adjusting limits on x axis for spectrum
    finite_spectrum = np.isfinite(var_to_plot[0])
    valid_v = v_0_3[np.any(np.isfinite(var_to_plot[0]), axis=0)]
    axes[0].set_xlim((np.nanmin(valid_v)-0.5, np.nanmax(valid_v)+0.5))

    # Adjusting limit on y axis to part with valid moments
    finite_params = np.isfinite(var_to_plot_raw[1])
    if np.sum(finite_params):
        for ax in axes:
            ax.set_ylim((r[finite_params].min()/1000.-0.02, r[finite_params].max()/1000.+0.05))

    plt.tight_layout()
    
    return fig, axes


def plot_initial_specrum_and_vars(fpath, spectrum_varname='spectrum_raw'):
    '''
    Function to plot spectrum and few variables from the initial netCDF file.

    The function create a plot with 4 panels:
    - the first one, larger, contains the spectrum;
    - the remainin three show the attenuated equivalent reflectivity (Zea), the Doppler velocity
      (VEL) and the signal to noise ratio (SNR).


    Parameters
    ----------
    fpath : str
        Full path to the initial netCDF file generated by the MRR
    spectrum_varname : str
        Name of the variable containing the spectrum

    '''
    with nc.Dataset(fpath) as ncfile:
        # Getting the variables from the netCDF file
        spectrum_raw = np.array(ncfile.variables[spectrum_varname])
        zea = np.array(ncfile.variables['Zea'])
        VEL = np.array(ncfile.variables['VEL'])
        SNR = np.array(ncfile.variables['SNR'])
        r = np.array(ncfile.variables['range'])

        # Size of spectrum
        N = spectrum_raw.shape[1]
        m = spectrum_raw.shape[2]

        # Reading configuration to compute velocity limits
        config_fpath = "config.ini"
        with open(config_fpath) as fp:
            config_object = ConfigParser()
            config_object.read_file(fp)
        fixed_params = config_object['FIXED_PARAMETERS']
        f_s = float(fixed_params['f_s'])
        lam = float(fixed_params['lam'])
        d_v = (lam * f_s) / (4. * N * m) # Velocity resolution

        # Velocity bins
        v_0 = np.arange(0., d_v * m, d_v)

        # Defining variable to plot, and assignign title and color to them
        var_to_plot = [spectrum_raw, zea, VEL, SNR]
        title_list = [spectrum_varname, 'Zea', 'VEL', 'SNR']
        color_list = [None, 'tab:red', 'tab:blue', 'tab:green', 'k', 'tab:purple']
        
        # Setting up plot layout
        width_ratios = np.ones(len(var_to_plot))
        width_ratios[0] = 3. # Spectrum plotted larger
        fig, axes = plt.subplots(1, len(var_to_plot), figsize=figsize_4panels, sharey=True,
                                 gridspec_kw={'width_ratios':width_ratios})

        # Plotting masked spectrum
        ax0 = axes[0]
        mappable = ax0.pcolormesh(v_0, r/1000., var_to_plot[0][0,:,:],
                                  cmap=spectrum_cmap, shading='nearest')

        # Some formatting of the panel
        ax0.set_ylabel('Height AMSL [km]')
        ax0.set_xlabel('Doppler vel. [m/s]')
        ax0.set_title('Spectrum raw [dB]')
        plt.sca(ax0)
        plt.colorbar(mappable=mappable, label='[dB]')

        # Plotting profiles of the variables 
        for i in range(1,len(var_to_plot)):
            ax = axes[i]
            ax.grid(ls=':', c='gray', alpha=0.3)
            ax.plot(var_to_plot[i][0,:], r/1000, marker='o', ls='--', lw=0.6, ms=2., c=color_list[i])
            ax.set_xlabel(title_list[i])
            ax.set_title(title_list[i])

        plt.tight_layout()


def plot_timeserie_one_file(in_fpath, out_fpath, dpi=150):
    '''
    Function to plot the timeseries of few of the variables from the processed file.

    The function creates a plot with 4 panels, illustrating the values Zea, VEL, WIDTH and SNR
    over the time interval saved in the netCDF file.
    The output plot is saved in PNG format.
    '''

    # Loading the file
    with nc.Dataset(in_fpath) as ncfile:
        # Dimensions
        r = np.array(ncfile.variables['range'])
        t = np.array(ncfile.variables['time'])

        # The processed variables to plot
        Zea = np.array(ncfile.variables['Zea'])
        VEL = np.array(ncfile.variables['VEL'])
        WID = np.array(ncfile.variables['WIDTH'])
        SNR = np.array(ncfile.variables['SNR'])

    fig = plt.figure(figsize=figsize_4panels_verical)

    # Plot structure
    widths = [1., 0.1]
    heights = [1., 1., 1., 1.]
    spec = fig.add_gridspec(ncols=2, nrows=4,
                            height_ratios=heights, width_ratios=widths)

    # Defining some limits
    vel_lim = np.nanquantile(np.abs(VEL), q=0.999)
    z_lims = (-20, 10)

    # In case the colorbar norm needs to be specified manually
    # norm_za = colors.BoundaryNorm(boundaries=np.arange(-10., 25.0, 1.), ncolors=256)
    # norm_vel = colors.BoundaryNorm(boundaries=np.arange(-1., 3.6, 0.1), ncolors=256)

    # Reflectivity (attenuated equivalent)
    ax_za = fig.add_subplot(spec[0, 0])
    ax_za.set_axisbelow(True)
    ax_za.set_facecolor('gray')
    pza = ax_za.pcolormesh(t, r, Zea.T, cmap=zea_cmap, vmin=z_lims[0], vmax=z_lims[1], shading='nearest')
    ax_za.set_ylabel('Height [m]')
    ax_za.set_xlabel('')
    # ax_za.set_xlim([ti, tf])
    # ax_za.xaxis.set_major_formatter(timeformat)
    ax_za.set_xticklabels([])
    ax_za.grid(c='w', ls=':', alpha=0.9)
    ax_za.set_title(os.path.basename(in_fpath))

    # Colorbar
    cbar_za_ax = fig.add_subplot(spec[0, 1])
    cbar_za_ax.set_xticklabels([])
    cbar_za_ax.set_xticks([])
    cbar_za_ax.set_yticks([])
    for sp in cbar_za_ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(pza, ax=cbar_za_ax, pad=0., label='Atten. equiv. reflectivity [dBZ]')

    # Vertical velocity
    ax_vel = fig.add_subplot(spec[1, 0])
    ax_vel.set_axisbelow(True)
    ax_vel.set_facecolor('gray')
    pvel = ax_vel.pcolormesh(t, r, VEL.T, cmap=vel_cmap, vmin=-vel_lim, vmax=vel_lim, shading='nearest')
    ax_vel.set_ylabel('Height [m]')
    ax_vel.set_xlabel('')
    # ax_vel.set_xlim([ti, tf])
    # ax_vel.xaxis.set_major_formatter(timeformat)
    ax_vel.set_xticklabels([])
    ax_vel.grid(c='w', ls=':', alpha=0.9)

    # Colorbar
    cbar_vel_ax = fig.add_subplot(spec[1, 1])
    cbar_vel_ax.set_xticks([])
    cbar_vel_ax.set_xticklabels([])
    cbar_vel_ax.set_yticks([])
    for sp in cbar_vel_ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(pvel, ax=cbar_vel_ax, pad=0., label='Doppler velocity [m/s]')

    # Spectral width
    ax_sw = fig.add_subplot(spec[2, 0])
    ax_sw.set_axisbelow(True)
    pza = ax_sw.pcolormesh(t, r, WID.T, cmap=width_cmap, shading='nearest')
    ax_sw.set_ylabel('Height [m]')
    ax_sw.set_xlabel('')
    # ax_sw.set_xlim([ti, tf])
    # ax_sw.xaxis.set_major_formatter(timeformat)
    ax_sw.set_xticklabels([])
    ax_sw.grid(c='gray', ls=':', alpha=0.6)

    # Colorbar
    cbar_sw_ax = fig.add_subplot(spec[2, 1])
    cbar_sw_ax.set_xticklabels([])
    cbar_sw_ax.set_xticks([])
    cbar_sw_ax.set_yticks([])
    for sp in cbar_sw_ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(pza, ax=cbar_sw_ax, pad=0., label='Spectral width [m/s]')

    # SNR
    ax_snr = fig.add_subplot(spec[3, 0])
    ax_snr.set_axisbelow(True)
    pvel = ax_snr.pcolormesh(t, r, SNR.T, cmap=snr_cmap, shading='nearest')#, norm=norm_vel)
    ax_snr.set_ylabel('Height [m]')
    ax_snr.set_xlabel('Time [HH:MM]')
    # ax_snr.set_xlim([ti, tf])
    ax_snr.xaxis.set_major_formatter(timeformat)
    ax_snr.set_xticklabels([])
    ax_snr.grid(c='gray', ls=':', alpha=0.6)

    # Colorbar
    cbar_snr_ax = fig.add_subplot(spec[3, 1])
    cbar_snr_ax.set_xticklabels([])
    cbar_snr_ax.set_xticks([])
    cbar_snr_ax.set_yticks([])
    for sp in cbar_snr_ax.spines.values():
        sp.set_visible(False)
    plt.colorbar(pvel, ax=cbar_snr_ax, pad=0., label='Signal to noise ratio [dB]')

    # Adjusting plot margins
    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05)
    # Saving
    fig.savefig(out_fpath, dpi=dpi)
    # Closing
    plt.close(fig)


def plot_postprocessing(t, r, ini_zea, out_zea, postprocessing_stage='Interf. removal',
                        z_min=-10., z_max=30.):
    '''
    Function to plot the timeseries of the postprocessed attenuated equivalent reflectivity.
    '''

    fig, axes = plt.subplots(1, 2, figsize=figsize_2panels)

    # In case the colorbar norm needs to be specified manually
    # norm_zea = colors.BoundaryNorm(boundaries=np.arange(-10., 25.0, 1.), ncolors=256)

    # Defining some limits
    if (z_min is None) or (z_max is None):
        z_lims = (np.nanmin(out_zea), np.nanmax(out_zea))
    else:
        z_lims = (z_min, z_max)

    # -----------------------------------
    # Reflectivity before prostprocessing
    ax_zea_before = axes[0]
    ax_zea_before.set_axisbelow(True)
    ax_zea_before.set_facecolor('gray')
    pzea_before = ax_zea_before.pcolormesh(t, r, ini_zea.T, cmap=zea_cmap,
                                           vmin=z_lims[0], vmax=z_lims[1], shading='nearest')
    ax_zea_before.set_ylabel('Height [m]')
    ax_zea_before.set_xlabel('Time [HH:MM]')
    ax_zea_before.set_title('Before %s' % postprocessing_stage)

    ax_zea_before.xaxis.set_major_formatter(timeformat)
    ax_zea_before.set_xticklabels([])
    ax_zea_before.grid(c='w', ls=':', alpha=0.9)

    # Colorbar
    plt.sca(ax_zea_before)
    plt.colorbar(pzea_before, label='Atten. equiv. reflectivity [dBZ]')

    # -----------------------------------
    # Reflectivity after prostprocessing
    ax_zea_after = axes[1]
    ax_zea_after.set_axisbelow(True)
    ax_zea_after.set_facecolor('gray')
    pzea_after = ax_zea_after.pcolormesh(t, r, out_zea.T, cmap=zea_cmap,
                                           vmin=z_lims[0], vmax=z_lims[1], shading='nearest')
    ax_zea_after.set_ylabel('Height [m]')
    ax_zea_after.set_xlabel('Time [HH:MM]')
    ax_zea_after.set_title('After %s' % postprocessing_stage)

    ax_zea_after.xaxis.set_major_formatter(timeformat)
    ax_zea_after.set_xticklabels([])
    ax_zea_after.grid(c='w', ls=':', alpha=0.9)

    # Colorbar
    plt.sca(ax_zea_after)
    plt.colorbar(pzea_after, label='Atten. equiv. reflectivity [dBZ]')

    plt.tight_layout()
    return fig, axes
