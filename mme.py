"""Module to read and plot Multi-Model Ensemble System for the outer Heliosphere (MMESH) outputs"""

from cycler import cycler

import numpy as np

from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import pandas as pd

from astropy.table import QTable

from IoIO.juno import JunoTimes, PJAXFormatter
from IoIO.torus import plot_epsilons

MME = '/data/sun/Jupiter_MME/JupiterMME.csv'
TORUS = '/data/IoIO/Torus/Torus.ecsv'

# From https://zenodo.org/records/10687651
def readMMESH_fromFile(fullfilename):
    """
    

    Parameters
    ----------
    fullfilename : string
        The fully qualified (i.e., absolute) filename pointing to the MMESH 
        output .csv file

    Returns
    -------
    mmesh_results : pandas DataFrame
        The results of the MMESH run

    """
    import pandas as pd
    
    #   Read the csv, parsing MultiIndex columns correctly
    mmesh_results = pd.read_csv(fullfilename, 
                                comment='#', header=[0,1], index_col=[0])
    
    #   Reset the index to be datetimes, rather than integers
    mmesh_results.index = pd.to_datetime(mmesh_results.index)
    
    return mmesh_results

def plot_mme(mme=MME,
             colname=None,
             hourly=True,
             alpha=0.1,
             filt_width=None,
             fig=None,
             ax=None,
             top_axis=False,
             tlim = None,
             ylim = None,
             yscale='linear',
             show=False,
             **kwargs):

    if mme is None:
        mme = MME
    if isinstance(mme, str):
        df = readMMESH_fromFile(mme)        

    if filt_width is None:
        filt_width = np.asarray((300, 100, 30, 7))
        filt_width = filt_width*np.timedelta64(1, 'D')
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    col_labels = {'B_mag': r'$|\vec B_{\mathrm{IMF}}|$ (nT)',
                  'n_tot': r'Solar wind $\rho$ (cm$^{-3}$)',
                  'p_dyn': r'Solar wind $P_{\mathrm{dynamic}}$ (nPa)',
                  'u_mag': r'Solar wind velocity (km s$^{-1}$)'}


    goodidx = df[('ensemble', 'u_mag')] != 0
    df = df[goodidx]
    colval = df[('ensemble', colname)]
    col_neg_unc = df[('ensemble', f'{colname}_neg_unc')]
    col_pos_unc = df[('ensemble', f'{colname}_pos_unc')]

    if hourly:
        plt.errorbar(df.index, colval, (col_neg_unc, col_pos_unc), alpha=alpha)

    # Plot running averages
    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    # median delta T beween model points
    dt = np.median(df.index[1:] - df.index[0:-1])
    if np.isscalar(filt_width):
        filt_width = [filt_width]
    for fw in filt_width:
        filt_points = fw/dt
        filt_points = filt_points.astype(int)
        colavg = uniform_filter1d(colval, size=filt_points)
        plt.plot(df.index, colavg, linewidth=5, label=fw)

    plt.legend()

    ax.set_xlabel('Date')
    ax.set_yscale(yscale)
    ax.set_ylabel(col_labels[colname])

    ax.set_xlim(tlim)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylim(ylim)
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(df.index, colval)

    if top_axis:
        jts = JunoTimes()
        secax = ax.secondary_xaxis('top',
                                   functions=(jts.plt_date2pj,
                                              jts.pj2plt_date))
        secax.xaxis.set_minor_locator(MultipleLocator(1))
        secax.set_xlabel('PJ')

    if show:
        plt.show()

def calc_mme_epsilon_corr(df_mme=None,
                          t_torus=None,
                          window='30D'):
    if df_mme is None:
        df_mme = MME
    if isinstance(df_mme, str):
        df_mme = readMMESH_fromFile(df_mme)
    if t_torus is None:
        t_torus = TORUS
    if isinstance(t_torus, str):
        t_torus = QTable.read(t_torus)
    t_torus = t_torus[~t_torus['mask']]
    t_torus.sort('tavg')

    # Hack until I get epsilons properly calculated outside of plot routine
    t_torus = plot_epsilons(t_torus)
    plt.close()

    # Filter out multi-dimensional columns Pandas can't deal with
    one_d_colnames = [cn for cn in t_torus.colnames
                      if len(t_torus[cn].shape) <= 1]
    df_torus = t_torus[one_d_colnames].to_pandas()
    df_torus = df_torus.set_index(df_torus['tavg'])

    # I am getting all NANs in rolling_corr, so I think I need to
    # interpolate the MME points to match the IoIO points
    series1 = df_mme['ensemble', 'p_dyn']
    series2 = df_torus['medfilt_interp_epsilon']
    series1i = np.interp(series2.index, series1.index, series1)
    series1i = pd.DataFrame(series1i, index=series2.index,
                            columns=['p_dyn'])
    rolling_corr = series1i.rolling(window).corr(series2)
    return rolling_corr

def plot_mme_epsilon_corr(rolling_corr,
                          fig=None,
                          ax=None,
                          x_axis='tavg',
                          y_axis='p_dyn',
                          top_axis=False,
                          tlim = None,
                          ylim = None,
                          show=False,
                          **kwargs):

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    rolling_corr.reset_index().plot(kind='scatter',
                                    x=x_axis,
                                    y=y_axis,
                                    ax=ax)
    # Or just extract the data and plot it with plt.scatter or
    # plt.plot
    if show:
        plt.show()

rolling_corr = calc_mme_epsilon_corr()
plot_mme_epsilon_corr(rolling_corr, show=True)

# I was thinking of getting fancy with interpolating torus instead of
#MME, but from a data perspective, that is a bad idea 
#from scipy.signal import medfilt
## Calculate interpolated epsilons for a particular set of times
## --> This eventually goes into torus.py
## --> Hey, there are lots of great functions in there!
#from astropy.utils.masked.function_helpers import interp
#from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans
##from IoIO.torus import add_medfilt,add_interpolated#, plot_epsilons
#medfilt_width=21
#t = t_torus
#
#meds = nanmedian(t['ansa_right_r_peak'], medfilt_width=medfilt_width
#
#
#add_medfilt(t, 'ansa_right_r_peak', medfilt_width=medfilt_width)
#add_medfilt(t, 'ansa_left_r_peak', medfilt_width=medfilt_width)
#r_peak = t['ansa_right_r_peak']
#l_peak = t['ansa_left_r_peak']
#av_peak = (np.abs(r_peak) + np.abs(l_peak)) / 2
#epsilon = -(r_peak + l_peak) / av_peak
#denom_var = t['ansa_left_r_peak_err']**2 + t['ansa_right_r_peak_err']**2
#num_var = denom_var / 2
#epsilon_err = epsilon * ((denom_var / (r_peak + l_peak)**2)
#                         + (num_var / av_peak**2))**0.5
#epsilon = epsilon.filled(np.NAN)
#epsilon_err = np.abs(epsilon_err.filled(np.NAN))
#epsilon_biweight = nan_biweight(epsilon)
#epsilon_mad = nan_mad(epsilon)
#bad_mask = np.isnan(epsilon)
#good_epsilons = epsilon[~bad_mask]
#assert len(good_epsilons) > 20
#med_epsilon = medfilt(good_epsilons, 11)
#kernel = Gaussian1DKernel(10)
#
#interp
#
#add_interpolated(t, 'ansa_right_r_peak_medfilt', kernel)
#add_interpolated(t, 'ansa_left_r_peak_medfilt', kernel)
#
#r_med_peak = t['ansa_right_r_peak_medfilt_interp']
#l_med_peak = t['ansa_left_r_peak_medfilt_interp']
#av_med_peak = (np.abs(r_med_peak) + np.abs(l_med_peak)) / 2
#t['medfilt_interp_epsilon'] = -(r_med_peak + l_med_peak) / av_med_peak  




#plot_mme(colname='B_mag', top_axis=True, show=True)
#plot_mme(colname='n_tot', top_axis=True, show=True)
#plot_mme(colname='p_dyn', top_axis=True, show=True)
#plot_mme(colname='u_mag', top_axis=True, show=True)




#colname = 'u_mag'
##filt_width = np.timedelta64(1000, 'D')
#filt_width = [np.timedelta64(1000, 'D'),
#              np.timedelta64(100, 'D'),
#              np.timedelta64(10, 'D')]
#
#df = readMMESH_fromFile(MME)
#goodidx = df[('ensemble', 'u_mag')] != 0
#df = df[goodidx]
#colval = df[('ensemble', colname)]
#col_neg_unc = df[('ensemble', f'{colname}_neg_unc')]
#col_pos_unc = df[('ensemble', f'{colname}_pos_unc')]
#
#plt.errorbar(df.index, colval, (col_neg_unc, col_pos_unc))
#
#
#custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
#
## Plot running averages
## median delta T beween model points
#dt = np.median(df.index[1:] - df.index[0:-1])
#if np.isscalar(filt_width):
#    filt_width = [filt_width]
#for fw in filt_width:
#    filt_points = fw/dt
#    filt_points = filt_points.astype(int)
#    colavg = uniform_filter1d(colval, size=filt_points)
#    plt.plot(df.index, colavg, label=fw)
#
#plt.legend()
#plt.show()
