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
from IoIO.torus import add_epsilons

MME = '/data/sun/Jupiter_MME/JupiterMME.csv'
MODEL = 'ensemble'
RUNNING = 'running_' # prepended to running average columns
TORUS = '/data/IoIO/Torus/Torus.ecsv'

# This could go in some util thing
def qtable2df(t, index=None):
    # If you need to filter or sort, do that first!
    if isinstance(t, str):
        t = QTable.read(t_torus)
    one_d_colnames = [cn for cn in t.colnames
                      if len(t[cn].shape) <= 1]
    df = t[one_d_colnames].to_pandas()
    if index is not None:
        df = df.set_index(df[index])
    return df

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

def add_mme_running_av(df,
                       colname,
                       model=MODEL,
                       filt_width=None):
    if filt_width is None:
        filt_width = np.asarray((300, 100, 30, 7))
        filt_width = filt_width*np.timedelta64(1, 'D')
    # median delta T beween model points
    dt = np.median(df.index[1:] - df.index[0:-1])
    if np.isscalar(filt_width):
        filt_width = [filt_width]
    for fw in filt_width:
        filt_points = fw/dt
        filt_points = filt_points.astype(int)
        colavg = uniform_filter1d(df[model, colname],
                                     size=filt_points)
        filt_colname = fw.astype(str)
        filt_colname = filt_colname.replace(' ', '_')
        filt_colname = f'{RUNNING}{colname}_{filt_colname}'
        df[model, filt_colname] = colavg

def add_df_running_av(df,
                      colname,
                      filt_width=None):
    if filt_width is None:
        filt_width = np.asarray((300, 100, 30, 7))
        filt_width = filt_width*np.timedelta64(1, 'D')
    if isinstance(colname, tuple):
        # Make manipulations easier
        colname_list = list(colname)
        multi_col = colname_list[0:-1]
        filt_colname = colname_list[-1]
    else:
        filt_colname = colname
    # median delta T beween model points
    dt = np.median(df.index[1:] - df.index[0:-1])
    if np.isscalar(filt_width):
        filt_width = [filt_width]
    for fw in filt_width:
        filt_points = fw/dt
        filt_points = filt_points.astype(int)
        colavg = uniform_filter1d(df[colname],
                                     size=filt_points)
        fw_str = fw.astype(str)
        fw_str = fw_str.replace(' ', '_')
        tfilt_colname = f'{RUNNING}{filt_colname}_{fw_str}'
        if isinstance(colname, tuple):
            tmulti_col = multi_col.copy()
            tmulti_col.extend([tfilt_colname])
            tmulti_col = tuple(tmulti_col)
            df[tmulti_col] = colavg
        else:
            df[tfilt_colname] = colavg

def plot_mme(mme=MME,
             colname=None,
             model=MODEL,
             hourly=True,
             alpha=0.1,
             fig=None,
             ax=None,
             top_axis=False,
             tlim = None,
             ylim = None,
             yscale='linear',
             show=False,
             **kwargs): # passed on to add_mme_running_av

    if mme is None:
        mme = MME
    if isinstance(mme, str):
        df = readMMESH_fromFile(mme)        
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    col_labels = {'B_mag': r'$|\vec B_{\mathrm{IMF}}|$ (nT)',
                  'n_tot': r'Solar wind $\rho$ (cm$^{-3}$)',
                  'p_dyn': r'Solar wind $P_{\mathrm{dynamic}}$ (nPa)',
                  'u_mag': r'Solar wind velocity (km s$^{-1}$)'}

    # Filter out times for which model is not yet calculated
    goodidx = df[(model, 'u_mag')] != 0
    df = df[goodidx]
    colval = df[(model, colname)]
    col_neg_unc = df[(model, f'{colname}_neg_unc')]
    col_pos_unc = df[(model, f'{colname}_pos_unc')]

    if hourly:
        ax.errorbar(df.index, colval, (col_neg_unc, col_pos_unc), alpha=alpha)

    # Plot running averages
    custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    add_mme_running_av(df, colname, model=model, **kwargs)
    
    for col in df.columns:
        if col[1][0:len(RUNNING)] != RUNNING:
            continue
        fw = col[1][len(RUNNING)+1:]
        ax.plot(df.index, df[col], linewidth=5, label=fw)

    ax.legend()

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

def calc_running_corr(df1, df2, colname1, colname2, window):
    # Match time points of df1 (e.g. a model calculated on a regularly
    # space axis) to df2 (e.g. irregularly space data).  Otherwise
    # rolling.corr gives NANs everywhere
    series1 = df1[colname1]
    series2 = df2['medfilt_interp_epsilon']
    series1i = np.interp(series2.index, series1.index, series1)
    series1i = pd.DataFrame(series1i, index=series2.index,
                            columns=[colname1])
    rolling_corr = series1i.rolling(window).corr(series2)
    return rolling_corr
    

# This might be better just calc_corr with two dfs as input
def calc_mme_epsilon_corr(df_mme=None,
                          colname='p_dyn',
                          model=MODEL,
                          t_torus=None,
                          window=None):
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

    add_epsilons(t_torus)

    # Filter out multi-dimensional columns Pandas can't deal with
    one_d_colnames = [cn for cn in t_torus.colnames
                      if len(t_torus[cn].shape) <= 1]
    df_torus = t_torus[one_d_colnames].to_pandas()
    df_torus = df_torus.set_index(df_torus['tavg'])

    # I am getting all NANs in rolling_corr, so I think I need to
    # interpolate the MME points to match the IoIO points
    series1 = df_mme[model, colname]
    series2 = df_torus['medfilt_interp_epsilon']
    series1i = np.interp(series2.index, series1.index, series1)
    series1i = pd.DataFrame(series1i, index=series2.index,
                            columns=[colname])
    rolling_corr = series1i.rolling(window).corr(series2)
    return rolling_corr

def plot_mme_epsilon_corr(df_mme=None,
                          rolling_corr=None,
                          fig=None,
                          ax=None,
                          window='14D',
                          x_axis='tavg',
                          y_axis='p_dyn',
                          top_axis=False,
                          tlim = None,
                          ylim = None,
                          show=False,
                          **kwargs):

    if rolling_corr is None:
        rolling_corr = calc_mme_epsilon_corr(
            df_mme, colname=y_axis, window=window, **kwargs)
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    rolling_corr.reset_index().plot(
        kind='scatter',
        x=x_axis,
        y=y_axis,
        ax=ax,
        xlim=tlim,
        ylim=ylim,
        ylabel=f'{y_axis} {window}')
    #ax.set_xlim(tlim)
    #ax.set_ylim(ylim)
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(rolling_corr.index,
                                    rolling_corr[y_axis])

    if top_axis:
        jts = JunoTimes()
        secax = ax.secondary_xaxis('top',
                                   functions=(jts.plt_date2pj,
                                              jts.pj2plt_date))
        secax.xaxis.set_minor_locator(MultipleLocator(1))
        secax.set_xlabel('PJ')

    if show:
        plt.show()

t_torus = QTable.read(TORUS)
t_torus = t_torus[~t_torus['mask']]
t_torus.sort('tavg')
add_epsilons(t_torus)
df_torus = qtable2df(t_torus, index='tavg')
add_df_running_av(df_torus, 'epsilon')

#df_mme = readMMESH_fromFile(MME)
#add_df_running_av(df_mme, ('ensemble', 'p_dyn'))
#add_mme_running_av(df_mme, 'p_dyn')
#plot_mme(colname='p_dyn', top_axis=True, show=True)

#plot_mme_epsilon_corr(show=True)
#add_mme_running_av(df_mme, 'p_dyn')
#plot_mme_epsilon_corr(df_mme, y_axis='running_p_dyn_300_days', show=True)
#rolling_corr = calc_mme_epsilon_corr()
#plot_mme_epsilon_corr(rolling_corr, show=True)

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
