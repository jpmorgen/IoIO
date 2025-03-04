#!/usr/bin/python3

"""Module to read and plot Multi-Model Ensemble System for the outer Heliosphere (MMESH) outputs and compare to IoIO epsilon data"""

import os

from cycler import cycler

import numpy as np

from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import pandas as pd

import astropy.units as u
from astropy.table import QTable

from IoIO.utils import (qtable2df, plot_column, contiguous_sections,
                        nan_biweight, nan_mad)
from IoIO.juno import JunoTimes, PJAXFormatter
from IoIO.torus import TORUS_ROOT, BASE as TORUS_BASE #add_epsilons

MME = '/data/sun/Jupiter_MME/JupiterMME.csv'
MODEL = 'ensemble'
TORUS_DAY_TABLE = os.path.join(TORUS_ROOT, TORUS_BASE + '_day_table.ecsv')
MME_COL_LABELS = {'B_mag': r'$|\vec B_{\mathrm{IMF}}|$ (nT)',
                  'n_tot': r'Solar wind $\rho$ (cm$^{-3}$)',
                  'p_dyn': r'Solar wind $P_{\mathrm{dynamic}}$ (nPa)',
                  'u_mag': r'Solar wind velocity (km s$^{-1}$)'}

# Obsolete
RUNNING = 'running_' # prepended to running average columns
TORUS = '/data/IoIO/Torus/Torus_cleaned.ecsv'

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

# Obsolete
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

# This could be implemented with rolling stuff
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
    series2 = df2[colname2]
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

# --> This is where the new stuff begins
def add_rolling_funct_cols(df, colnames,
                           windows=None, # pandas.Timedelta
                           function=None,
                           min_periods=None,
                           center=True,
                           out_colbases=None):
    added_colnames = []
    if out_colbases is None:
        out_colbases = (None,) * len(colnames)
    for colname, out_colbase in zip(colnames, out_colbases):
        s = df[colname]
        for window in windows:
            r = s.rolling(window,
                          min_periods=min_periods,
                          center=center)
            mean_s = r.apply(function)
            outbase = out_colbase or colname
            filt_colname = f'{window.days}D'
            filt_colname = \
                f'rolling_{function.__name__}_{outbase}_{filt_colname}'
            df[filt_colname] = mean_s
            added_colnames.append(filt_colname)

    return added_colnames

def interp_to(target_df, source_df, source_colnames, suffix=''):
    added_colnames = []
    for colname in source_colnames:
        target_colname = f'{colname}{suffix}'
        s = np.interp(target_df.index, source_df.index, source_df[colname])
        target_df[target_colname] = s
        added_colnames.append(target_colname)

    return added_colnames

def add_rolling_corr(df, col1, window, col2, out_colname=None):
    if out_colname is None:
        out_colname = f'{col1}_{window.days}D_rolling_corr_{col2}'
    rolling_corr = df[col1].rolling(window).corr(df[col2])
    df[out_colname] = rolling_corr

def plot_mme(
        df_mme=None,
        torus_mme=None,
        mme_param='p_dyn',
        mme_rolling='3D',
        mme_rolling_func='nanmean',
        fig=None,
        ax=None,
        tlim=None,
        ylim=None):

    err_colname=(f'{mme_param}_neg_unc',
                 f'{mme_param}_pos_unc')

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot()
    handles = []

    label = f'{mme_param} original model'
    h = plot_column(df_mme,
                    time_col='datetime',
                    colname=mme_param,
                    err_colname=err_colname,
                    fmt='b.', alpha=0.1,
                    label=label,
                    fig=fig, ax=ax,
                    tlim=tlim)
    handles.append(h)
    label = f'{mme_param} rolling {mme_rolling} mean'
    rolling_col = f'rolling_{mme_rolling_func}_{mme_param}_{mme_rolling}'
    h = plot_column(df_mme,
                    colname=rolling_col,
                    fmt='c*', label=label,
                    markersize=8,
                    fig=fig, ax=ax,
                    tlim=tlim)
    handles.append(h)

    #label = label + ' interpolated onto IoIO times'
    #h = plot_column(torus_mme,
    #                colname=rolling_col,
    #                fmt='k*', label=label,
    #                markersize=8,
    #                fig=fig, ax=ax,
    #                tlim=tlim)
    #handles.append(h)


    ax.set_ylabel(MME_COL_LABELS[mme_param])
    ax.set_ylim(ylim)
    ax.legend(handles=handles)

def plot_mme_corr(torus_mme,
                  # 'epsilon_from_day_table' or epsilon_nnD, where nnD
                  # probably should match mme_rolling
                  epsilon_cols=['epsilon_3D'],
                  mme_params=['p_dyn'],
                  mme_rollings=['3D'], # '' to use raw interp
                  rolling_corrs=['10D'], # rolling corr window
                  fig=None,
                  ax=None,
                  tlim=None,
                  ylim=None,
                  legend_ncol=1,
                  **kwargs):

    fig = fig or plt.figure()
    ax = ax or fig.add_subplot()
    handles = []

    for mme_param in mme_params:
        for epsilon_col in epsilon_cols:
            for mme_rolling in mme_rollings:
                for rolling_corr in rolling_corrs:
                    epsilon_out = epsilon_col.replace('_', ' ')
                    label = f'Torus {epsilon_out} w/ '\
                        f'{mme_param} {mme_rolling} {rolling_corr} corr. window'
                    out1 = epsilon_col
                    if mme_rolling == '':
                        out2 = mme_param
                    else:
                        out2 = f'{mme_param}_{mme_rolling}'
                        colname = f'{out1}-{rolling_corr}_rolling_corr-{out2}'

                    h = plot_column(torus_mme,
                                    colname=colname,
                                    label=label,
                                    fig=fig, ax=ax,
                                    tlim=tlim,
                                    **kwargs)
                    handles.append(h)



    ax.set_ylabel(f'Rolling correlation coef.')
    ax.set_ylim(ylim)
    ax.legend(handles=handles, ncol=legend_ncol)

if __name__ == '__main__':
    mme_outname = os.path.join(TORUS_ROOT,
                               f'Jupiter_mme_{MODEL}.csv')

    # These are the time windows for making our rolling correlation.  I
    # want to make sure I synchronize the rolling nanmean of each function
    # to these in some way
    windows = ['3D', '10D', '30D', '100D', '200D']
    windows = [pd.Timedelta(w) for w in windows]
    mme_colnames=['B_mag', 'n_tot', 'p_dyn', 'u_mag']
    mme_neg_unc=[f'{c}_neg_unc' for c in mme_colnames]
    mme_pos_unc=[f'{c}_pos_unc' for c in mme_colnames]

    if os.path.exists(mme_outname):
        df_mme = pd.read_csv(mme_outname, index_col=[0])
        df_mme.index = pd.to_datetime(df_mme.index)
        mme_rolling_colnames = [rc for rc in df_mme.columns
                                if 'rolling_' in rc]
    else:
        df_mme = readMMESH_fromFile(MME)
        df_mme = df_mme['ensemble']
        # --> In an idea world, I propagate the uncertainties
        rc = add_rolling_funct_cols(df_mme, colnames=mme_colnames,
                                    windows=windows, function=np.nanmean)
        mme_rolling_colnames = rc
        df_mme.to_csv(mme_outname)


    #plot_column(df_mme, colname='rolling_nanmean_p_dyn_30D')
    #plt.show()

    df_torus = qtable2df(TORUS_DAY_TABLE, index='tavg')
    torus_rolling_colnames = []
    epsilon_colname='epsilon_from_day_table'
    rc = add_rolling_funct_cols(df_torus, colnames=[epsilon_colname],
                                windows=windows, function=np.nanmean,
                                out_colbases=['epsilon'])
    torus_rolling_colnames = rc
    interp_colnames = (mme_colnames + mme_neg_unc + mme_pos_unc +
                       mme_rolling_colnames)
    interp_colnames = interp_to(df_torus, df_mme, interp_colnames)

    # This is the meat of our code.  I have made it sort of general as
    # if it could be add_multi_corr or something like, that if need be
    c1s = [epsilon_colname] + torus_rolling_colnames
    c2s = interp_colnames
    remove1='rolling_nanmean_'
    remove2='rolling_nanmean_'
    windows=windows

    # Fortunately, replace doesn't raise an error if the string to
    # replacement is missing
    for c1 in c1s:
        out1 = c1.replace(remove1, '')
        for c2 in c2s:
            out2 = c2.replace(remove2, '')
            for window in windows:
                out_colname = f'{out1}-{window.days}D_rolling_corr-{out2}'
                add_rolling_corr(df_torus, c1, window, c2,
                                 out_colname=out_colname)
                # Defragment, since it's a long loop
                df_torus = df_torus.copy()

    outname = os.path.join(TORUS_ROOT, f'{TORUS_BASE}_mme_{MODEL}.csv')
    df_torus.to_csv(outname)


    # Print opposition correlations long term
    # --> This is not necessarily the best place to put this, but it
    # works for now.
    time_col = 'tavg'
    mme_param = 'p_dyn'
    #mme_param = 'u_mag'
    #mme_param = 'B_mag'
    #mme_param = 'n_tot'
    conjunction_time = 3*30*u.day
    t_torus = os.path.join(TORUS_ROOT, 'Torus_cleaned.ecsv')
    t_torus = QTable.read(t_torus)

    df_mme = readMMESH_fromFile(MME)
    df_mme = df_mme[MODEL]
    mme_datetimes = df_mme.index.to_pydatetime()

    torus_opp_list = contiguous_sections(t_torus, 'tavg', 3*30*u.day)
    n_oppositions =  len(torus_opp_list)

    midtimes = []
    to_corr = np.empty((2, n_oppositions))

    print('Long-term MME correlation with epsilon')
    print(f'Date range, av {mme_param}, epsilon biweight, mean')
    for it, t in enumerate(torus_opp_list):
        start = t['tavg'][0]
        stop = t['tavg'][-1]
        mid = start + (stop - start) / 2
        mask = np.logical_and(start.datetime < mme_datetimes,
                              mme_datetimes < stop.datetime)
        df = df_mme[mask]
        midtimes.append(mid)
        to_corr [0, it] = np.nanmean(t['epsilon']).value
        to_corr[1,it] = np.nanmean(df[mme_param])
        start.format = 'fits'
        stop.format = 'fits'
        starts, _ = str(start).split('T')
        stops, _ = str(stop).split('T')
        mme_mean = np.nanmean(df[mme_param])
        eps_biweight = nan_biweight(t['epsilon'])
        eps_mean = np.nanmean(t['epsilon'])
        eps_mad = nan_mad(t['epsilon'])
        print(f'{starts}, {stops}, {mme_mean:6.4f}, {eps_biweight:6.4f}+/-{eps_mad:6.4f}, {eps_mean:6.4f}+/-{eps_mad:6.4f}')
        #print(f'{np.nanmean(df[mme_param]):4.2f}')
        #print(start, stop, nan_biweight(t['epsilon']), nan_mad(t['epsilon']))
        #print(start, stop, np.nanmean(t['epsilon']), nan_mad(t['epsilon']))

    midtimes = np.asarray(midtimes)

    cov = np.corrcoef(to_corr)
    print(f'{mme_param} correlation with epsilon = {cov[0,1]:4.2f}')

#df_mme=None
#torus_mme=None
#
#if df_mme is None:
#    df_mme = os.path.join(TORUS_ROOT,
#                          f'Jupiter_mme_{MODEL}.csv')
#if isinstance(df_mme, str):
#        df_mme = pd.read_csv(df_mme, index_col=[0])
#        df_mme.index = pd.to_datetime(df_mme.index)
#if torus_mme is None:
#    torus_mme = os.path.join(TORUS_ROOT,
#                             f'{TORUS_BASE}_mme_{MODEL}.csv')
#if isinstance(torus_mme, str):
#    torus_mme = pd.read_csv(torus_mme, index_col='tavg')
#    torus_mme.index = pd.to_datetime(torus_mme.index)
#
#
#figsize=[12, 10]
#fig = plt.figure(figsize=figsize)
##plot_mme_corr(torus_mme, fig=fig)
##plot_mme_corr(torus_mme, fig=fig,
##              rolling_corrs=['10D', '30D'])
##plot_mme_corr(torus_mme, fig=fig,
##              mme_params=['p_dyn'],                          
##              epsilon_cols=['epsilon_10D'],
##              mme_rollings=['10D'],
##              rolling_corrs=['10D', '30D', '100D', '200D'])
##plot_mme_corr(torus_mme, fig=fig,
##              mme_params=['u_mag'],                          
##              epsilon_cols=['epsilon_10D'],
##              mme_rollings=['10D'],
##              rolling_corrs=['10D', '30D', '100D', '200D'])
##plot_mme_corr(torus_mme, fig=fig,
##              mme_params=['p_dyn', 'u_mag'],                          
##              epsilon_cols=['epsilon_10D'],
##              mme_rollings=['10D'],
##              rolling_corrs=['100D', '200D'])
#
#plot_mme_corr_kwargs_list = [
#    {'mme_params' : ['p_dyn'],
#     'epsilon_cols' : ['epsilon_10D'],
#     'mme_rollings' : ['10D'],
#     'rolling_corrs' :  ['10D', '30D', '100D', '200D']
#     },
#    {'mme_params' : ['u_mag'],
#     'epsilon_cols' : ['epsilon_10D'],
#     'mme_rollings' : ['10D'],
#     'rolling_corrs' :  ['10D', '30D', '100D', '200D']
#     },
#    {'mme_params' : ['p_dyn', 'u_mag', 'B_mag', 'n_tot'],
#     'mme_rollings' : ['10D'],
#     'rolling_corrs' :  ['100D', '200D']
#     }]
#
#corr_kwargs = plot_mme_corr_kwargs_list[-1]
#plot_mme_corr(torus_mme, fig=fig, **corr_kwargs)
#              
##for corr_kwargs in plot_mme_corr_kwargs_list:
##    plot_mme_corr(torus_mme, fig=fig
#
#
#plt.show()
#
#
#####
#import astropy.units as u
#from astropy.time import Time, TimeDelta
#from astropy.table import Table, QTable, vstack
#
#from IoIO.utils import filled_columns, fill_plot_col
#
#def qtable2df(t, index=None):
#    # If you need to filter or sort, do that first!
#    if isinstance(t, str):
#        t = QTable.read(t)
#    # This may be unnecessary
#    filled_columns(t, t.colnames, unmask=True)
#    one_d_colnames = [cn for cn in t.colnames
#                      if len(t[cn].shape) <= 1]
#    df = t[one_d_colnames].to_pandas()
#    if index is not None:
#        # This is 
#        df = df.set_index(df[index])
#    return df
#
##df_torus = qtable2df(TORUS_DAY_TABLE, index='tavg')
#
#def df_from_csv(fname, datetime_index=True, **kwargs):
#    df = pd.read_csv(fname, **kwargs)
#    if datetime_index:
#        df.index = pd.to_datetime(df.index)
#    return df
#
#df_mme=None
#torus_mme=None
#
#if df_mme is None:
#    df_mme = os.path.join(TORUS_ROOT,
#                          f'Jupiter_mme_{MODEL}.csv')
#if isinstance(df_mme, str):
#        df_mme = pd.read_csv(df_mme, index_col=[0])
#        df_mme.index = pd.to_datetime(df_mme.index)
#if torus_mme is None:
#    torus_mme = os.path.join(TORUS_ROOT,
#                             f'{TORUS_BASE}_mme_{MODEL}.csv')
#if isinstance(torus_mme, str):
#    torus_mme = pd.read_csv(torus_mme, index_col='tavg')
#    torus_mme.index = pd.to_datetime(torus_mme.index)
#
#
#epsilon_rolling = '3D'
#if epsilon_rolling == '':
#    epsilon_col = 'epsilon_from_day_table'
#else:
#    epsilon_col = f'epsilon_{epsilon_rolling}'
#plot_mme_corr(torus_mme,
#              epsilon_col=epsilon_col)
#              
#plt.show()
#
#add_rolling_corr(df_torus, 'epsilon_from_day_table_medfilt',
#                 pd.Timedelta(30, 'd'),
#                 'rolling_nanmean_p_dyn_3D',
#                 out_colname='epsilon_30D_rolling_p_dyn_3D')



#figsize=[12, 10]
#h = plot_column(df_torus, colname=out_colname,
#                label=out_colname)
##legend(handles=handles)
#plt.show()


#plot_column(df_torus, colname='rolling_nanmean_p_dyn_30D')
#plt.show()


# This is going to be add_rolling_corr




# --> THIS IS GOOD STUFF IF I WANT TO RE-IMPLEMENT WITH THESE TOOLS
#from astropy.stats import biweight_location, mad_std
#
#from IoIO.utils import filled_columns, plot_column
#
#def biweight_ignore_nan(x):
#    return biweight_location(x, ignore_nan=True)
#def mad_ignore_nan(x):
#    return biweight_location(x, ignore_nan=True)
#
#
#
#
#
#
#df_mme = readMMESH_fromFile(MME)
#t_torus = QTable.read(TORUS)
#filled_columns(t_torus, t_torus.colnames)
#df_torus = qtable2df(t_torus, index='tavg')
#
#
#df=df_torus
#colname='ansa_left_r_peak'
#biweight_filt_width=pd.Timedelta(20, 'd')
##min_periods=None
#min_periods=5
#center=True
#s = df[colname]
#r = s.rolling(biweight_filt_width,
#              min_periods=min_periods,
#              center=center)
#biweight_s = r.apply(biweight_ignore_nan, raw=True)
#s = df[f'{colname}_err']
#r = s.rolling(biweight_filt_width,
#              min_periods=min_periods,
#              center=center)
#mad_s = r.apply(mad_ignore_nan, raw=True)
#df[f'biweight_{colname}'] = biweight_s
#df[f'std_{colname}'] = mad_s
#plot_column(df, time_col='tavg',
#            colname=f'biweight_{colname}',
#            err_colname=f'std_{colname}')
##plot_column(df, time_col='tavg',
##            colname=f'std_{colname}')
#plt.show()


#t_torus = QTable.read(TORUS)
#t_torus = t_torus[~t_torus['mask']]
#t_torus.sort('tavg')
#add_epsilons(t_torus)
#df_torus = qtable2df(t_torus, index='tavg')
#add_df_running_av(df_torus, 'epsilon')

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
#epsilon = epsilon.filled(np.nan)
#epsilon_err = np.abs(epsilon_err.filled(np.nan))
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
