"""Module to read and plot Multi-Model Ensemble System for the outer Heliosphere (MMESH) outputs"""

from cycler import cycler

import numpy as np

from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import pandas as pd

from IoIO.juno import JunoTimes, PJAXFormatter

MME = '/data/sun/Jupiter_MME/JupiterMME.csv'

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
             filt_width=None,
             fig=None,
             ax=None,
             top_axis=False,
             tlim = None,
             ylim = None,
             yscale='linear',
             show=False,
             **kwargs):

    if filt_width is None:
        filt_width = np.asarray((300, 100, 30, 7))
        filt_width = filt_width*np.timedelta64(1, 'D')
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    df = readMMESH_fromFile(mme)
    col_labels = {'B_mag': r'$|\vec B_{\mathrm{IMF}}|$ (nT)',
                  'n_tot': r'Solar wind $\rho$ (cm$^{-3}$)',
                  'p_dyn': r'Solar wind $P_{\mathrm{dynamic}}$ (nPa)',
                  'u_mag': r'Solar wind velocity (km s$^{-1}$)'}


    goodidx = df[('ensemble', 'u_mag')] != 0
    df = df[goodidx]
    colval = df[('ensemble', colname)]
    col_neg_unc = df[('ensemble', f'{colname}_neg_unc')]
    col_pos_unc = df[('ensemble', f'{colname}_pos_unc')]

    plt.errorbar(df.index, colval, (col_neg_unc, col_pos_unc), alpha=0.1)

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
