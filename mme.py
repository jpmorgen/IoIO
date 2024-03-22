"""Module to read and plot Multi-Model Ensemble System for the outer Heliosphere (MMESH) outputs"""

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
             fig=None,
             ax=None,
             top_axis=False,
             tlim = None,
             ylim = None,
             yscale='linear',
             show=False,
             **kwargs):

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    df = readMMESH_fromFile(mme)


    df = pd.read_csv(MME, delimiter=',',
                     comment='#')

    datetime = df['Unnamed: 0'][2:]
    datetime = pd.to_datetime(datetime)
    p_dyn = df['ensemble.7'][2:]
    p_dyn = p_dyn.astype(float)

    ax.plot(datetime, p_dyn)
    ax.set_xlabel('Date')
    ax.set_yscale(yscale)
    ax.set_ylabel('Solar Wind Dynamic Pressure (nPa)') 

    ax.set_xlim(tlim)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylim(ylim)
    fig.autofmt_xdate()
    ax.format_coord = PJAXFormatter(datetime, 'p_dyn')

    if top_axis:
        jts = JunoTimes()
        secax = ax.secondary_xaxis('top',
                                   functions=(jts.plt_date2pj, jts.pj2plt_date))
        secax.xaxis.set_minor_locator(MultipleLocator(1))
        secax.set_xlabel('PJ')

    if show:
        plt.show()

#plot_mme(top_axis=True, show=True)

import astropy.units as u
from astropy.time import TimeDelta
import datetime

colname = 'u_mag'
filt_width = np.timedelta64(1000, 'D')
#filt_width = datetime.timedelta(days=100)
#filt_width = 100*u.day
#filt_width = TimeDelta(filt_width)

df = readMMESH_fromFile(MME)
dt = np.median(df.index[1:] - df.index[0:-1])
filt_points = filt_width/dt
filt_points = filt_points.astype(int)
goodidx = df[('ensemble', 'u_mag')] != 0
df = df[goodidx]
colval = df[('ensemble', colname)]
col_neg_unc = df[('ensemble', f'{colname}_neg_unc')]
col_pos_unc = df[('ensemble', f'{colname}_pos_unc')]
colavg = uniform_filter1d(colval, size=filt_points)

plt.errorbar(df.index, colval, (col_neg_unc, col_pos_unc))
plt.plot(df.index, colavg, 'r-')

plt.show()
