from datetime import date

import numpy as np

from astropy.table import QTable

import matplotlib.pyplot as plt

from bigmultipipe import assure_list

from IoIO.utils import savefig_overwrite
from IoIO.torus import (add_epsilons, plot_ansa_brights, plot_epsilons,
                        plot_ansa_pos, plot_ansa_r_amplitudes,
                        plot_ansa_y_peaks, plot_ansa_r_stddevs,
                        plot_dawn_r_stddev, plot_dusk_r_stddev,
                        plot_dawn_y_stddev, plot_dusk_y_stddev,
                        plot_dawn_cont, plot_dusk_cont,
                        plot_dawn_slope, plot_dusk_slope)
from IoIO.mme import qtable2df, plot_mme, plot_mme_epsilon_corr

def torus_epsilon(t_torus, plots=['ansa_brights', 'epsilons'],
                  mme_colname='p_dyn',
                  mme_windows=None,
                  start=None, stop=None,
                  outname=None, figsize=None, **kwargs):

    plots = assure_list(plots)
    mme_windows = assure_list(mme_windows)
    if start is None:
        start = date.fromisoformat('2017-01-01')
    if stop is None:
        stop = date.today()
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(stop, str):
        stop = date.fromisoformat(stop)
    nplots = len(plots) + len(mme_windows)
    if 'mme_epsilon_corr' in plots and len(mme_windows) > 0:
        nplots -= 1
    if figsize is None:
        if nplots == 2:
            figsize = [18, 11]
        else:
            #figsize = [15, 11.6]
            figsize = [21, 18]
        
    # Hints from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nplots, hspace=0)
    # This is giving trouble with dates being expressed as datetime objects whether or not sharex=True
    axs = gs.subplots(sharex=True)
    #axs = gs.subplots()


    vlines = [('2018-04-19', 'green'),
              ('2018-05-08', 'orange'),
              ('2018-06-06', 'green'),
              ('2019-03-15', 'orange'),
              ('2019-04-24', 'green'),
              ('2019-05-05', 'orange'),
              ('2019-06-01', 'green'),
              ('2019-06-20', 'orange'),
              ('2019-07-15', 'green'),
              ('2019-08-17', 'orange'),
              ('2019-10-13', 'green'),
              ('2020-05-10', 'orange'),
              ('2020-06-01', 'green'),
              ('2020-06-20', 'orange'),
              ('2022-10-16', 'green'),
              ('2022-10-29', 'orange'),
              ('2022-11-10', 'green')]
    vlines = None

    iplot = 0
    for plotname in plots:
        if nplots == 1:
            ax = axs
        else:
            ax = axs[iplot]
        top_axis = iplot == 0
        if plotname == 'ansa_brights':    
            plot_ansa_brights(
                t_torus,
                fig=fig, ax=ax,
                vlines=vlines,
                tlim=(start, stop),
                ylim=(0, 200),
                top_axis=top_axis,
                **kwargs)
        elif plotname == 'epsilons':    
            plot_epsilons(
                t_torus, fig=fig, ax=ax,
                vlines=vlines,
                tlim=(start, stop),
                show=False,
                top_axis=top_axis,
                **kwargs)
        elif plotname == 'ansa_pos':
            plot_ansa_pos(
                t_torus, fig=fig, ax=ax,
                vlines=vlines,
                tlim=(start, stop),
                show=False,
                top_axis=top_axis,
                **kwargs)            
        elif plotname == 'mme':
            plot_mme(
                fig=fig, ax=ax,
                tlim=(start, stop),
                show=False,
                colname=mme_colname,
                top_axis=top_axis,
                **kwargs)
        elif plotname == 'mme_epsilon_corr':
            if len(mme_windows) == 0:
                mme_windows = ['30D']
            for iwindow, window in enumerate(mme_windows):
                ax = axs[iplot+iwindow]
                plot_mme_epsilon_corr(
                    window=window,
                    fig=fig, ax=ax,
                    tlim=(start, stop),
                    y_axis=mme_colname,
                    top_axis=top_axis,
                    **kwargs)
            iplot = iplot + len(mme_windows)-1
        else:
            # --> Write code for the rest as needed
            log.error(f'Unknown plot name: {plotname}')
        iplot += 1


    # if nplots > 2:
    #     plot_ansa_r_amplitudes(t_torus, fig=fig, ax=axs[2],
    #                            tlim=(start, stop),
    #                            show=False)

    #if nplots > 3:
    #    # Flat
    #    plot_ansa_y_peaks(t_torus, fig=fig, ax=axs[3],
    #                      tlim=(start, stop),
    #                      show=False)

    #if nplots > 3:
    #    # Too mixed in with large error bars
    #    plot_ansa_r_stddevs(t_torus, fig=fig, ax=axs[3],
    #                        tlim=(start, stop),
    #                        show=False)

    #if nplots > 4:
    #    plot_dusk_cont(t_torus, fig=fig, ax=axs[4],
    #                   tlim=(start, stop),
    #                   show=False)
    #
    #if nplots > 5:
    #    plot_dawn_cont(t_torus, fig=fig, ax=axs[5],
    #                   tlim=(start, stop),
    #                   show=False)
    #
    #if nplots > 6:
    #    plot_dusk_slope(t_torus, fig=fig, ax=axs[6],
    #                   tlim=(start, stop),
    #                   show=False)
    #
    #if nplots > 7:
    #    plot_dawn_slope(t_torus, fig=fig, ax=axs[7],
    #                   tlim=(start, stop),
    #                   show=False)

    # if nplots > 3:
    #     plot_dusk_r_stddev(t_torus, fig=fig, ax=axs[3],
    #                         tlim=(start, stop),
    #                         show=False)
    # 
    # if nplots > 4:
    #     plot_dawn_r_stddev(t_torus, fig=fig, ax=axs[4],
    #                         tlim=(start, stop),
    #                         show=False)
    # 
    # if nplots > 5:
    #     plot_dusk_y_stddev(t_torus, fig=fig, ax=axs[5],
    #                         tlim=(start, stop),
    #                         show=False)
    # 
    # if nplots > 6:
    #     plot_dawn_y_stddev(t_torus, fig=fig, ax=axs[6],
    #                         tlim=(start, stop),
    #                         show=False)

    for ax in axs:
        ax.label_outer()

    plt.tight_layout()
    if outname is None:
        plt.show()
    else:
        savefig_overwrite(outname)
        plt.close()

outdir = '/data/IoIO/analysis/'

t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')

#import astropy.units as u
#from utils import contiguous_sections
#t_torus.sort('tavg')
#
#
#time_col = 'tavg'
#max_night_gap=15*u.day
#t = t_torus
#t = t_torus[0:15]
#print(t)
#ts = contiguous_sections(t, time_col, 15*u.day)
#print(ts)



t_torus = t_torus[~t_torus['mask']]
t_torus.sort('tavg')
add_epsilons(t_torus)




torus_epsilon(t_torus, plots=['ansa_brights',
                              'epsilons',
                              'mme_epsilon_corr',
                              'mme'],
              mme_windows=['2D', '7D', '14D', '30D'])
              


#torus_epsilon(t_torus, 2)

#torus_epsilon(t_torus, 2, start='2018-02-15', stop='2020-08-01')
# For AGU abstract
#torus_epsilon(t_torus, 3, start='2018-02-15', stop='2020-08-01',
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2018--2020.png')

## torus_epsilon(t_torus, 2, start='2018-02-15', stop='2020-08-01',
##               figsize=(9, 5.5),
## #              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_2018--2020.png')
##               outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/IPT_epsilon_2018--2020.png')

#nplots = 2
#figsize=(12.5, 5.5)
#if nplots <= 2:
#    figsize=(5.5, 5.5)
#else:
#    figsize=(5.5, 8.5)

#torus_epsilon(t_torus, figsize=figsize, nplots=2, start='2018-02-15')
#torus_epsilon(t_torus, figsize=figsize, nplots=2, start='2018-02-15', stop='2018-07-01')

#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=3, colname='B_mag')
#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=3, colname='n_tot')
#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=3,
#              colname='p_dyn', alpha=0.01)#, hourly=False)

#filt_width = np.asarray((300, 100))
#filt_width = np.asarray((30, 2))
#filt_width = filt_width*np.timedelta64(1, 'D')
#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=7,
#              colname='p_dyn', filt_width=filt_width, alpha=0.01)#, hourly=False)

#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=3, colname='u_mag')

#torus_epsilon(t_torus, figsize=(12.5, 12.5), nplots=3, start='2017-10-01', stop='2021-03-01')

#torus_epsilon(t_torus, nplots, start='2018-02-15', stop='2018-07-01',
#              figsize=figsize, outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/IPT_epsilon_pos_2018.png')
#
#torus_epsilon(t_torus, nplots, start='2019-02-01', stop='2019-11-01',
#              figsize=figsize, outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/IPT_epsilon_pos_2019.png')
#
#torus_epsilon(t_torus, nplots, start='2020-03-27', stop='2020-08-01',
#              figsize=figsize, outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/IPT_epsilon_pos_2020.png')
#
#torus_epsilon(t_torus, nplots, start='2022-05-24', stop='2023-02-01',
#              figsize=figsize, outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/IPT_epsilon_pos_2022.png')

#torus_epsilon(t_torus, nplots, start='2023-09-01', stop='2024-01-01',
#              figsize=figsize, outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2023.png')

#torus_epsilon(t_torus, nplots=2,# start='2018-02-15',
#              figsize=(8.5, 5.5), outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_all.png')


#nplots = 2
#figscale = 2/3
##figsize=np.asarray(((5.5, 8.5))) * figscale
#figsize=np.asarray(((2.25, 5.5)))
#
#torus_epsilon(t_torus, nplots, start='2018-02-01', stop='2018-07-01',
#              figsize=figsize, legend=False,
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2018.png')
#
## More coverage in 2019
##fsize = np.asarray((9, 8.5)) * figscale
#fsize = np.asarray((3.25, 5.5))
#torus_epsilon(t_torus, nplots, start='2019-02-01', stop='2019-11-01',
#              figsize=fsize, legend=True,
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2019.png')
#
#torus_epsilon(t_torus, nplots, start='2020-03-15', stop='2020-08-15',
#              figsize=figsize, legend=False,
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2020.png')
#
#torus_epsilon(t_torus, nplots, start='2022-09-01', stop='2023-02-01',
#              figsize=figsize, legend=False,
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2022.png')
#
#torus_epsilon(t_torus, nplots, start='2023-08-01', stop='2024-01-01',
#              figsize=figsize, legend=False,
#outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2023.png')
#


# t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')
# add_mask_col(t_torus)
# t_torus = t_torus[~t_torus['mask']]
# torus_validate(t_torus)


#import numpy as np
#delta = t_torus['ansa_right_y_stddev'] - t_torus['ansa_right_r_stddev']
#print(np.nanmin(delta), np.nanmax(delta))
#delta = t_torus['ansa_left_y_stddev'] - t_torus['ansa_left_r_stddev']
#print(np.nanmin(delta), np.nanmax(delta))
