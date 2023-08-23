from datetime import date

from astropy.table import QTable

import matplotlib.pyplot as plt

from IoIO.utils import savefig_overwrite
from IoIO.torus import (add_mask_col, plot_ansa_brights, plot_epsilons,
                        plot_ansa_pos, plot_ansa_r_amplitudes,
                        plot_ansa_y_peaks, plot_ansa_r_stddevs,
                        plot_dawn_r_stddev, plot_dusk_r_stddev,
                        plot_dawn_y_stddev, plot_dusk_y_stddev,
                        plot_dawn_cont, plot_dusk_cont,
                        plot_dawn_slope, plot_dusk_slope)

def torus_epsilon(t_torus, nplots=2, start=None, stop=None,
                   outname=None, figsize=None):

    if start is None:
        start = date.fromisoformat('2018-01-01')
    if stop is None:
        #stop = date.today()
        stop = date.fromisoformat('2022-12-07')

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(stop, str):
        stop = date.fromisoformat(stop)

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
    plot_ansa_brights(t_torus,
                      fig=fig, ax=axs[0],
                      vlines=vlines,
                      tlim=(start, stop),
                      ylim=(0, 200),
                      top_axis=True)

    if nplots > 1:
        plot_epsilons(t_torus, fig=fig, ax=axs[1],
                      vlines=vlines,
                      tlim=(start, stop),
                      show=False)

    if nplots > 2:
        plot_ansa_pos(t_torus, fig=fig, ax=axs[2],
                      vlines=vlines,
                      tlim=(start, stop),
                      show=False)

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

    if nplots > 3:
        plot_dusk_cont(t_torus, fig=fig, ax=axs[3],
                       tlim=(start, stop),
                       show=False)
    
    if nplots > 4:
        plot_dawn_cont(t_torus, fig=fig, ax=axs[4],
                       tlim=(start, stop),
                       show=False)
    
    if nplots > 5:
        plot_dusk_slope(t_torus, fig=fig, ax=axs[5],
                       tlim=(start, stop),
                       show=False)
    
    if nplots > 6:
        plot_dawn_slope(t_torus, fig=fig, ax=axs[6],
                       tlim=(start, stop),
                       show=False)

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

figsize = (10, 11)
t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
#torus_epsilon(t_torus, 2)

#torus_epsilon(t_torus, 2, start='2018-02-15', stop='2020-08-01')
# For AGU abstract
#torus_epsilon(t_torus, 3, start='2018-02-15', stop='2020-08-01',
#              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_pos_2018--2020.png')

torus_epsilon(t_torus, 2, start='2018-02-15', stop='2020-08-01',
              figsize=(9, 5.5),
              outname='/home/jpmorgen/Conferences/AGU/2023/IPT_epsilon_2018--2020.png')


# t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')
# add_mask_col(t_torus)
# t_torus = t_torus[~t_torus['mask']]
# torus_validate(t_torus)


#import numpy as np
#delta = t_torus['ansa_right_y_stddev'] - t_torus['ansa_right_r_stddev']
#print(np.nanmin(delta), np.nanmax(delta))
#delta = t_torus['ansa_left_y_stddev'] - t_torus['ansa_left_r_stddev']
#print(np.nanmin(delta), np.nanmax(delta))
