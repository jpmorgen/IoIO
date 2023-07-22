from datetime import date

from astropy.table import QTable

import matplotlib.pyplot as plt

from IoIO.utils import savefig_overwrite
from IoIO.na_nebula import plot_nightly_medians
from IoIO.torus import plot_ansa_brights, plot_epsilons, plot_ansa_pos

def master_stripchart(t_na, t_torus, nplots=4, start=None, stop=None,
                      na_medfilt_width=11, outname=None, figsize=None):

    if start is None:
        start = date.fromisoformat('2017-01-01')
    if stop is None:
        stop = date.today()

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(stop, str):
        stop = date.fromisoformat(stop)

    if figsize is None:
        if nplots == 2:
            figsize = [11, 11]
        else:
            #figsize = [15, 11.6]
            figsize = [21, 18]
        

    # Hints from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nplots, hspace=0)
    # This is giving trouble with dates being expressed as datetime objects whether or not sharex=True
    axs = gs.subplots(sharex=True)
    #axs = gs.subplots()

    #fig, axs = plt.subplots(nplots, figsize=figsize)
    plot_nightly_medians(t_na,
                         fig=fig, ax=axs[0],
                         tlim=(start, stop),
                         show=False,
                         medfilt_width=na_medfilt_width)

    plot_ansa_brights(t_torus,
                      fig=fig, ax=axs[1],
                      tlim=(start, stop))

    if nplots > 2:
        plot_epsilons(t_torus, fig=fig, ax=axs[2],
                      tlim=(start, stop),
                      show=False)

    if nplots > 3:
        plot_ansa_pos(t_torus, fig=fig, ax=axs[3],
                      tlim=(start, stop),
                      show=False)

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
t_na = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
#t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')
master_stripchart(t_na, t_torus)
#master_stripchart(t_na, t_torus, nplots=3)
#master_stripchart(t_na, t_torus, nplots=2)
#master_stripchart(t_na, t_torus, start='2018-01-01', stop='2018-08-01', figsize=figsize)
#master_stripchart(t_na, t_torus, start='2019-01-01', stop='2019-12-31', figsize=figsize)
#master_stripchart(t_na, t_torus, start='2019-01-01', stop='2019-12-31', nplots=2)
#master_stripchart(t_na, t_torus, start='2020-03-01', stop='2020-12-31', figsize=figsize)
#master_stripchart(t_na, t_torus, start='2020-01-01', stop='2020-12-31', nplots=2)
#master_stripchart(t_na, t_torus, start='2022-01-01', stop='2023-04-01',)
#master_stripchart(t_na, t_torus, start='2022-05-01', stop='2023-02-15', figsize=figsize)
#master_stripchart(t_na, t_torus, start='2022-05-01', stop='2023-02-01', nplots=2)

#master_stripchart(
#    t_na, t_torus, nplots=2,
#    start='2017-01-01', stop='2023-05-01',
#    outname=outdir + 'Na_SII_time_sequence.png')
#master_stripchart(t_na, t_torus,
#                  outname=outdir + 'Na_SII_epsilon_time_sequence.png')
#master_stripchart(t_na, t_torus, start='2018-01-01', stop='2018-08-01',
#                  outname=outdir + 'Na_SII_epsilon_2018.png')
#master_stripchart(t_na, t_torus, start='2019-02-01', stop='2019-11-01',
#                  outname=outdir + 'Na_SII_epsilon_2019.png')
#master_stripchart(t_na, t_torus, start='2020-03-01', stop='2020-12-31',
#                  outname=outdir + 'Na_SII_epsilon_2020.png')
#master_stripchart(t_na, t_torus, start='2022-05-01', stop='2023-03-01',
#                  outname=outdir + 'Na_SII_epsilon_2022-3.png')


#from IoIO.na_nebula import plot_obj_surf_bright
#plot_obj_surf_bright(t_na, show=True, fig_close=True)
