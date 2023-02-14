from datetime import date

from astropy.table import QTable

import matplotlib.pyplot as plt

from IoIO.utils import savefig_overwrite
from IoIO.na_nebula import plot_nightly_medians
from IoIO.torus import plot_ansa_brights, plot_epsilons, plot_ansa_pos

def master_stripchart(t_na, t_torus, nplots=4, start=None, stop=None,
                      outname=None):

    if start is None:
        start = date.fromisoformat('2017-01-01')
    if stop is None:
        stop = date.today()

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(stop, str):
        stop = date.fromisoformat(stop)

    if nplots == 2:
        fig = plt.figure(figsize=[8.5, 8.5])
    else:
        fig = plt.figure(figsize=[22, 17])
    ax = plt.subplot(nplots, 1, 1)

    plot_nightly_medians(t_na,
                         fig=fig, ax=ax,
                         tlim=(start, stop),
                         show=False)

    ax = plt.subplot(nplots, 1, 2)
    plot_ansa_brights(t_torus,
                      fig=fig, ax=ax,
                      tlim=(start, stop))

    if nplots > 2:
        ax = plt.subplot(nplots, 1, 3)
        plot_epsilons(t_torus, fig=fig, ax=ax,
                      tlim=(start, stop),
                      show=False)

    if nplots > 3:
        ax = plt.subplot(nplots, 1, 4)
        plot_ansa_pos(t_torus, fig=fig, ax=ax,
                      tlim=(start, stop),
                      show=False)

    plt.tight_layout()
    if outname is None:
        plt.show()
    else:
        savefig_overwrite(outname)
        plt.close()

outdir = '/home/jpmorgen/Papers/io/IoIO_2017--2022/'

t_na = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
#t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')
master_stripchart(t_na, t_torus)
###master_stripchart(t_na, t_torus, start='2022-05-01', stop='2023-02-01')
##
##master_stripchart(t_na, t_torus, nplots=2, start='2017-01-01', stop='2023-03-01', outname='/home/jpmorgen/Papers/io/IoIO_2017--2022/Na_SII_time_sequence_2023.png')
##master_stripchart(t_na, t_torus,
##                  outname=outdir + 'Na_SII_epsilon_time_sequence.png')
##master_stripchart(t_na, t_torus, start='2018-01-01', stop='2018-08-01',
##                  outname=outdir + 'Na_SII_epsilon_2018.png')
##master_stripchart(t_na, t_torus, start='2019-02-01', stop='2019-11-01',
##                  outname=outdir + 'Na_SII_epsilon_2019.png')
##master_stripchart(t_na, t_torus, start='2020-03-01', stop='2020-12-31',
##                  outname=outdir + 'Na_SII_epsilon_2020.png')
##master_stripchart(t_na, t_torus, start='2022-05-01', stop='2023-03-01',
##                  outname=outdir + 'Na_SII_epsilon_2022-3.png')
##
