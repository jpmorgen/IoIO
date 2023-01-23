"""Create Na and IPT time sequence plot and tables for Morgenthaler et al. 2023"""
from datetime import date

from astropy.table import QTable

import matplotlib.pyplot as plt

from IoIO.utils import savefig_overwrite
from IoIO.na_nebula import plot_nightly_medians
from IoIO.torus import plot_ansa_brights, plot_epsilons, plot_ansa_pos

def master_stripchart(t_na, t_torus, nplots=4, start=None, stop=None,
                      outname=None):

    if start is None:
        start = '2017-01-01'
    if stop is None:
        stop = '2023-01-01'

    start = date.fromisoformat(start)
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


t_na = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
t_torus = t_torus[~t_torus['mask']]

master_stripchart(t_na, t_torus, nplots=2, start='2017-01-01', stop='2023-01-01', outname='/home/jpmorgen/Papers/io/IoIO/Na_SII_time_sequence.png')

# Incomplete until reprocess.  Shutter time uncertainties < 0.5s
t_na.remove_column('tavg_uncertainty')
t_na.write('/home/jpmorgen/Papers/io/IoIO/Na_nebula_Morgenthaler_etal_2023.ecsv')

remove_cols = ['tavg_uncertainty', 'ansa_left_r_peak', 'ansa_left_r_peak_err', 'ansa_left_r_stddev', 'ansa_left_r_stddev_err', 'ansa_left_r_amplitude', 'ansa_left_r_amplitude_err', 'ansa_left_y_peak', 'ansa_left_y_peak_err', 'ansa_left_y_stddev', 'ansa_left_y_stddev_err', 'ansa_right_r_peak', 'ansa_right_r_peak_err', 'ansa_right_r_stddev', 'ansa_right_r_stddev_err', 'ansa_right_r_amplitude', 'ansa_right_r_amplitude_err', 'ansa_right_y_peak', 'ansa_right_y_peak_err', 'ansa_right_y_stddev', 'ansa_right_y_stddev_err']

t_torus.remove_columns(remove_cols)
t_torus.write('/home/jpmorgen/Papers/io/IoIO/Torus_Morgenthaler_etal_2023.ecsv')
