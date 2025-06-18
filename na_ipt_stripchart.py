from datetime import date

from astropy import log
from astropy.table import QTable

import matplotlib.pyplot as plt

from bigmultipipe import assure_list

from IoIO.utils import savefig_overwrite
from IoIO.master_stripchart import read_tables, master_stripchart

try:
    table_dict
except NameError:
    table_dict = read_tables()

outdir = '/data/IoIO/analysis/'

#figsize = (10, 11)
#t_na = QTable.read('/home/jpmorgen/Papers/io/IoIO_2017--2023_Sublimation_JGR/Na_nebula_Morgenthaler_etal_2023.ecsv')
#t_torus = QTable.read('/home/jpmorgen/Papers/io/IoIO_2017--2023_Sublimation_JGR/Torus_Morgenthaler_etal_2023.ecsv')
#master_stripchart(t_na, t_torus, nplots=2, figsize = [11, 7],
#                  outname='/home/jpmorgen/Papers/io/IoIO_2017--2023_Sublimation_JGR/Na_SII_time_sequence.png')

#t_na = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
#t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
#torus_day_table = 
##t_torus = QTable.read('/data/IoIO/Torus/Torus.ecsv')
#
## Clean up masked values --> obsolete?
#t_na = t_na[~t_na['mask']]
#t_torus = t_torus[~t_torus['mask']]

figsize = None
figsize = (7.5,5.5)
master_stripchart(plots=['na_nebula',
                         'ansa_brights'],
                  figsize=figsize,
                  **table_dict)

#figsize = (6, 3)
#master_stripchart(plots=['ansa_brights'],
#                  figsize=figsize,
#                  start='2018-01-01',
#                  stop='2018-07-15',
#                  **table_dict)

#master_stripchart(plots=['ansa_brights',
#                         'ansa_pos',
#                         'ansa_dusk_r_stddev',
#                         'ansa_dawn_r_stddev'],
#                  **table_dict)


## Testing to see if bad radial profile fits, yielding bad widths, are
## the problem with 2020 torus shrink.  Doesn't seem to make much
## difference in the 
#mask = table_dict['t_torus']['ansa_left_r_stddev'] > 0.165*u.Rjup
#table_dict['t_torus'] = table_dict['t_torus'][mask]
#mask = table_dict['t_torus']['ansa_left_r_stddev'] < 0.45*u.Rjup
#table_dict['t_torus'] = table_dict['t_torus'][mask]
#
#master_stripchart(plots=['na_nebula',
#                         'ansa_brights',
#                         'ansa_pos'],
#                  **table_dict)

#master_stripchart(plots=['na_nebula',
#                         'ansa_brights',
#                         'epsilons'],
#                  **table_dict)

#master_stripchart(plots=['ansa_brights',
#                         'ansa_pos',
#                         'epsilons'],
#                  **table_dict)

#master_stripchart(t_na, t_torus,
#                  plots=['Na_nighly_medians',
#                         'ansa_brights',
#                         'epsilons',
#                         'mme',
#                         'mme_epsilon_corr'],
#                  window='300D')



#master_stripchart(t_na, t_torus, nplots=2)
#master_stripchart(t_na, t_torus, nplots=2, start='2017-04-01', stop='2018-07-10')
#master_stripchart(t_na, t_torus, nplots=1, figsize=[12,8])

# Morgenthaler et al. 2019 errata
#_ = master_stripchart(t_na, t_torus, nplots=1, figsize=[6,6],
#                  start='2017-04-01', stop='2018-07-10',
#                  outname='/home/jpmorgen/Papers/IoIO/2018_event/IoIO_errata/Na_vs_T.png')
#_ = master_stripchart(t_na, t_torus, nplots=1, figsize=[6,6],
#                  start='2018-01-15', stop='2018-07-10',
#                  outname='/home/jpmorgen/Papers/IoIO/2018_event/IoIO_errata/Na_vs_T_2018.png')

#master_stripchart(t_na, t_torus, nplots=3)
#t_na, t_torus = master_stripchart(
#    t_na, t_torus, nplots=3, figsize=[12, 10],
#    outname='/home/jpmorgen/Conferences/JpGU/2024/IoIO_Na_IPT_epsilon_2017--2024.png')
#master_stripchart(t_na, t_torus, nplots=2)

#master_stripchart(t_na, t_torus, nplots=2, figsize = [12, 5.5])
#master_stripchart(t_na, t_torus, nplots=2, figsize = [12, 10])

#master_stripchart(t_na, t_torus, nplots=2, figsize = [12, 10],
#                  stop='2024-01-01',
#                  outname='/home/jpmorgen/Conferences/AGU/2023/Na_IPT_time_history.png')

#master_stripchart(t_na, t_torus, nplots=2, figsize = [12, 5.5],
#                  stop='2024-01-01',
#                  outname='/home/jpmorgen/Conferences/AGU/2023/Na_IPT_time_history_wide.png')

 
#master_stripchart(t_na, t_torus, nplots=2, figsize = [12, 5.5],
#                  outname='/home/jpmorgen/Conferences/DPS/2023_San_Antonio/Na_IPT_time_history.png')

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
