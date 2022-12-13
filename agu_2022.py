from datetime import date

from astropy.table import QTable

import matplotlib.pyplot as plt

from IoIO.utils import finish_stripchart
from IoIO.na_nebula import plot_nightly_medians
from IoIO.torus import plot_ansa_brights

start = date.fromisoformat('2017-01-01')
stop = date.fromisoformat('2023-01-01')

fig = plt.figure(figsize=[8.5, 11])
ax = plt.subplot(2, 1, 1)

plot_nightly_medians('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv',
                     fig=fig, ax=ax,
                     tlim=(start, stop),
                     show=False)

ax = plt.subplot(2, 1, 2)
t = QTable.read('/data/IoIO/Torus/Torus.ecsv')
plot_ansa_brights(t,
                  fig=fig, ax=ax,
                  tlim=(start, stop))
#plt.show()
finish_stripchart('/home/jpmorgen/Conferences/AGU/2022', 'Na_SII', show=True)



#f = plt.figure(figsize=[8.5, 4])
#torus_stripchart('/data/IoIO/Torus/Torus.ecsv', '/tmp', n_plots=1,
#                 tlim=(start, stop),
#                 fig=f)
