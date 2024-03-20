import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import pandas as pd

from astropy.table import Table

from IoIO.juno import JunoTimes, PJAXFormatter

MME = '/data/sun/Jupiter_MME/JupiterMME.csv'

#mme = Table.read('/data/sun/Jupiter_MME/JupiterMME_single_hdr.csv')

mme=MME
fig=None
ax=None
top_axis=True
tlim = None
ylim = None

df = pd.read_csv(MME, delimiter=',',
                 comment='#')

datetime = df['Unnamed: 0'][2:]
datetime = pd.to_datetime(datetime)
p_dyn = df['ensemble.7'][2:]
p_dyn = p_dyn.astype(float)

fig, ax = plt.subplots()
ax.plot(datetime, p_dyn)
ax.set_xlabel('date')
ax.set_yscale('log')
ax.set_ylabel('solar wind dynamic pressure (nPa)') 

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

plt.show()
