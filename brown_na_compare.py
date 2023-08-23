import numpy as np

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import QTable

from IoIO.utils import savefig_overwrite

from IoIO.na_nebula import plot_nightly_medians

t = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')

#sb = t['Na_sum_1_jupiterRad']/t['Na_area_1_jupiterRad']
#plt.plot(t['tavg'].datetime, sb.value, '.')
#plt.show()

#sb = t['biweight_largest_sub_annular_sb_0.2_jupiterRad']
#sb_err = t['biweight_largest_sub_annular_sb_0.2_jupiterRad']

plot_nightly_medians(t, min_av_ap_dist=0, max_av_ap_dist=5*u.R_jup,
                     max_sb=1000, show=True)

