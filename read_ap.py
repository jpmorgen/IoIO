#!/usr/bin/python3

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

from ReduceCorObs import get_dirs

data_root = '/data/io/IoIO'
ap_sum_fname = 'ap_sum.csv'

master_list = []
median_list = []
median_times = []

#line = '[SII]'
line = 'Na'

top = os.path.join(data_root, 'reduced')

for d in get_dirs(top):
    if (('cloudy' in d
         or 'marginal' in d
         or 'dew' in d
         or 'bad' in d)):
        continue
    dlist = []
    this_ap_sum_fname = os.path.join(d, ap_sum_fname)
    with open(this_ap_sum_fname, newline='') as csvfile:
        csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        # Collect all measurements for this day
        for row in csvr:
            dlist.append(row)
        # Put all measurements into master_list
        master_list.extend(dlist)
        # Separate out Na measurements for this day and take the median
        line_list = [row for row in dlist if row['LINE'] == line]
        #m = np.median([row['TOTAL'] for row in line_list])
        #m = np.median([row['TOTAL'] - row['AP_STRIP_300'] for row in line_list])
        #n = np.median([row['AP_STRIP_300'] for row in line_list])
        #m = m - n
        #m = np.median([row['AP_STRIP_300'] for row in line_list])
        m = np.median([row['AP_STRIP_600'] for row in line_list])
        #m = np.median([row['AP_STRIP_1200'] for row in line_list])
        median_list.append(m)
        median_times.append(dlist[0]['TMID'])
        

        
line_list = [row for row in master_list if row['LINE'] == line]
Tlist = [row['TMID'] for row in line_list]

T = Time(Tlist, format='fits')
TM = Time(median_times, format='fits')
#plt.plot_date(T.plot_date, [row['TOTAL'] - row['AP_STRIP_1200'] for row in line_list]),
#plt.plot_date(T.plot_date, [row['TOTAL'] for row in line_list]),
#plt.plot_date(T.plot_date, [row['AP_STRIP_300'] for row in line_list], 'v')
plt.plot_date(T.plot_date, [row['AP_STRIP_600'] for row in line_list], 's')
#plt.plot_date(T.plot_date, [row['AP_STRIP_1200'] for row in line_list], '*')
#plt.plot_date(T.plot_date, [row['DONBSUB'] for row in line_list], 'o')
#plt.plot_date(T.plot_date, [row['DOFFBSUB'] for row in line_list], '^')
plt.plot_date(TM.plot_date, median_list, 'rs')

plt.title(line + ' Full-frame Surface Brightness')
plt.legend(['Individual obs.', 'Nightly median'])
plt.xlabel('Date')
plt.ylabel(line + ' Surf. Bright (approx. R)')
plt.gcf().autofmt_xdate()  # orient date labels at a slant
axes = plt.gca()
if line == '[SII]':
    axes.set_ylim([0, 100])
else:
    axes.set_ylim([0, 300])
plt.show()



#this_ap_sum_fname = '/data/io/IoIO/reduced/2018-07-02/ap_sum.csv'
#dlist = []
#with open(this_ap_sum_fname, newline='') as csvfile:
#    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#    for row in csvr:
#        dlist.append(row)
#
#line_list = [row for row in dlist if row['LINE'] == 'Na']
#Tlist = [row['TMID'] for row in line_list]
#AP1200 = [row['AP_STRIP_1200'] for row in line_list]
#
#T = Time(Tlist, format='fits')
#plt.plot_date(T.plot_date, [row['TOTAL'] for row in line_list]),
#plt.plot_date(T.plot_date, [row['AP_STRIP_300'] for row in line_list], 'v')
#plt.plot_date(T.plot_date, [row['AP_STRIP_600'] for row in line_list], 's')
#plt.plot_date(T.plot_date, [row['AP_STRIP_1200'] for row in line_list], '*')
#plt.gcf().autofmt_xdate()  # orient date labels at a slant
#plt.show()

#jyear = np.linspace(2000, 2001, 20)
#t = Time(jyear, format='jyear', scale='utc')
#plt.plot_date(t.plot_date, jyear)
#plt.gcf().autofmt_xdate()  # orient date labels at a slant
##plt.draw()
#plt.show()
