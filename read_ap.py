#!/usr/bin/python3

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

from ReduceCorObs import get_dirs

line = 'Na'
#line = '[SII]'
ap_sum_fname = '/data/io/IoIO/reduced/ap_sum.csv'
#ap_sum_fname = '/data/io/IoIO/reduced.previous_versions/Rj_strip_sum/ap_sum.csv'
#ap_sum_fname = '/data/io/IoIO/reduced/2018-02-22/ap_sum.csv'

# list of days in matplotlib plot_date format, which happen to start
# at midnight UT, which is the way I like it
# https://matplotlib.org/2.2.3/_modules/matplotlib/dates.html
pdlist = [] 
# complete list of rows
rlist = []
# Read in file
with open(ap_sum_fname, newline='') as csvfile:
    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)    
    for row in csvr:
        if row['LINE'] != line:
            continue
        T = Time(row['TMID'], format='fits')
        pdlist.append(T.plot_date)
        rlist.append(row)
pds = np.asarray(pdlist)
idays = pds.astype(int)

ap_keys = [k for k in row.keys()
           if 'AP' in k]
# Compute daily medians
median_ap_list = []
for id in list(set(idays)):
    this_day_idx = np.where(idays == id)
    this_day_idx = this_day_idx[0]
    #print(this_day_idx)
    this_mpd = np.median(pds[this_day_idx])
    this_day_medians = {'TMID': this_mpd}
    for ap in ap_keys:
        #this_day_ap_list = [row[ap] for row in rlist[this_day_idx]]
        this_day_ap_list = [rlist[i][ap] for i in this_day_idx]
        this_day_medians[ap] = np.median(this_day_ap_list)
    median_ap_list.append(this_day_medians)

#print(median_ap_list)
#print(pds)
#print([row['AP_Rj_0'] for row in median_ap_list])
mpds = [row['TMID'] for row in median_ap_list]
#plt.plot_date(pds, 
#              [row['APRj_0'] for row in rlist], '.')
#plt.plot_date(mpds, 
#              [row['APRj_0'] for row in median_ap_list], 's')
#plt.plot_date(pds, 
#              [row['APRjp40'] for row in rlist], '.')
#plt.plot_date(pds, 
#              [row['APRjp60'] for row in rlist], 's')
#plt.plot_date(pds, 
#              [row['APRjm40'] for row in rlist], 'x')
plt.plot_date(pds, 
              [row['APRjp30'] for row in rlist], '.')
plt.plot_date(mpds, 
              [row['APRjp30'] for row in median_ap_list], 's')
plt.plot_date(mpds, 
              [row['APRjp15'] for row in median_ap_list], '^')
#plt.plot_date(mpds, 
#              [row['APRjm50'] for row in median_ap_list], 'x')
c15 = np.asarray([row['APRjp15']*15**2 for row in median_ap_list])
c50 = np.asarray([row['APRjp50']*50**2 for row in median_ap_list])
c40 = np.asarray([row['APRjp40']*40**2 for row in median_ap_list])
c60 = np.asarray([row['APRjp60']*60**2 for row in median_ap_list])
back = (c60 - c50) / (60**2 - 50**2)
plt.plot_date(mpds, back, 'x')
# It is wrong to subtract rates like this
#plt.plot_date(mpds, 
#              [(row['APRjp40']-row['APRjp60']) for row in median_ap_list], 'x')
#plt.plot_date(pds, 
#              [row['APRjp15'] for row in rlist], '.')
#plt.plot_date(mpds, 
#              [row['APRjp15'] for row in median_ap_list], 's')
#plt.plot_date(pds, 
#              [-row['DOFFBSUB'] for row in rlist], '^')
#plt.plot_date(mpds, 
#              [row['APRjm15'] for row in median_ap_list], 'x')
#plt.legend(['60 Rj box surface brightness', '60 Rj box nightly median', '>60 Rj box nightly median'])
plt.legend(['30 Rj box surface brightness', '30 Rj box nightly median', '15 Rj box nightly median', '50 < Rj < 60 nightly median'])
plt.xlabel('UT Date')
plt.ylabel(line + ' Surf. Bright. (approx. R)')
plt.gcf().autofmt_xdate()  # orient date labels at a slant
axes = plt.gca()
if line == '[SII]':
    axes.set_ylim([-100, 1500])
else:
    axes.set_ylim([0, 1500])
plt.show()


#-#         
#-# 
#-# master_list = []
#-# median_list = []
#-# center_list = []
#-# median_times = []
#-# back_list = []
#-# 
#-# #line = '[SII]'
#-# line = 'Na'
#-# 
#-# top = os.path.join(data_root, 'reduced')
#-# 
#-# for d in get_dirs(top):
#-#     if (('cloudy' in d
#-#          or 'marginal' in d
#-#          or 'dew' in d
#-#          or 'bad' in d)):
#-#         continue
#-#     dlist = []
#-#     this_ap_sum_fname = os.path.join(d, ap_sum_fname)
#-#     with open(this_ap_sum_fname, newline='') as csvfile:
#-#         csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#-#         # Collect all measurements for this day
#-#         for row in csvr:
#-#             dlist.append(row)
#-#         # Put all measurements into master_list
#-#         master_list.extend(dlist)
#-#         # Separate out Na measurements for this day and take the median
#-#         line_list = [row for row in dlist if row['LINE'] == line]
#-#         m = np.median([row['AP_Rj_0'] for row in line_list])
#-#         #c = np.median([row['APp300'] for row in line_list])
#-#         b = np.median([row['AP_Rjm50'] for row in line_list])
#-#         #b = np.median([row['APm1200'] for row in line_list])
#-#         #b = np.median([row['AP_Rjm70'] for row in line_list])
#-#         back_list.append(b)#*9)
#-#         #m -= b
#-#         #m = np.median([row['AP_0'] - row['AP_300'] for row in line_list])
#-#         #n = np.median([row['AP_300'] for row in line_list])
#-#         #m = m - n
#-#         #m = np.median([row['AP_300'] for row in line_list])
#-#         #m = np.median([row['AP_600'] for row in line_list])
#-#         #m = np.median([row['AP_1200'] for row in line_list])
#-#         #m = np.median([row['On_0'] for row in line_list])
#-#         #m = np.median([row['Off_0'] for row in line_list])
#-#         median_list.append(m)#*2)
#-#         center_list.append(c)
#-#         median_times.append(dlist[0]['TMID'])
#-#         
#-# 
#-#         
#-# line_list = [row for row in master_list if row['LINE'] == line]
#-# Tlist = [row['TMID'] for row in line_list]
#-# 
#-# T = Time(Tlist, format='fits')
#-# TM = Time(median_times, format='fits')
#-# #a = [row['APp300'] - row['APm1200'] for row in line_list]
#-# #a = [row['APp1200'] - row['APm1200'] for row in line_list]
#-# #a = [row['AP_0'] - row['APm1200'] for row in line_list]
#-# a = [row['AP_Rj_0'] for row in line_list]
#-# plt.plot_date(T.plot_date, a, '.')
#-# #plt.plot_date(T.plot_date, [row['APm1200'] for row in line_list]),
#-# #plt.plot_date(T.plot_date, [row['AP_0'] - row['AP_1200'] for row in line_list]),
#-# #plt.plot_date(T.plot_date, [row['AP_0'] for row in line_list]),
#-# #plt.plot_date(T.plot_date, [row['APp300'] for row in line_list]),
#-# #plt.plot_date(T.plot_date, [row['AP_300'] for row in line_list], 'v')
#-# #plt.plot_date(T.plot_date, [row['AP_600'] for row in line_list], 's')
#-# #plt.plot_date(T.plot_date, [row['AP_1200'] for row in line_list], '*')
#-# #plt.plot_date(T.plot_date, [row['DONBSUB'] for row in line_list], 'o')
#-# #plt.plot_date(T.plot_date, [row['DOFFBSUB'] for row in line_list], '^')
#-# #plt.plot_date(T.plot_date, [row['On_0'] for row in line_list], 's')
#-# #plt.plot_date(T.plot_date, [row['Off_0'] for row in line_list], 's')
#-# 
#-# #plt.plot_date(TM.plot_date, median_list, 'rs')
#-# #plt.plot_date(TM.plot_date, center_list, 'ys')
#-# #plt.plot_date(TM.plot_date, back_list, 'gs')
#-# 
#-# plt.plot_date(TM.plot_date, median_list, 's')
#-# plt.plot_date(TM.plot_date, back_list, 'x')
#-# 
#-# #plt.title(line + ' Full-frame Surface Brightness')
#-# plt.legend(['Full-frame surf. bright.', 'Full frame nightly median', 'Frame edge nightly median'])
#-# plt.xlabel('UT Date')
#-# plt.ylabel(line + ' Surf. Bright (approx. R)')
#-# plt.gcf().autofmt_xdate()  # orient date labels at a slant
#-# axes = plt.gca()
#-# if line == '[SII]':
#-#     axes.set_ylim([0, 100])
#-# else:
#-#     axes.set_ylim([0, 500])
#-# plt.show()
#-# 
#-# 
#-# 
#-# #this_ap_sum_fname = '/data/io/IoIO/reduced/2018-07-02/ap_sum.csv'
#-# #dlist = []
#-# #with open(this_ap_sum_fname, newline='') as csvfile:
#-# #    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
#-# #    for row in csvr:
#-# #        dlist.append(row)
#-# #
#-# #line_list = [row for row in dlist if row['LINE'] == 'Na']
#-# #Tlist = [row['TMID'] for row in line_list]
#-# #AP1200 = [row['AP_1200'] for row in line_list]
#-# #
#-# #T = Time(Tlist, format='fits')
#-# #plt.plot_date(T.plot_date, [row['AP_0'] for row in line_list]),
#-# #plt.plot_date(T.plot_date, [row['AP_300'] for row in line_list], 'v')
#-# #plt.plot_date(T.plot_date, [row['AP_600'] for row in line_list], 's')
#-# #plt.plot_date(T.plot_date, [row['AP_1200'] for row in line_list], '*')
#-# #plt.gcf().autofmt_xdate()  # orient date labels at a slant
#-# #plt.show()
#-# 
#-# #jyear = np.linspace(2000, 2001, 20)
#-# #t = Time(jyear, format='jyear', scale='utc')
#-# #plt.plot_date(t.plot_date, jyear)
#-# #plt.gcf().autofmt_xdate()  # orient date labels at a slant
#-# ##plt.draw()
#-# #plt.show()
#-# 
