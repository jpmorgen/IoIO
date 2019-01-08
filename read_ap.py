#!/usr/bin/python3

import os
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import medfilt
from astropy.time import Time

# More careful calculation detailed in Morgenthaler et al. 2019 ApJL
# suggests that ND filter is ~30% low and in MR, 15% light lost to
# Fraunhofer lines.  Net is 15% too low.  Correct here.
# --> Eventually the full correction will be absorbed into the pipeline
#\begin{equation}
#  \mathrm{ADU2R} = {on\_jup * ND \over MR}
#  \label{eq:ADU2R}
#\end{equation}
ADU2R_adjust = 1.15
telluric_Na = 55
N_med = 11

line = 'Na'
onoff = 'AP'	# on-band minus off-band, fully reduced images
#onoff = 'On'	# on-band images after bias and dark subtraction and rayleigh calibration
#onoff = 'Off'	# off-band images after bias and dark subtraction and rayleigh calibration

ap_sum_fname = '/data/io/IoIO/reduced/ap_sum.csv'

# list of days in matplotlib plot_date format, which happen to start
# at midnight UT, which is the way I like it
# https://matplotlib.org/2.2.3/_modules/matplotlib/dates.html
pdlist = [] 
# complete list of rows
rlist = []
# Read in file
with open(ap_sum_fname, newline='') as csvfile:
    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    fieldnames = csvr.fieldnames
    ap_keys = [k for k in fieldnames
               if ('AP' in k
                   or 'On' in k
                   or 'Off' in k)]
    for row in csvr:
        if row['LINE'] != line:
            continue
        if (line == 'Na'
            and row['ADU2R'] < 0.18
            and (row['OFFSCALE'] < 1 or
                 row['OFFSCALE'] > 1.26)):
            continue
        T = Time(row['TMID'], format='fits')
        pdlist.append(T.plot_date)
        rlist.append(row)
        for ap in ap_keys:
            # Apply ADU2R adjustment for sodium
            if row['LINE'] == 'Na':
                row[ap] *= ADU2R_adjust
            # compute counts so we can make annular apertures 
            ap_split = ap.split('Rjp')
            if len(ap_split) == 2:
                ap_size = int(ap_split[1])
                counts = row[ap] * ap_size**2
                row['CTS_' + ap] = counts
        # calculate specific apertures for plotting purposes
        back = ((row['CTS_' + onoff + 'Rjp50']
                 - row['CTS_' + onoff + 'Rjp40'])
                / (50**2 - 40**2))
        fore = ((row['CTS_' + onoff + 'Rjp30']
                 - row['CTS_' + onoff + 'Rjp15'])
                / (30**2 - 15**2))
        center = ((row['CTS_' + onoff + 'Rjp15']
                   - row['CTS_' + onoff + 'Rjp10'])
                  / (15**2 - 10**2))
        # Add these to the row
        row[onoff + 'back'] = back
        row[onoff + 'fore'] = fore
        row[onoff + 'center'] = center
        # Make sure we save a row with all the keys for the code below
        saverow = row

# supplement ap_keys with the CTS columns we just added 
ap_keys = [k for k in saverow.keys()
           if ('AP' in k
               or 'On' in k
               or 'Off' in k)]

# plt.plot_date list helps us get integer days which line up at
# nighttime for IoIO in Arizona
pds = np.asarray(pdlist)
idays = pds.astype(int)

# Compute daily medians and linear regressions between apertures
median_ap_list = []
for id in list(set(idays)):
    this_day_idx = np.where(idays == id)
    this_day_idx = this_day_idx[0]
    #print(this_day_idx)
    this_mpd = np.median(pds[this_day_idx])
    this_day_medians = {'TMID': this_mpd}
    for ap in ap_keys:
        this_day_ap_list = [rlist[i][ap] for i in this_day_idx]
        this_day_medians[ap] = np.median(this_day_ap_list)
    # I could pythonify this with some sort of loop
    this_back = [rlist[i][onoff + 'back']  for i in this_day_idx]
    this_fore = [rlist[i][onoff + 'fore']  for i in this_day_idx]
    this_center = [rlist[i][onoff + 'center']  for i in this_day_idx]
    ### # Sample scatter plots
    ### plt.scatter(this_back, this_fore)
    ### dt = datetime.fromordinal(id)
    ### plt.title(line + ' ' + dt.strftime('%Y-%m-%d'))
    ### plt.xlabel('20 < Rj < 25 surface brightness (R)')
    ### plt.ylabel('Rj < 15 surface brightness (R)')
    ### plt.show()
    
    # stick linear regression results onto end of median dict
    l = linregress(this_back, this_fore)
    add_dict = {onoff + 'back_fore_' + k : l[i]
                for i, k in enumerate(l._fields)}
    this_day_medians.update(add_dict)
    l = linregress(this_back, this_center)
    add_dict = {onoff + 'back_center_' + k : l[i]
                for i, k in enumerate(l._fields)}
    this_day_medians.update(add_dict)
    l = linregress(this_fore, this_center)
    add_dict = {onoff + 'fore_center_' + k : l[i]
                for i, k in enumerate(l._fields)}
    this_day_medians.update(add_dict)
    median_ap_list.append(this_day_medians)

mpds = [row['TMID'] for row in median_ap_list]

######## UNCOMMENT APPROPRIATE BLOCK TO CREATE DESIRED FIGURE

##-## ######## Time series of primary apertures.
##-## # CHANGE onoff ABOVE TO PLOT FOR ON-BAND and OFF-BAND images 
##-## plt.plot_date(mpds, 
##-##               [row[onoff + 'Rjp15'] for row in median_ap_list], '^')
##-## plt.plot_date(mpds, 
##-##               [row[onoff + 'Rjp30'] for row in median_ap_list], 's')
##-## plt.plot_date(pds, 
##-##               [row[onoff + 'Rjp30'] for row in rlist], 'k.', ms=1) #, alpha=0.2) # doesn't show up in eps
##-## back = [row[onoff + 'back'] for row in median_ap_list]
##-## plt.plot_date(mpds, back, 'x')
##-## #back_mav = np.convolve(back, np.ones((N_med,))/N_med, mode='same')
##-## #back_med = medfilt(back, N_med)
##-## #plt.plot_date(mpds, back_med, ',', linestyle='-')
##-## axes = plt.gca()
##-## if onoff == 'AP':
##-##     axes.set_ylim([0, 1700])
##-##     ylabel = ''
##-## else:
##-##     if onoff == 'On':
##-##         axes.set_ylim([0, 3000])
##-##     else:
##-##         axes.set_ylim([0, 1500])
##-##     ylabel = onoff + '-band'
##-## plt.legend(['Rj < 7.5 nightly median', 'Rj < 15 nightly median', 'Rj < 15 surface brightness', '20 < Rj < 25 nightly median'], ncol=2)
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.show()

######## Time series of 25 Rj aperture.
# Subtract the estimated telluric sodium background

back = [row[onoff + 'back'] - telluric_Na for row in median_ap_list]
mpds = np.asarray(mpds)
back = np.asarray(back)
T0 = Time('2017-12-01T00:00:00', format='fits')
T1 = Time('2018-07-10T00:00:00', format='fits')
good_idx = np.where(mpds > T0.plot_date)
# unwrap
mpds = mpds[good_idx]
back = back[good_idx]
sorted_idx = np.argsort(mpds)
mpds = mpds[sorted_idx]
back = back[sorted_idx]
plt.plot_date(mpds, back, 'C2x')
back_med = medfilt(back, N_med)
plt.plot_date(mpds, back_med, ',', linestyle='-')
axes = plt.gca()
legend_list = ['20 < Rj < 25 nightly median',
               str(N_med) + '-day running median']
if onoff == 'AP':
    T2 = Time('2018-01-10T00:00:00', format='fits')
    T3 = Time('2018-02-20T00:00:00', format='fits')
    plt.plot_date([T2.plot_date, T3.plot_date],
                  [25, 150], ',', linestyle='-')
    legend_list.append('linear extrapolation')
    axes.set_ylim([0, 225])
    ylabel = ''
else:
    if onoff == 'On':
        axes.set_ylim([0, 3000])
    else:
        axes.set_ylim([0, 1500])
    ylabel = onoff + '-band'
axes.set_xlim([T0.plot_date, T1.plot_date])
plt.legend(legend_list, loc='upper right')
plt.xlabel('UT Date')
plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
plt.gcf().autofmt_xdate()  # orient date labels at a slant
plt.show()

##-## ######### Check offsets and scaling for final reduced images
##-## if onoff == 'AP':
##-##     plt.plot_date(mpds, 
##-##                   [row[onoff + 'Rjp15'] - 900*ADU2R_adjust for row in median_ap_list], '^')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'Rjp30'] - 320*ADU2R_adjust) * 1.5 for row in median_ap_list], 's')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'back'] - 70*ADU2R_adjust) * 2.2 for row in median_ap_list], 'x')
##-##     axes = plt.gca()
##-##     axes.set_ylim([-50, 700])
##-##     plt.xlabel('UT Date')
##-##     plt.ylabel(line + ' Surface Brightness (R)')
##-##     plt.legend(['Rj < 7.5 nightly median - 900', '(Rj < 15 nightly median - 320) * 1.5', '(20 < Rj < 25 nightly median - 70) * 2.2'])
##-##     plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-##     plt.show()


##-## ####### Plot ADU2R
##-## plt.plot_date(pds,
##-##               [row['ADU2R'] for row in rlist])
##-## axes = plt.gca()
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' ADU2R')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.show()

##--## ###### Plot OFFSCALE
##--## 
##--## plt.plot_date(pds,
##--##               #[(2.1 - row['OFFSCALE'])*500 for row in rlist], 'g,')
##--##               [row['OFFSCALE'] for row in rlist])
##--## axes = plt.gca()
##--## axes.set_ylim([0.75, 1.75])
##--## plt.xlabel('UT Date')
##--## plt.ylabel(line + ' OFFSCALE')
##--## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##--## plt.show()

## ######### Plot results of scatter plot linear regression
## plt.plot_date(mpds, 
##               [row[onoff + 'back_fore_slope']
##                for row in median_ap_list], '.')
## plt.plot_date(mpds, 
##               [row[onoff + 'back_fore_rvalue']
##                for row in median_ap_list], '.')
## plt.plot_date(mpds, 
##               [row[onoff + 'back_fore_pvalue']
##                for row in median_ap_list], '.')
## plt.plot_date(mpds, 
##               [row[onoff + 'back_fore_stderr']
##                for row in median_ap_list], '.')
## axes = plt.gca()
## plt.legend(['slope', 'rvalue', 'pvalue', 'stderr'])
## plt.xlabel('UT Date')
## plt.ylabel('20 < Rj < 25 to Rj < 15 Scatter Plot Results')
## axes.set_ylim([0, 2])
## plt.gcf().autofmt_xdate()  # orient date labels at a slant
## plt.show()


# plt.plot_date(pds, 
#               [row[onoff + 'back'] for row in rlist], '.')
# plt.plot_date(pds, 
#               [row[onoff + 'fore'] for row in rlist], '.')
# plt.plot_date(pds, 
#               [row[onoff + 'center'] for row in rlist], '.')

#plt.plot_date(mpds, 
#              [row[onoff + 'center'] for row in median_ap_list], '.')


# plt.plot_date(pds, 
#               [row[onoff + 'fore']
#                - row[onoff + 'back']
#                   for row in rlist], '.')
# plt.plot_date(pds, 
#               [row[onoff + 'center']
#                - row[onoff + 'back']
#                   for row in rlist], '.')

#plt.plot_date(pds, 
#              [row[onoff + 'center']
#               - row[onoff + 'fore']
#                  for row in rlist], '.')


###plt.legend(['Rj < 7.5 nightly median', 'Rj < 15 nightly median', 'Rj < 15 surface brightness', '20 < Rj < 25 nightly median'])
###plt.xlabel('UT Date')
###plt.ylabel(line + ' Surface Brightness (R)')
###plt.gcf().autofmt_xdate()  # orient date labels at a slant
###axes = plt.gca()
###
###if line == '[SII]':
###    axes.set_ylim([-100, 1500])
###else:
###    axes.set_ylim([0, 1500])
####    axes.set_ylim([-100, 500])
###    #axes.set_ylim([0, 4000])
###    #axes.set_ylim([0.5, 1.75])
###    #axes.set_ylim([0, 0.5])
###plt.show()


#print(median_ap_list)
#print(pds)
#print([row[onoff + '_Rj_0'] for row in median_ap_list])
#plt.plot_date(pds, 
#              [row[onoff + 'Rj_0'] for row in rlist], '.')
#plt.plot_date(mpds, 
#              [row[onoff + 'Rj_0'] for row in median_ap_list], 's')
#plt.plot_date(pds, 
#              [row[onoff + 'Rjp40'] for row in rlist], '.')
#plt.plot_date(pds, 
#              [row[onoff + 'Rjp60'] for row in rlist], 's')
#plt.plot_date(pds, 
#              [row[onoff + 'Rjm40'] for row in rlist], 'x')
#plt.plot_date(mpds, 
#              [row[onoff + 'Rjp15'] for row in median_ap_list], '^')
#plt.plot_date(mpds, 
#              [row[onoff + 'Rjp30'] for row in median_ap_list], 's')
#plt.plot_date(pds, 
#              [row[onoff + 'Rjp30'] for row in rlist], 'k.', ms=1) #, alpha=0.2) # doesn't show up in eps
#plt.plot_date(mpds, 
#              [row[onoff + 'Rjm50'] for row in median_ap_list], 'x')
##c5 = np.asarray([row[onoff + 'Rjp5']*5**2 for row in median_ap_list])
##c10 = np.asarray([row[onoff + 'Rjp10']*10**2 for row in median_ap_list])
##c15 = np.asarray([row[onoff + 'Rjp15']*15**2 for row in median_ap_list])
##c30 = np.asarray([row[onoff + 'Rjp30']*30**2 for row in median_ap_list])
##c40 = np.asarray([row[onoff + 'Rjp40']*40**2 for row in median_ap_list])
##c50 = np.asarray([row[onoff + 'Rjp50']*50**2 for row in median_ap_list])
##c60 = np.asarray([row[onoff + 'Rjp60']*60**2 for row in median_ap_list])
###back = (c60 - c50) / (60**2 - 50**2)
###center = (c15 - c5) / (15**2 - 5**2)
##center = (c15 - c10) / (15**2 - 10**2)
##fore = (c30 - c15) / (30**2 - 15**2)
##back = (c50 - c40) / (50**2 - 40**2)
##plt.plot_date(mpds, center-back, '^')
##plt.plot_date(mpds, (fore-back), 's')
#plt.plot_date(mpds, back, 'x')
# It is wrong to subtract rates like this
#plt.plot_date(mpds, 
#              [(row[onoff + 'Rjp40']-row[onoff + 'Rjp60']) for row in median_ap_list], 'x')
#plt.plot_date(pds, 
#              [row[onoff + 'Rjp15'] for row in rlist], '.')
#plt.plot_date(mpds, 
#              [row[onoff + 'Rjp15'] for row in median_ap_list], 's')
#plt.plot_date(pds, 
#              [-row['DOFFBSUB'] for row in rlist], '^')
#plt.plot_date(mpds, 
#              [row[onoff + 'Rjm15'] for row in median_ap_list], 'x')
#plt.plot_date(pds,
#              #[(2 - row['OFFSCALE'])*500 for row in rlist], 'y^')
#              [row['OFFSCALE'] for row in rlist], 'y^')
#plt.plot_date(pds,
#              [1000*row['ADU2R'] for row in rlist], 'gs')
#plt.plot_date(mpds, 
#              [row['OnRjp60'] for row in median_ap_list], 'rs')
#plt.plot_date(mpds, 
#              [row['OffRjp60'] for row in median_ap_list], 'bs')
#plt.legend(['60 Rj box surface brightness', '60 Rj box nightly median', '>60 Rj box nightly median'])
#plt.legend(['30 Rj box surface brightness', '10 < Rj < 15 nightly median', '15 < Rj < 30 nightly median', '40 < Rj < 50 nightly median'])
#plt.legend(['10 < Rj < 15 nightly median', '15 < Rj < 30 nightly median', '40 < Rj < 50 nightly median'])

# plt.legend(['5 < Rj < 7.5 nightly median', '7.5 < Rj < 15 nightly median', '20 < Rj < 25 nightly median'])
# plt.xlabel('UT Date')
# plt.ylabel(line + ' Surf. Bright. (R)')
# plt.gcf().autofmt_xdate()  # orient date labels at a slant
# axes = plt.gca()
# #axes.set_yscale('log')
# if line == '[SII]':
#     axes.set_ylim([-100, 1500])
# else:
#     axes.set_ylim([0, 1500])
#     #axes.set_ylim([0, 4000])
#     #axes.set_ylim([0.5, 1.75])
#     #axes.set_ylim([0, 0.5])
# plt.show()


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
#-# plt.ylabel(line + ' Surf. Bright (R)')
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
