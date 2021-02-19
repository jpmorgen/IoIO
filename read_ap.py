#!/usr/bin/python3

import os
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import medfilt
from astropy.time import Time


# plt changed its plotting epoch, but astropy has not caught up
# [Thu Feb 11 11:15:38 2021 EST  jpmorgen@snipe fixed]
#pdconvert = 2440587.50000 - 1721424.5

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

#line = 'Na'
line = '[SII]'
onoff = 'AP'	# on-band minus off-band, fully reduced images
#onoff = 'On'	# on-band images after bias and dark subtraction and rayleigh calibration
#onoff = 'Off'	# off-band images after bias and dark subtraction and rayleigh calibration

ap_sum_fname = '/data/io/IoIO/reduced/ap_sum.csv'
#ap_sum_fname = '/data/io/IoIO/reduced/ap_sum_20190411+.csv'
#ap_sum_fname = '/data/io/IoIO/NO_BACKUP/reduced.previous_versions/202009_before_off_jup/ap_sum.csv'


# list of days in matplotlib plot_date format, which happen to start
# at midnight UT, which is the way I like it
# https://matplotlib.org/2.2.3/_modules/matplotlib/dates.html
pdlist = [] 
# complete list of rows
rlist = []
# Try to hack together a consistent calibration and
if line == 'Na':
    #ADU2R_recalib = 1
    #ADU2R_recalib = 1/2
    #ADU2R_recalib = 1/4
    #ADU2R_recalib = 0.4
    ADU2R_recalib = 0.41
    #ADU2R_recalib = 0.45  # Too low
    # --> long-term I need to change this in RedCorObs
    #on_loss_adjust = 0.5/0.8
    #on_loss_adjust = 0.6/0.8 # too high, clearly over-subtracting
    on_loss_adjust = 1
    #on_loss_adjust = 0.9
if line == '[SII]':
    ADU2R_recalib = 1/4.8
    on_loss_adjust = 1

# Notable times
T2019 = Time('2019-01-01T00:00:00', format='fits')
Twheel = Time('2019-03-27T00:00:00', format='fits')

# Read in file
with open(ap_sum_fname, newline='') as csvfile:
    csvr = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    fieldnames = csvr.fieldnames
    ap_keys = [k for k in fieldnames
               if ('AP' in k
                   or 'On' in k
                   or 'Off' in k)]
    for row in csvr:
        # Skip cal
        if row['EXPTIME'] < 60:
            continue
        # Skip RAOFF and DECOFF observations
        if 'Jupiter' in row['FNAME']:
            continue
        if row['LINE'] != line:
            continue
        T = Time(row['TMID'], format='fits')
        # Skip known bad files
        if (T < T2019
            and line == 'Na'
            and row['ADU2R'] < 0.18
            and (row['OFFSCALE'] < 1 or
                 row['OFFSCALE'] > 1.26)):
            continue
        if (T > T2019
            and line == 'Na'
            and row['ADU2R'] < 0.04):
            continue
        # If we made it here, we should be a good datapoint
        pdlist.append(T.plot_date)
        rlist.append(row)
        if T > Twheel:
            row['ADU2R'] /= ADU2R_recalib
        for ap in ap_keys:
            # Apply 2019 recalibration for filter wheel change
            if T > Twheel:
                row[ap] *= ADU2R_recalib
                if onoff == 'Off':
                    row[ap] *= on_loss_adjust
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
        #torus = ((row['CTS_' + onoff + 'Rjp30']
        #           - row['CTS_' + onoff + 'Rjp10'])
        #          / (30**2 - 10**2))
        torus = ((row['CTS_' + onoff + 'Rjp30']
                   - row['CTS_' + onoff + 'Rjp5'])
                  / (30**2 - 5**2))
        # Add these to the row
        row[onoff + 'back'] = back
        row[onoff + 'fore'] = fore
        row[onoff + 'center'] = center
        row[onoff + 'torus'] = torus
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

print('Total number of observations: ', len(pds))
print('Total number of days: ', len(np.unique(idays)))

T0 = Time('2017-12-01T00:00:00', format='fits')
T1 = Time('2018-07-10T00:00:00', format='fits')

#good_idx = np.where(np.logical_and(T0.plot_date < pds, pds < T1.plot_date))
#good_idx = np.where(pds < T1.plot_date)
#print(len(np.squeeze(good_idx)))
#print(len(pds))

# Compute daily medians and linear regressions between apertures
median_ap_list = []
for id in list(set(idays)):
    this_day_idx = np.where(idays == id)
    this_day_idx = this_day_idx[0]
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
mpds = np.asarray(mpds)

######## UNCOMMENT APPROPRIATE BLOCK TO CREATE DESIRED FIGURE

######## Time series of primary apertures.
# CHANGE onoff ABOVE TO PLOT FOR ON-BAND and OFF-BAND images 
plt.plot_date(mpds, 
              [row[onoff + 'Rjp15'] for row in median_ap_list], '^')
plt.plot_date(mpds, 
              [row[onoff + 'Rjp30'] for row in median_ap_list], 's')
plt.plot_date(pds, 
              [row[onoff + 'Rjp30'] for row in rlist], 'k.', ms=1) #, alpha=0.2) # doesn't show up in eps
back = [row[onoff + 'back'] for row in median_ap_list]
plt.plot_date(mpds, back, 'x')
#back_mav = np.convolve(back, np.ones((N_med,))/N_med, mode='same')
#back_med = medfilt(back, N_med)
#plt.plot_date(mpds, back_med, ',', linestyle='-')
axes = plt.gca()
if onoff == 'AP':
#    axes.set_ylim([0, 1700])
#    axes.set_ylim([0, 2300])
    axes.set_ylim([0, 5000])
    ylabel = ''
else:
    if onoff == 'On':
        axes.set_ylim([0, 15000])
    else:
        axes.set_ylim([0, 2000])
    ylabel = onoff + '-band'
plt.legend(['Rj < 7.5 nightly median', 'Rj < 15 nightly median', 'Rj < 15 surface brightness', '20 < Rj < 25 nightly median'], ncol=2)
plt.xlabel('UT Date')
plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
plt.gcf().autofmt_xdate()  # orient date labels at a slant
plt.show()

##-## ######## Time series of your key of choice
##-## #key = 'ONBSUB'
##-## #key = 'OFFSCALE'
##-## key = 'ADU2R'
##-## plt.plot_date(pds, 
##-##               [row[key] for row in rlist], 'k.', ms=1) #, alpha=0.2) # doesn't show up in eps
##-## plt.xlabel('UT Date')
##-## plt.ylabel(f'{line} {key}')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.show()


##-## ######## Time series of 25 Rj aperture.
##-## # Subtract the estimated telluric sodium background
##-## back = [row[onoff + 'back'] - telluric_Na for row in median_ap_list]
##-## mpds = np.asarray(mpds)
##-## back = np.asarray(back)
##-## T0 = Time('2017-12-01T00:00:00', format='fits')
##-## T1 = Time('2018-07-10T00:00:00', format='fits')
##-## #good_idx = np.where(mpds > T0.plot_date)
##-## ## unwrap
##-## #mpds = mpds[good_idx]
##-## #back = back[good_idx]
##-## sorted_idx = np.argsort(mpds)
##-## mpds = mpds[sorted_idx]
##-## back = back[sorted_idx]
##-## plt.plot_date(mpds, back, 'C2x')
##-## back_med = medfilt(back, N_med)
##-## plt.plot_date(mpds, back_med, ',', linestyle='-')
##-## axes = plt.gca()
##-## legend_list = ['20 < Rj < 25 nightly median',
##-##                str(N_med) + '-day running median']
##-## if onoff == 'AP':
##-##     axes.set_ylim([-100, 250])
##-##     ylabel = ''
##-##     #axes.set_ylim([0, 225])
##-##     #T2 = Time('2018-01-10T00:00:00', format='fits')
##-##     #T3 = Time('2018-02-20T00:00:00', format='fits')
##-##     #plt.plot_date([T2.plot_date, T3.plot_date],
##-##     #              [25, 150], ',', linestyle='-')
##-##     #legend_list.append('linear extrapolation')
##-## else:
##-##     if onoff == 'On':
##-##         axes.set_ylim([0, 3000])
##-##     else:
##-##         axes.set_ylim([0, 1500])
##-##     ylabel = onoff + '-band'
##-## #axes.set_xlim([T0.plot_date, T1.plot_date])
##-## plt.legend(legend_list, loc='upper right')
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.savefig('Na_vs_T_25Rj_hires.png', transparent=True, dpi=1200)
##-## plt.show()

##-## ######## Time series of torus full aperture.
##-## #
##-## #torus = [row[onoff + 'torus'] for row in median_ap_list]
##-## #torus = [row[onoff + 'Rjp15'] for row in median_ap_list]
##-## torus = [row[onoff + 'Rjp50'] for row in median_ap_list]
##-## mpds = np.asarray(mpds)
##-## torus = np.asarray(torus)
##-## T0 = Time('2017-12-01T00:00:00', format='fits')
##-## #T0 = Time('2016-12-01T00:00:00', format='fits')
##-## T1 = Time('2018-07-10T00:00:00', format='fits')
##-## #good_idx = np.where(mpds > T0.plot_date)
##-## ## unwrap
##-## #mpds = mpds[good_idx]
##-## #torus = torus[good_idx]
##-## sorted_idx = np.argsort(mpds)
##-## mpds = mpds[sorted_idx]
##-## torus = torus[sorted_idx]
##-## plt.plot_date(mpds, torus, 'C1.')
##-## torus_med = medfilt(torus, N_med)
##-## plt.plot_date(mpds, torus_med, ',', linestyle='-')
##-## axes = plt.gca()
##-## legend_list = ['Torus full-aperture nightly median',
##-##                str(N_med) + '-day running median']
##-## if onoff == 'AP':
##-##     legend_list.append('linear extrapolation')
##-##     axes.set_ylim([0, 50])
##-##     #axes.set_ylim([0, 1000])
##-##     ylabel = ''
##-## else:
##-##     if onoff == 'On':
##-##         axes.set_ylim([0, 3000])
##-##     else:
##-##         axes.set_ylim([0, 1500])
##-##     ylabel = onoff + '-band'
##-## #axes.set_xlim([T0.plot_date, T1.plot_date])
##-## plt.legend(legend_list, loc='upper right')
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.savefig('Torus_vs_T_hires.png', transparent=True, dpi=1200)
##-## plt.show()


##-## ######## Time series of on-torus apertures
##-## 
##-## #ap = '5_7_1'
##-## ap = '5_7_3'
##-## plt.plot_date(pds, 
##-##               [row[onoff + '_IPT_east_' + ap] for row in rlist], 'C1.', ms=1)
##-## plt.plot_date(pds, 
##-##               [row[onoff + '_IPT_west_' + ap] for row in rlist], 'C2.', ms=1)
##-## east = [row[onoff + '_IPT_east_' + ap] for row in median_ap_list]
##-## west = [row[onoff + '_IPT_west_' + ap] for row in median_ap_list]
##-## mpds = np.asarray(mpds)
##-## east = np.asarray(east)
##-## west = np.asarray(west)
##-## #T0 = Time('2017-12-31T00:00:00', format='fits')
##-## #T0 = Time('2016-12-01T00:00:00', format='fits')
##-## #T1 = Time('2018-07-10T00:00:00', format='fits')
##-## good_idx = np.where(mpds > T0.plot_date)
##-## # unwrap
##-## mpds = mpds[good_idx]
##-## east = east[good_idx]
##-## west = west[good_idx]
##-## sorted_idx = np.argsort(mpds)
##-## mpds = mpds[sorted_idx]
##-## east = east[sorted_idx]
##-## west = west[sorted_idx]
##-## east_med = medfilt(east, N_med)
##-## west_med = medfilt(west, N_med)
##-## #plt.plot_date(mpds, east, 'C1.')
##-## #plt.plot_date(mpds, west, 'C2.')
##-## #plt.plot_date(mpds, east_med, ',', linestyle='-', color='C1')
##-## #plt.plot_date(mpds, west_med, ',', linestyle='--', color='C2')
##-## plt.plot_date(mpds, east, 'C1.')
##-## plt.plot_date(mpds, west, 'C2.')
##-## plt.plot_date(mpds, east_med, 'C1-')
##-## plt.plot_date(mpds, west_med, 'C2-')
##-## axes = plt.gca()
##-## legend_list = ['east',
##-##                'west',
##-##                str(N_med) + '-point running median',
##-##                str(N_med) + '-point running median']
##-## if onoff == 'AP':
##-##     #legend_list.append('linear extrapolation')
##-##     axes.set_ylim([-300, 300])
##-##     ylabel = ''
##-## else:
##-##     if onoff == 'On':
##-##         axes.set_ylim([-200, 200])
##-##     else:
##-##         axes.set_ylim([0, 1500])
##-##     ylabel = onoff + '-band'
##-## #axes.set_xlim([T0.plot_date, T1.plot_date])
##-## plt.legend(legend_list, loc='upper right')
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' ' + ylabel + ' Surface Brightness (R)')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.savefig('Torus_vs_T_hires.png', transparent=True, dpi=1200)
##-## plt.show()

######## Time series of torus annular aperture.
# Subtract the estimated telluric sodium background

torus = [row[onoff + 'torus'] for row in median_ap_list]
#torus = [row[onoff + 'Rjp15'] for row in median_ap_list]
#torus = [row[onoff + 'Rjp50'] for row in median_ap_list]
mpds = np.asarray(mpds)
torus = np.asarray(torus)
T0 = Time('2017-12-01T00:00:00', format='fits')
#T0 = Time('2016-12-01T00:00:00', format='fits')
T1 = Time('2018-07-10T00:00:00', format='fits')
good_idx = np.where(mpds > T0.plot_date)
# unwrap
mpds = mpds[good_idx]
torus = torus[good_idx]
sorted_idx = np.argsort(mpds)
mpds = mpds[sorted_idx]
torus = torus[sorted_idx]
plt.plot_date(mpds, torus, 'C1.')
torus_med = medfilt(torus, N_med)
plt.plot_date(mpds, torus_med, ',', linestyle='-')
axes = plt.gca()
legend_list = ['5 < Rj < 15 nightly median',
               str(N_med) + '-day running median']
if onoff == 'AP':
    legend_list.append('linear extrapolation')
    axes.set_ylim([0, 50])
    #axes.set_ylim([0, 1000])
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
plt.savefig('Torus_vs_T_hires.png', transparent=True, dpi=1200)
plt.show()


##-## ######### Check offsets and scaling for final reduced images
##-## ### Morgenthaler et al 2019 value for 2018
##-## if onoff == 'AP':
##-##     plt.plot_date(mpds, 
##-##                   [row[onoff + 'Rjp15'] - 900*ADU2R_adjust for row in median_ap_list], '^')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'Rjp30'] - 320*ADU2R_adjust) * 1.5 for row in median_ap_list], 's')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'back'] - 70*ADU2R_adjust) * 2.2 for row in median_ap_list], 'x')
##-##     axes = plt.gca()
##-##     #axes.set_ylim([-50, 700])
##-##     axes.set_ylim([-500, 7000])
##-##     plt.xlabel('UT Date')
##-##     plt.ylabel(line + ' Surface Brightness (R)')
##-##     plt.legend(['Rj < 7.5 nightly median - 900', '(Rj < 15 nightly median - 320) * 1.5', '(20 < Rj < 25 nightly median - 70) * 2.2'])
##-##     plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-##     plt.show()

##-## ######### Check offsets and scaling for final reduced images
##-## ### 2018
##-## if onoff == 'AP':
##-##     plt.plot_date(mpds, 
##-##                   [row[onoff + 'Rjp15'] - 950*ADU2R_adjust for row in median_ap_list], '^')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'Rjp30'] - 330*ADU2R_adjust) * 1.5 for row in median_ap_list], 's')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'back'] - 70*ADU2R_adjust) * 2.2 for row in median_ap_list], 'x')
##-##     axes = plt.gca()
##-##     #axes.set_ylim([-50, 700])
##-##     axes.set_ylim([-500, 7000])
##-##     plt.xlabel('UT Date')
##-##     plt.ylabel(line + ' Surface Brightness (R)')
##-##     plt.legend(['Rj < 7.5 nightly median - 900', '(Rj < 15 nightly median - 320) * 1.5', '(20 < Rj < 25 nightly median - 70) * 2.2'])
##-##     plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-##     plt.show()

##-## ######### Check offsets and scaling for final reduced images
##-## ### 2019
##-## if onoff == 'AP':
##-##     plt.plot_date(mpds, 
##-##                   [row[onoff + 'Rjp15'] - (730)*ADU2R_adjust for row in median_ap_list], '^')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'Rjp30'] - (280)*ADU2R_adjust) * 1 for row in median_ap_list], 's')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'back'] - (70)*ADU2R_adjust) * 1 for row in median_ap_list], 'x')
##-##     axes = plt.gca()
##-##     #axes.set_ylim([-50, 700])
##-##     axes.set_ylim([-500, 2000])
##-##     plt.xlabel('UT Date')
##-##     plt.ylabel(line + ' Surface Brightness (R)')
##-##     plt.legend(['Rj < 7.5 nightly median - 900', '(Rj < 15 nightly median - 320) * 1.5', '(20 < Rj < 25 nightly median - 70) * 2.2'])
##-##     plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-##     plt.show()

##-## ######### Check offsets and scaling for final reduced images
##-## ### 2020
##-## if onoff == 'AP':
##-##     plt.plot_date(mpds, 
##-##                   [row[onoff + 'Rjp15'] - (670)*ADU2R_adjust for row in median_ap_list], '^')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'Rjp30'] - (260)*ADU2R_adjust) * 1.5 for row in median_ap_list], 's')
##-##     plt.plot_date(mpds, 
##-##                   [(row[onoff + 'back'] - (60)*ADU2R_adjust) * 2.5 for row in median_ap_list], 'x')
##-##     axes = plt.gca()
##-##     #axes.set_ylim([-50, 700])
##-##     axes.set_ylim([-500, 2000])
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

##-## ###### Plot OFFSCALE
##-## 
##-## plt.plot_date(pds,
##-##               #[(2.1 - row['OFFSCALE'])*500 for row in rlist], 'g,')
##-##               [row['OFFSCALE'] for row in rlist])
##-## axes = plt.gca()
##-## #axes.set_ylim([0.75, 1.75])
##-## plt.xlabel('UT Date')
##-## plt.ylabel(line + ' OFFSCALE')
##-## plt.gcf().autofmt_xdate()  # orient date labels at a slant
##-## plt.show()

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


#plt.plot_date(pds, 
#              [row[onoff + 'back'] for row in rlist], '.')
#plt.plot_date(pds, 
#              [row[onoff + 'fore'] for row in rlist], '.')
#plt.plot_date(pds, 
#              [row[onoff + 'center'] for row in rlist], '.')
#plt.show()
#
#plt.plot_date(mpds, 
#              [row[onoff + 'center'] for row in median_ap_list], '.')
#plt.plot_date(mpds, 
#              [row[onoff + 'fore'] for row in median_ap_list], '.')
#plt.plot_date(mpds, 
#              [row[onoff + 'back'] for row in median_ap_list], '.')
#plt.show()


## plt.plot_date(pds,
##               [row[onoff + 'fore']
##                - row[onoff + 'back']
##                for row in rlist], 'k.', ms=1)
## plt.plot_date(mpds,
##               [row[onoff + 'fore']
##                - row[onoff + 'back']
##                   for row in median_ap_list], '.')
## plt.ylabel(line + ' foreground - background Surface Brightness (R)')
## plt.gcf().autofmt_xdate()  # orient date labels at a slant
## plt.show()
## plt.show()



#plt.plot_date(pds, 
#              [row[onoff + 'center']
#               - row[onoff + 'back']
#                  for row in rlist], '.')

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
