"""Get on-off ratios for [SII] and Na filters from flats"""

import os
import glob

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

from astropy.io.fits import getheader
from astropy.stats import mad_std, biweight_location
from astropy.time import Time

from cormultipipe import (CALIBRATION_ROOT, RedCorData,
                          CorMultiPipe, Calibration)

def flist_to_dict(flist):
    dlist = []
    for f in flist:
        bname = os.path.basename(f)
        s = bname.split('_')
        d = {'fname': f, 'date': s[0], 'band': s[1], 'onoff': s[2]}
        dlist.append(d)
    return dlist
    
def flat_flux(fname_or_ccd):
    print(fname_or_ccd)
    hdr = getheader(fname_or_ccd)
    #ccd = RedCorData.read(fname_or_ccd)
    maxval = hdr['FLATDIV']
    exptime = hdr['EXPTIME']
    flux = maxval/exptime
    return flux
    
c = Calibration(reduce=True)
#c = Calibration(start_date='2020-07-07', stop_date='2020-08-22', reduce=True)
#c = Calibration(start_date='2020-01-01', stop_date='2020-04-24', reduce=True)
#c = Calibration(start_date='2020-01-01', stop_date='2020-12-31', reduce=True)

on_list = glob.glob(os.path.join(CALIBRATION_ROOT, '*on_flat.fits'))
off_list = glob.glob(os.path.join(CALIBRATION_ROOT, '*off_flat.fits'))

on_dlist = flist_to_dict(on_list)
off_dlist = flist_to_dict(off_list)

ratio_dlist = []
for on_dict in on_dlist:
    date = on_dict['date']
    tm = Time(date, format='fits')
    band = on_dict['band']
    for off_dict in off_dlist:
        if off_dict['date'] != date:
            continue
        if off_dict['band'] != band:
            continue
        off_flux = flat_flux(off_dict['fname'])
        on_flux = flat_flux(on_dict['fname'])
        ratio = off_flux / on_flux
        ratio_dict = {'band': band,
                      'date': date,
                      'time': tm.tt.datetime,
                      'ratio': ratio}
        ratio_dlist.append(ratio_dict)

#print(ratio_dlist)
df = pd.DataFrame(ratio_dlist)

f = plt.figure(figsize=[8.5, 11])
plt.title('Sky flat ratios')
for ib, band in enumerate(['Na', 'SII']):
    plt.title('Sky flat ratios')
    this_band = df[df['band'] == band]
    med_ratio = np.median(this_band['ratio'])
    std_ratio = np.std(this_band['ratio'])
    biweight_ratio = biweight_location(this_band['ratio'])
    mad_std_ratio = mad_std(this_band['ratio'])
    print(f'{band} med {med_ratio:.2f} +/- {std_ratio:.2f}')
    print(f'{band} biweight {biweight_ratio:.2f} +/- {mad_std_ratio:.2f}')
    ax = plt.subplot(2, 1, ib+1)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    #plt.plot(df['time'], df['ratio'], 'k.')
    plt.plot(this_band['time'], this_band['ratio'], 'k.')
    plt.ylabel(f'{band} off/on ratio')
    plt.axhline(y=biweight_ratio, color='red')
    plt.text(0.5, biweight_ratio + 0.1*mad_std_ratio, 
             f'{biweight_ratio:.2f} +/- {mad_std_ratio:.2f}',
             ha='center', transform=ax.get_yaxis_transform())
    plt.axhline(y=biweight_ratio+mad_std_ratio,
                linestyle='--', color='k', linewidth=1)
    plt.axhline(y=biweight_ratio-mad_std_ratio,
                linestyle='--', color='k', linewidth=1)
    plt.ylim([biweight_ratio-3*mad_std_ratio, biweight_ratio+3*mad_std_ratio])
    plt.gcf().autofmt_xdate()

plt.savefig(ps.path.join(CALIBRATION_ROOT, 'flat__ratio_vs_time.png'), transparent=True)
show= True
if show:
    plt.show()
plt.close()

