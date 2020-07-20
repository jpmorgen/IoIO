#!/usr/bin/python3

"""
Module to reduce biases and darks in a directory
"""

import os
import psutil
import csv
from multiprocessing import Pool

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
from skimage import exposure

from astropy import units as u
from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time, TimeDelta

import ccdproc

from west_aux.west_aux import add_history
from ReduceCorObs import get_dirs

# CCD size to make sure we don't get the guider or binning
NAXIS1 = 2750
NAXIS2 = 2200
#
threads_per_core = 2

def calc_max_num_biases(max_mem_fraction=0.75,
                        num_processes=None,
                        logical=False):
    # Calculate how many bias images we can combine based on how many
    # CPUs and how much memory we want to use.  Default to using the
    # same fraction of the CPUs as the memory.  Default to using
    # "real" CPUs, not hyperthreads, since at least for reduction done
    # for Morgenthaler et al. 2019, I found going beyond the real CPU
    # limit didn't increase efficiency much.  Not psutil has better
    # facility with cpu_count than os.
    if num_processes is None:
        num_processes = max_mem_fraction * psutil.cpu_count(logical=logical)
    assert num_processes > 0
    # Let user do fraction to get at num_processes
    if num_processes == 0:
        psutil.cpu_count(logical=logical)
    if num_processes < 1:
        num_processes *= psutil.cpu_count(logical=logical)
    # The combiner doubles the size of a normal CCDData, which
    # consists of two double-precision images (primary plus error)
    # plus a single-byte mask image
    combiner_mem_per_image = 2 * NAXIS1 * NAXIS2 * (8*2 + 1)
    mem = psutil.virtual_memory()
    max_num_biases = (mem.available * max_mem_fraction /
                      combiner_mem_per_image / num_processes)
    return np.int(max_num_biases), num_processes

def bias_combine(directory=None,
                 collection=None,
                 outdir='/data/io/IoIO/reduced/bias_dark',
                 show=False,
                 temperature_tolerance=0.5,
                 max_num_biases=None):
    """Play with biases in a directory
    """
    if collection is None:
        if not os.path.isdir(directory):
            log.debug('Not a directory, skipping: ' + directory)
            return False
        # Speed things up considerably by globbing the bias fnames, of
        # which I only have two types: those recorded by ACP (Bias*)
        # and those recorded by MaxIm (Bias* and *_bias.fit)
        collection = ccdproc.ImageFileCollection(directory,
                                                 glob_include='Bias*')
        if collection.summary is None:
            collection = ccdproc.ImageFileCollection(directory,
                                                     glob_include='*_bias.fit')
        if collection.summary is None:
            subdir = os.path.join(directory, 'Calibration')
            if not os.path.isdir(subdir):
                return False
            return bias_combine(subdir)            
    if collection.summary is None:
        return False
    log.debug('found biases in ' + directory)
    if max_num_biases is None:
        # Make sure we don't overwhelm our memory
        max_num_biases, num_processes = calc_max_num_biases(num_processes=1)
    # Prepare to write output
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #bias_summary_fname = os.path.join(outdir, 'bias_summary.csv')
    
    # Find the distinct sets of temperatures within temperature_tolerance
    
    isbias = collection.summary['imagetyp'] == 'BIAS'
    ts = collection.summary['ccd-temp'][isbias]
    bias_fnames = collection.summary['file'][isbias]
    # ccd-temp is recorded as a string.  Convert it to a number so
    # we can sort +/- values properly
    ts = np.asarray(ts)
    # Get the sort indices so we can extract fnames in proper order
    tsort_idx = np.argsort(ts)
    ts = ts[tsort_idx]
    bias_fnames = bias_fnames[tsort_idx]
    # Spot jumps in t and translate them into slices into ts
    dts = ts[1:] - ts[0:-1]
    jump = np.flatnonzero(dts > temperature_tolerance)
    slice = np.append(0, jump+1)
    slice = np.append(slice, -1)
    for it in range(len(slice)-1):
        # Loop through each temperature set
        flist = [os.path.join(collection.location, f)
                 for f in bias_fnames[slice[it]:slice[it+1]]]
        lccds = [CCDData.read(f, unit=u.adu) for f in flist]
        stats = []
        jds = []
        this_ts = ts[slice[it]:slice[it+1]]
        for iccd, ccd in enumerate(lccds):
            hdr = ccd.header
            im = ccd.data
            s = im.shape
            # Spot guider biases and binned main camera biases.  Note
            # Pythonic C index ordering
            if s != (NAXIS2, NAXIS1):
                log.debug('bias wrong shape: ' + str(s))
                break
            # Spot biases recorded when too bright.
            m = np.asarray(s)/2 # Middle of CCD
            q = np.asarray(s)/4 # 1/4 point
            m = m.astype(int)
            q = q.astype(int)
            # --> check lowest y too, since large filters go all the
            # --> way to the edge See 20200428 dawn biases
            dark_patch = im[m[0]-50:m[0]+50, 0:100]
            light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
            mdp = np.median(dark_patch)
            mlp = np.median(light_patch)
            if (np.median(light_patch) - np.median(dark_patch) > 1):
                log.debug('bias recorded during light conditions: ' +
                          flist[iccd])
                log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
                continue
            tm = Time(hdr['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
            rdnoise = np.sqrt(np.median((im[1:] - im[0:-1])**2))
            stats.append({'time': tt,
                          'ccdt': this_ts[iccd],
                          'median': np.median(im),
                          'mean': np.mean(im),
                          'std': np.std(im),
                          'rdnoise': rdnoise,
                          'min': np.min(im),  
                          'max': np.max(im)})
        mean_t = np.mean(this_ts)
        if len(stats) < 3:
            log.debug('Not enough good biases found at CCDT = {} C in {}'.format(mean_t, collection.location))
            continue
        df = pd.DataFrame(stats)
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        this_ccdt = '{:.2f}'.format(mean_t)
        f = plt.figure(figsize=[8.5, 11])

        ax = plt.subplot(5, 1, 1)
        plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['max'], 'k.')
        plt.ylabel('max')

        ax = plt.subplot(5, 1, 2)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['median'], 'k.')
        plt.plot(df['time'], df['mean'], 'r.')
        plt.ylabel('median & mean')
        plt.legend(['median', 'mean'])

        ax=plt.subplot(5, 1, 3)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['min'], 'k.')
        plt.ylabel('min')

        ax=plt.subplot(5, 1, 4)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['std'], 'k.')
        plt.ylabel('std')

        ax=plt.subplot(5, 1, 5)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['rdnoise'], 'k.')
        plt.ylabel('rdnoise')

        plt.gcf().autofmt_xdate()


        fbase = os.path.join(outdir, this_dateb + '_ccdT_' + this_ccdt)
        plt.savefig((fbase + '_vs_time.png'), transparent=True)
        if show:
            plt.show()
        plt.close()
        # At the 0.5 deg level, there seems to be no correlation between T and bias level
        #plt.plot(df['ccdt'], df['mean'], 'k.')
        #plt.xlabel('ccdt')
        #plt.ylabel('mean')
        #plt.show()
        
        # Some nights were recorded with a bad ACP plan that recursively multiplied the number of biases
        if len(stats) > max_num_biases:
            log.debug('Limiting to {} the {} biases found at CCDT = {} C in {}'.format(max_num_biases, len(stats), mean_t, collection.location))
            # Thanks to https://note.nkmk.me/en/python-pandas-list/ to
            # get this into a list.  The Series object works OK
            # internally to Pandas but not ouside
            best_idx = df['mean'].argsort().values.tolist()
            df = df.iloc[best_idx[max_num_biases:]]
            lccds = [lccds[i] for i in best_idx[max_num_biases:]]
            flist = [flist[i] for i in best_idx[max_num_biases:]]
            
        # Use ccdproc Combiner object iteratively as per example to
        # mask out bad pixels
        combiner = ccdproc.Combiner(lccds)
        old_n_masked = -1  # dummy value to make loop execute at least once
        new_n_masked = 0
        while (new_n_masked > old_n_masked):
            combiner.sigma_clipping(low_thresh=2, high_thresh=5,
                                    func=np.ma.median)
            old_n_masked = new_n_masked
            new_n_masked = combiner.data_arr.mask.sum()
            print(old_n_masked, new_n_masked)
        #Just one iteration for testing
        #combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
        # I prefer average to get sub-ADU accuracy.  Median outputs in
        # double-precision anyway, so it doesn't save space
        combined_average = combiner.average_combine()
        # Propagate our header from the last bias
        th = combined_average.header
        combined_average.header = hdr
        add_history(hdr, 'Combining NCOMBINE biases indicated in FILENN')
        combined_average.header.update(th)
        #combined_median = combiner.median_combine()
        im = combined_average
        # Prepare to write
        fname = fbase + '_combined_bias.fits'
        outHDUL = combined_average.to_hdu()
        hdr = outHDUL[0].header
        # Collect metadata, preparing to write to FITS header and CSV
        # file.  FITS wants a tuple, CSV wants a dict
        std =  np.asscalar(np.std(im).data   )
        med =  np.median(im)
        mean = np.asscalar(np.mean(im).data  )
        tmin = np.min(im)
        tmax = np.max(im)
        av_rdnoise = np.mean(df['rdnoise'])
        print(std, mean, med, tmin, tmax)
        #meta = []
        #row = {'FNAME': fname}
        hdr['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
        hdr['CCD-TEMP'] = (mean_t, 'average CCD temperature for combined biases')
        hdr['RDNOISE'] = (av_rdnoise, 'average readnoise for combined biases')
        hdr['STD'] = (std)
        hdr['MEDIAN'] = (med)
        hdr['MEAN'] = (mean)
        hdr['MIN'] = (tmin)
        hdr['MAX'] = (tmax)
        for i, f in enumerate(flist):
            hdr['FILE{0:02}'.format(i)] = f
        outHDUL.writeto(fname, overwrite=True)
        impl = plt.imshow(im, cmap=plt.cm.gray, filternorm=0,
                          interpolation='none', vmin=med-std, vmax=med+std)
        plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
        plt.savefig((fbase + '_combined_bias.png'), transparent=True)
        if show:
            plt.show()
        plt.close()


        #fieldnames = list(row.keys())
        #if not os.path.exists(bias_summary_fname):
        #    with open(bias_summary_fname, 'w', newline='') as csvfile:
        #        csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
        #                               quoting=csv.QUOTE_NONNUMERIC)
        #        csvdw.writeheader()
        #with open(bias_summary_fname, 'a', newline='') as csvfile:
        #    csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
        #                           quoting=csv.QUOTE_NONNUMERIC)
        #    csvdw.writerow(row)

def metadata(im, hdr):
    return hdr, row

def bias_subtract(fname, bias_fname, out_fname, show=False):
    try:
        im = CCDData.read(fname)
    except Exception as e: 
        im = CCDData.read(fname, unit="adu")
    bias = CCDData.read(bias_fname)
    im = ccdproc.subtract_bias(im, bias)
    # Gain is something I measured that agrees very well with manufacturer
    im = ccdproc.gain_correct(im, 0.3*u.electron/u.adu)
    std =  np.asscalar(np.std(im).data   )
    med =  np.median(im)
    mean = np.asscalar(np.mean(im).data  )
    tmin = np.min(im)
    tmax = np.max(im)
    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
    im.write(out_fname, overwrite=True)
    print(im.header)
    print('std, rdnoise, mean, med, min, max')
    print(std, rdnoise, mean, med, tmin, tmax)
    med = np.median(im[0:100, 0:100])
    std = np.asscalar(np.std(im[0:100, 0:100]).data)
    impl = plt.imshow(im, cmap=plt.cm.gray, filternorm=0,
                      interpolation='none', vmin=med-std, vmax=med+std)
    if show:
        plt.show()
    plt.close()

def bulk_bias_combine(directory='/data/io/IoIO/raw',
                      start=None, # Start and stop dates passed to get_dirs
                      stop=None,
                      **kwargs):
    print(calc_max_num_biases(**kwargs))
    return
    dirs = reversed(get_dirs(directory,
                             start=start,
                             stop=stop))
    with Pool(int(num_processes)) as p:
        p.map(bias_combine, dirs)


def bias_analyze(directory='/data/io/IoIO/reduced/bias_dark'):
    collection = ccdproc.ImageFileCollection(directory)
    s = collection.summary
    f = plt.figure(figsize=[8.5, 11])
    #good_idx = s['date-obs'] > '2020-04-17T00:00'
    #s = s[good_idx]
    plt.plot(s['ccd-temp'], s['median'], 'k.')
    plt.plot(s['ccd-temp'], s['mean'], 'r.')
    plt.xlabel('CCD Temperature (C)')
    plt.ylabel('median & mean (ADU)')
    plt.legend(['median', 'mean'])
    plt.show()
    plt.close

log.setLevel('DEBUG')
#bias_analyze()
#bulk_bias_combine()
bias_combine('/data/io/IoIO/raw/20200422', show=True)
#bias_combine('/data/io/IoIO/raw/20200421', show=True)
#bias_combine('/data/io/IoIO/raw/20200419', show=True)
#bias_combine('/data/io/IoIO/raw/20200417')#, show=True)
#bias_combine('/data/io/IoIO/raw/20200416')#, show=True)
#bias_combine('/data/io/IoIO/raw/20200408')#, show=True)
#bias_combine('/data/io/IoIO/raw/20200326/Calibration')
#bias_combine('/data/io/IoIO/raw/20191014')

#bias_subtract('/data/io/IoIO/raw/20200326/Calibration/Bias-S001-R001-C001-B1_dupe-1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-03-26_ccdT_-15.00_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)
#bias_subtract('/data/io/IoIO/raw/20200326/Calibration/Bias-S001-R001-C001-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-03-25_ccdT_9.46_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)
#
#bias_subtract('/data/io/IoIO/raw/20200326/Calibration/Dark-S001-R006-C001-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-03-25_ccdT_9.46_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)
#
#bias_subtract('/data/io/IoIO/raw/20200408/Dark-S002-R006-C001-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-04-08_ccdT_-15.89_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)


