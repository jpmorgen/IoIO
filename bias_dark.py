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
from scipy import signal, stats

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
from skimage import exposure

from astropy import units as u
from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.time import Time, TimeDelta
from astropy.stats import mad_std, biweight_location
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import ccdproc

from west_aux.west_aux import add_history
from ReduceCorObs import get_dirs
from IoIO import global_gain, global_readnoise, hist_of_im

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
                 gain=None):
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
    if gain is None:
        gain = global_gain
    log.debug('found biases in ' + directory)
    #if max_num_biases is None:
    #    # Make sure we don't overwhelm our memory
    #    max_num_biases, num_processes = calc_max_num_biases(num_processes=1)
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
            ccd.meta['BUNIT'] = ('ADU', 'Unit of pixel value')
            im = ccd.data
            s = im.shape
            # Create uncertainty as small as we can.  Keep in default
            # endian (little = <f4) even though it will eventually be
            # FITS-transformed to big endian (>f4)
            # Leave this big for now for speed, since there are not
            # many master biases
            uncertainty = np.multiply(global_readnoise, np.ones(s))#, dtype='f4')
            ccd.uncertainty = StdDevUncertainty(uncertainty)
            # Spot guider biases and binned main camera biases.  Note
            # Pythonic C index ordering
            if s != (NAXIS2, NAXIS1):
                log.debug('bias wrong shape: ' + str(s))
                continue
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
            # Prepare to create a pandas data frame to track relevant
            # quantities
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
            diffs2 = (im[1:] - im[0:-1])**2
            rdnoise = np.sqrt(biweight_location(diffs2))
            stats.append({'time': tt,
                          'ccdt': this_ts[iccd],
                          'median': np.median(im),
                          'mean': np.mean(im),
                          'std': np.std(im)*gain,
                          'rdnoise': rdnoise*gain,
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
        plt.ylabel('max (ADU)')

        ax = plt.subplot(5, 1, 2)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['median'], 'k.')
        plt.plot(df['time'], df['mean'], 'r.')
        plt.ylabel('median & mean (ADU)')
        plt.legend(['median', 'mean'])

        ax=plt.subplot(5, 1, 3)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['min'], 'k.')
        plt.ylabel('min (ADU)')

        ax=plt.subplot(5, 1, 4)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['std'], 'k.')
        plt.ylabel('std (electron)')

        ax=plt.subplot(5, 1, 5)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['rdnoise'], 'k.')
        plt.ylabel('rdnoise (electron)')

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
        
        ### Some nights were recorded with a bad ACP plan that
        ### recursively multiplied the number of biases
        ### -->
        ### https://mwcraig.github.io/ccd-as-book/02-04-Combine-bias-images-to-make-master.html
        ### suggests that there is a separate ccdproc.combine function
        ### that takes care of this with mem_limit
        ##if len(stats) > max_num_biases:
        ##    log.debug('Limiting to {} the {} biases found at CCDT = {} C in {}'.format(max_num_biases, len(stats), mean_t, collection.location))
        ##    # Thanks to https://note.nkmk.me/en/python-pandas-list/ to
        ##    # get this into a list.  The Series object works OK
        ##    # internally to Pandas but not ouside
        ##    best_idx = df['mean'].argsort().values.tolist()
        ##    df = df.iloc[best_idx[:max_num_biases]]
        ##    lccds = [lccds[i] for i in best_idx[:max_num_biases]]
        ##    flist = [flist[i] for i in best_idx[:max_num_biases]]
            
        # Go through each image and subtract the median, since that
        # value wanders as a function of ambient (not CCD)
        # temperature.  To use ccd.subtract, type must match type of
        # array.  And they are integers anyway
        medians = df['median']
        overscan = np.mean(medians)
        # We need another list to keep track of the
        # overscan-subtracted ccds, since ccd.subtract returns a copy
        # which doesn't get put back into the original lccd.  And when
        # I tried to put it in the original lccd wierd things happened
        # anyway
        os_lccds = []
        for ccd, m in zip(lccds, medians):
            #ccd.data = ccd.data - m
            ccd = ccd.subtract(m*u.adu, handle_meta='first_found')
            os_lccds.append(ccd)
        lccds = os_lccds

        ### Use ccdproc Combiner object iteratively as per example to
        ### mask out bad pixels
        ##combiner = ccdproc.Combiner(lccds)
        ##old_n_masked = -1  # dummy value to make loop execute at least once
        ##new_n_masked = 0
        ##while (new_n_masked > old_n_masked):
        ##    #combiner.sigma_clipping(low_thresh=2, high_thresh=5,
        ##    #                        func=np.ma.median)
        ##    # Default to 1 std and np.ma.mean for func
        ##    combiner.sigma_clipping()
        ##    old_n_masked = new_n_masked
        ##    new_n_masked = combiner.data_arr.mask.sum()
        ##    print(old_n_masked, new_n_masked)
        ###Just one iteration for testing
        ###combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
        ##combiner.sigma_clipping(low_thresh=1, high_thresh=1, func=np.ma.mean)
        ## I prefer average to get sub-ADU accuracy.  Median outputs in
        ## double-precision anyway, so it doesn't save space
        #combined_average = combiner.average_combine()

        # Use ccdproc.combine since it enables memory management by
        # breaking up images to smaller chunks (better than throwing
        # images away)
        mem = psutil.virtual_memory()
        im = \
            ccdproc.combine(lccds,
                            method='average',
                            sigma_clip=True,
                            sigma_clip_low_thresh=5,
                            sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median,
                            sigma_clip_dev_func=mad_std,
                            mem_limit=mem.available*0.6)
        # This is ultimately where we set the gain for all the reductions
        im.meta['GAIN'] = (gain, 'Measured electrons per ADU')
        g = ccdproc.Keyword('GAIN', u.electron/u.adu)
        im = ccdproc.gain_correct(im, g)
        
        # Prepare to write
        fname = fbase + '_combined_bias.fits'
        # Collect metadata, preparing to write to FITS header and CSV
        # file.  FITS wants a tuple, CSV wants a dict
        # Note that std and mean behave differently on masked arrays,
        # returning a masked object themselves.  The output of
        # Combiner is a masked array.  The combine function does not
        # do that.  The later is appropriate for biases only
        #std =  np.asscalar(np.std(im).data   )
        std =  np.std(im)
        med =  np.median(im)
        #mean = np.asscalar(np.mean(im).data  )
        mean = np.mean(im)
        tmin = np.min(im)
        tmax = np.max(im)
        av_rdnoise = np.mean(df['rdnoise'])
        print('std, mean, med, tmin, tmax (electron)')
        print(std, mean, med, tmin, tmax)
        im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
        im.meta['CCD-TEMP'] = (mean_t, 'average CCD temperature for combined biases')
        im.meta['RDNOISE'] = (av_rdnoise, 'average measured readnoise')
        im.meta['STD'] = (std, 'Standard deviation of image')
        im.meta['MEDIAN'] = (med, 'Median of image')
        im.meta['MEAN'] = (mean, 'Mean of image')
        im.meta['MIN'] = (tmin, 'Min of image')
        im.meta['MAX'] = (tmax, 'Max of image')
        im.meta['OVERSCAN'] = (overscan, 'Average of raw bias medians (ADU)')
        #im.meta['NCOMBINE'] = (len(lccds), 'Number of biases combined')
        # Record each bias filename
        for i, f in enumerate(flist):
            im.meta['FILE{0:02}'.format(i)] = f
        # Prepare to write
        fname = fbase + '_combined_bias.fits'
        add_history(im.meta,
                    'Combining NCOMBINE biases indicated in FILENN')
        # Leave these large for fast calculations downstream and make
        # final results that primarily sit on disk in bulk small
        #im.data = im.data.astype('float32')
        #im.uncertainty.array = im.uncertainty.array.astype('float32')
        im.write(fname, overwrite=True)
        impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
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

def bias_closest_T():
    pass

def overscan_estimate(im, bias=None, flat=None):
    """Estimate overscan in the absense of a formal overscan region"""
    CCD beyond filter
    # Originally in IoIO.py as back_level
    # Provide bias for more precise readnoise

    # --> eventually may want to make this or another function do this
    # for a whole directory vs time, like ReduceCorObs.Background So I
    # could make im an fname too and find the absolute path to make a
    # collection.  Collection could be passed....  Maybe do it
    # iteratively, where

    # --> OTOH, I would want to make sure that images with comparable
    # exposure time were used, since they include dark current
    assert flat is None, ('Code not written yet to use flat to isolate background pixels')

    if bias is None:
        rdnoise = global_readnoise
        im.meta['RDNOISE'] = (rdnoise, '(overscan_estimate) readnoise supplied by user')
    else:
        # Prefer measured reanoise
        rdnoise = bias.meta['RDNOISE']
        im.meta['RDNOISE'] = (rdnoise, 'average readnoise for combined biases')
    # Use the histogram technique to spot the bias level of the image.
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.  The ND filter provides a
    # similar peak after bias subutraction (or, rather, it is the
    # second such peak).  Note that the 1.25" filters do a better job
    # at this than the 2" filters
    im_hist, im_hist_centers = hist_of_im(im.data, rdnoise)
    im_peak_idx = signal.find_peaks_cwt(im_hist,
                                        np.linspace(rdnoise*0.6, rdnoise*4))
    overscan = im_hist_centers[im_peak_idx[0]]
    im.meta['OVERSCAN'] = (overscan, 'Est. overscan from first peak in image hist')
    return overscan

def bias_subtract(fname, bias_fname, out_fname, show=False,
                  overscan=None, gain=None):
    # Think about making overscan possibly a directory or collection
    # to trigger calculation
    try:
        im = CCDData.read(fname)
    except Exception as e: 
        im = CCDData.read(fname, unit="adu")
        im.meta['BUNIT'] = ('ADU', 'Unit of pixel value')
    if show:
        im_orig = im.copy()
    bias = CCDData.read(bias_fname)
    if overscan is None:
        overscan = overscan_estimate(im, bias)
    im = im.subtract(overscan*u.adu, handle_meta='first_found')
    im = ccdproc.subtract_bias(im, bias)
    if gain is None:
        gain = global_gain
    im.meta['GAIN'] = (gain, 'Measured electrons per ADU')
    g = ccdproc.Keyword('GAIN', u.electron/u.adu)
    im = ccdproc.gain_correct(im, g)
    im.meta['BUNIT'] = 'electron'
    im.meta['BIASNAME'] = bias_fname
    add_history(im.meta,
                'Subtracted OVERSCAN, BIASNAME, and gain corrected')
    #std =  np.asscalar(np.std(im).data   )
    std =  np.std(im)
    med =  np.median(im)
    #mean = np.asscalar(np.mean(im).data  )
    mean = np.mean(im)
    tmin = np.min(im)
    tmax = np.max(im)
    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
    im.write(out_fname, overwrite=True)
    #print(im.header)
    print('std, rdnoise, mean, med, min, max')
    print(std, rdnoise, mean, med, tmin, tmax)
    if show:
        im_orig = im_orig - overscan
        med = np.median(im_orig[0:100, 0:100])
        std = np.std(im_orig[0:100, 0:100])
        #impl = plt.imshow(im, cmap=plt.cm.gray, filternorm=0,
        #                  interpolation='none', vmin=med-std, vmax=med+std)
        norm = ImageNormalize(stretch=SqrtStretch())
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9))
        #ax1.imshow(im_orig, origin='lower', cmap='Greys_r', norm=norm)
        ax1.imshow(im_orig, origin='lower', cmap=plt.cm.gray, filternorm=0,
                   interpolation='none', vmin=med-std, vmax=med+std)
        ax1.set_title('Raw Data Minus Overscan')
        #ax2.imshow(im, origin='lower', cmap='Greys_r', norm=norm)
        med = np.median(im[0:100, 0:100])
        std = np.std(im[0:100, 0:100])
        ax2.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
                   interpolation='none', vmin=med-std, vmax=med+std)
        ax2.set_title('Bias Subtracted')
        plt.show()
        plt.close()
    return im
    
# Copy and tweak ccdproc.ccd_process
def ccd_process(ccd, oscan=None, trim=None, error=False, master_bias=None,
                dark_frame=None, master_flat=None, bad_pixel_mask=None,
                gain=None, readnoise=None, oscan_median=True, oscan_model=None,
                min_value=None, dark_exposure=None, data_exposure=None,
                exposure_key=None, exposure_unit=None,
                dark_scale=False, gain_corrected=True):
    """Perform basic processing on IoIO ccd data.

    The following steps can be included:

    * overscan correction (:func:`subtract_overscan`)
    * trimming of the image (:func:`trim_image`)
    * create deviation frame (:func:`create_deviation`)
    * gain correction (:func:`gain_correct`)
    * add a mask to the data
    * subtraction of master bias (:func:`subtract_bias`)
    * subtraction of a dark frame (:func:`subtract_dark`)
    * correction of flat field (:func:`flat_correct`)

    The task returns a processed `~astropy.nddata.CCDData` object.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Frame to be reduced.

    oscan : `~astropy.nddata.CCDData`, str or None, optional
        For no overscan correction, set to None. Otherwise provide a region
        of ccd from which the overscan is extracted, using the FITS
        conventions for index order and index start, or a
        slice from ccd that contains the overscan.
        Default is ``None``.

    trim : str or None, optional
        For no trim correction, set to None. Otherwise provide a region
        of ccd from which the image should be trimmed, using the FITS
        conventions for index order and index start.
        Default is ``None``.

    error : bool, optional
        If True, create an uncertainty array for ccd.
        Default is ``False``.

    master_bias : `~astropy.nddata.CCDData` or None, optional
        A master bias frame to be subtracted from ccd. The unit of the
        master bias frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    dark_frame : `~astropy.nddata.CCDData` or None, optional
        A dark frame to be subtracted from the ccd. The unit of the
        master dark frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    master_flat : `~astropy.nddata.CCDData` or None, optional
        A master flat frame to be divided into ccd. The unit of the
        master flat frame should match the unit of the image **after
        gain correction** if ``gain_corrected`` is True.
        Default is ``None``.

    bad_pixel_mask : `numpy.ndarray` or None, optional
        A bad pixel mask for the data. The bad pixel mask should be in given
        such that bad pixels have a value of 1 and good pixels a value of 0.
        Default is ``None``.

    gain : `~astropy.units.Quantity` or None, optional
        Gain value to multiple the image by to convert to electrons.
        Default is ``None``.

    readnoise : `~astropy.units.Quantity` or None, optional
        Read noise for the observations. The read noise should be in
        electrons.
        Default is ``None``.

    oscan_median : bool, optional
        If true, takes the median of each line. Otherwise, uses the mean.
        Default is ``True``.

    oscan_model : `~astropy.modeling.Model` or None, optional
        Model to fit to the data. If None, returns the values calculated
        by the median or the mean.
        Default is ``None``.

    min_value : float or None, optional
        Minimum value for flat field. The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.
        Default is ``None``.

    dark_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the dark image; if specified, must also provided
        ``data_exposure``.
        Default is ``None``.

    data_exposure : `~astropy.units.Quantity` or None, optional
        Exposure time of the science image; if specified, must also provided
        ``dark_exposure``.
        Default is ``None``.

    exposure_key : `~ccdproc.Keyword`, str or None, optional
        Name of key in image metadata that contains exposure time.
        Default is ``None``.

    exposure_unit : `~astropy.units.Unit` or None, optional
        Unit of the exposure time if the value in the meta data does not
        include a unit.
        Default is ``None``.

    dark_scale : bool, optional
        If True, scale the dark frame by the exposure times.
        Default is ``False``.

    gain_corrected : bool, optional
        If True, the ``master_bias``, ``master_flat``, and ``dark_frame``
        have already been gain corrected.  Default is ``True``.

    Returns
    -------
    occd : `~astropy.nddata.CCDData`
        Reduded ccd.

    Examples
    --------
    1. To overscan, trim and gain correct a data set::

        >>> import numpy as np
        >>> from astropy import units as u
        >>> from astropy.nddata import CCDData
        >>> from ccdproc import ccd_process
        >>> ccd = CCDData(np.ones([100, 100]), unit=u.adu)
        >>> nccd = ccd_process(ccd, oscan='[1:10,1:100]',
        ...                    trim='[10:100, 1:100]', error=False,
        ...                    gain=2.0*u.electron/u.adu)
    """
    # make a copy of the object
    nccd = ccd.copy()

    # apply overscan correction unique to the IoIO SX694 CCD
    
    if isinstance(oscan, CCDData):
        nccd = subtract_overscan(nccd, overscan=oscan,
                                 median=oscan_median,
                                 model=oscan_model)
    elif isinstance(oscan, str):
        nccd = subtract_overscan(nccd, fits_section=oscan,
                                 median=oscan_median,
                                 model=oscan_model)
    elif oscan is None:
        pass
    else:
        raise TypeError('oscan is not None, a string, or CCDData object.')

    # apply the trim correction
    if isinstance(trim, str):
        nccd = trim_image(nccd, fits_section=trim)
    elif trim is None:
        pass
    else:
        raise TypeError('trim is not None or a string.')

    # create the error frame
    if error and gain is not None and readnoise is not None:
        nccd = create_deviation(nccd, gain=gain, readnoise=readnoise)
    elif error and (gain is None or readnoise is None):
        raise ValueError(
            'gain and readnoise must be specified to create error frame.')

    # apply the bad pixel mask
    if isinstance(bad_pixel_mask, np.ndarray):
        nccd.mask = bad_pixel_mask
    elif bad_pixel_mask is None:
        pass
    else:
        raise TypeError('bad_pixel_mask is not None or numpy.ndarray.')

    # apply the gain correction
    if not (gain is None or isinstance(gain, Quantity)):
        raise TypeError('gain is not None or astropy.units.Quantity.')

    if gain is not None and gain_corrected:
        nccd = gain_correct(nccd, gain)

    # subtracting the master bias
    if isinstance(master_bias, CCDData):
        nccd = subtract_bias(nccd, master_bias)
    elif master_bias is None:
        pass
    else:
        raise TypeError(
            'master_bias is not None or a CCDData object.')

    # subtract the dark frame
    if isinstance(dark_frame, CCDData):
        nccd = subtract_dark(nccd, dark_frame, dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
    elif dark_frame is None:
        pass
    else:
        raise TypeError(
            'dark_frame is not None or a CCDData object.')

    # test dividing the master flat
    if isinstance(master_flat, CCDData):
        nccd = flat_correct(nccd, master_flat, min_value=min_value)
    elif master_flat is None:
        pass
    else:
        raise TypeError(
            'master_flat is not None or a CCDData object.')

    # apply the gain correction only at the end if gain_corrected is False
    if gain is not None and not gain_corrected:
        nccd = gain_correct(nccd, gain)

    return nccd

###def dark_combine(directory=None,
###                 collection=None,
###                 outdir='/data/io/IoIO/reduced/bias_dark',
###                 show=False,
###                 temperature_tolerance=0.5):
###    if collection is None:
###        if not os.path.isdir(directory):
###            log.debug('Not a directory, skipping: ' + directory)
###            return False
###        # Speed things up considerably by globbing the bias fnames, of
###        # which I only have two types: those recorded by ACP (Bias*)
###        # and those recorded by MaxIm (Bias* and *_bias.fit)
###        collection = ccdproc.ImageFileCollection(directory,
###                                                 glob_include='Dark*')
###        if collection.summary is None:
###            collection = ccdproc.ImageFileCollection(directory,
###                                                     glob_include='*_dark.fit')
###        if collection.summary is None:
###            subdir = os.path.join(directory, 'Calibration')
###            if not os.path.isdir(subdir):
###                return False
###            return dark_combine(subdir)            
###    if collection.summary is None:
###        return False
###    log.debug('found biases in ' + directory)
###    if not os.path.exists(outdir):
###        os.mkdir(outdir)
###
###    # Find the distinct sets of temperatures within temperature_tolerance
###    # --> doing this for the second time.  Might want to make it some
###    # sort of function
###    
###    isdark = collection.summary['imagetyp'] == 'DARK'
###    ts = collection.summary['ccd-temp'][isdark]
###    dark_fnames = collection.summary['file'][isdark]
###    # ccd-temp is recorded as a string.  Convert it to a number so
###    # we can sort +/- values properly
###    ts = np.asarray(ts)
###    # Get the sort indices so we can extract fnames in proper order
###    tsort_idx = np.argsort(ts)
###    ts = ts[tsort_idx]
###    dark_fnames = dark_fnames[tsort_idx]
###    # Spot jumps in t and translate them into slices into ts
###    dts = ts[1:] - ts[0:-1]
###    jump = np.flatnonzero(dts > temperature_tolerance)
###    slice = np.append(0, jump+1)
###    slice = np.append(slice, -1)
###    for it in range(len(slice)-1):
###        # Loop through each temperature set
###        flist = [os.path.join(collection.location, f)
###                 for f in dark_fnames[slice[it]:slice[it+1]]]
###        lccds = [CCDData.read(f, unit=u.adu) for f in flist]
###        stats = []
###        jds = []
###        this_ts = ts[slice[it]:slice[it+1]]
###        for iccd, ccd in enumerate(lccds):
###            hdr = ccd.header
###            im = ccd.data
###            s = im.shape
###            # Spot guider darks and binned main camera biases.  Note
###            # Pythonic C index ordering
###            if s != (NAXIS2, NAXIS1):
###                log.debug('bias wrong shape: ' + str(s))
###                continue
###            # Spot darks recorded when too bright.
###            m = np.asarray(s)/2 # Middle of CCD
###            q = np.asarray(s)/4 # 1/4 point
###            m = m.astype(int)
###            q = q.astype(int)
###            # --> check lowest y too, since large filters go all the
###            # --> way to the edge See 20200428 dawn biases
###            dark_patch = im[m[0]-50:m[0]+50, 0:100]
###            light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
###            mdp = np.median(dark_patch)
###            mlp = np.median(light_patch)
###            # --> might need to adjust this tolerance
###            if (np.median(light_patch) - np.median(dark_patch) > 1):
###                log.debug('dark recorded during light conditions: ' +
###                          flist[iccd])
###                log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
###                continue



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

# From
# https://mwcraig.github.io/ccd-as-book/03-01-Dark-current-The-ideal-case.html 
def plot_dark_with_distributions(image, rn, dark_rate, 
                                 n_images=1,
                                 exposure=1,
                                 gain=1,
                                 show_poisson=True, 
                                 show_gaussian=True):
    """
    Plot the distribution of dark pixel values, optionally overplotting the expected Poisson and
    normal distributions corresponding to dark current only or read noise only.
    
    Parameters
    ----------
    
    image : numpy array
        Dark frame to histogram.
    
    rn : float
        The read noise, in electrons.
        
    dark_rate : float
        The dark current in electrons/sec/pixel.
    
    n_images : float, optional
        If the image is formed from the average of some number of dark frames then 
        the resulting Poisson distribution depends on the number of images, as does the 
        expected standard deviation of the Gaussian.
        
    exposure : float
        Exosure time, in seconds.
        
    gain : float, optional
        Gain of the camera, in electron/ADU.
        
    show_poisson : bool, optional
        If ``True``, overplot a Poisson distribution with mean equal to the expected dark
        counts for the number of images.
    
    show_gaussian : bool, optional
        If ``True``, overplot a normal distribution with mean equal to the expected dark
        counts and standard deviation equal to the read noise, scaled as appropiate for 
        the number of images.
    """
    
    h = plt.hist(image.flatten(), bins=20, align='mid', 
                 density=True, label="Dark frame");

    bins = h[1]
    
    expected_mean_dark = dark_rate * exposure / gain
    
    pois = stats.poisson(expected_mean_dark * n_images)

    pois_x = np.arange(0, 300, 1)

    new_area = np.sum(1/n_images * pois.pmf(pois_x))

    if show_poisson:
        plt.plot(pois_x / n_images, pois.pmf(pois_x) / new_area, 
                 label="Poisson dsitribution, mean of {:5.2f} counts".format(expected_mean_dark)) 

    if show_gaussian:
        # The expected width of the Gaussian depends on the number of images.
        expected_scale = rn / gain * np.sqrt(n_images)
        
        # Mean value is same as for the Poisson distribution 
        expected_mean = expected_mean_dark * n_images
        gauss = stats.norm(loc=expected_mean, scale=expected_scale)
        
        gauss_x = np.linspace(expected_mean - 5 * expected_scale,
                              expected_mean + 5 * expected_scale,
                              num=100)
        plt.plot(gauss_x / n_images, gauss.pdf(gauss_x) * n_images, label='Gaussian, standard dev is read noise in counts') 
        
    plt.xlabel("Dark counts in {} sec exposure".format(exposure))
    plt.ylabel("Fraction of pixels (area normalized to 1)")
    plt.grid()
    plt.legend()

def dark_combine(dark_fnames, overscan_fname=None, show=False):
    if overscan_fname is not None:
        o = CCDData.read(overscan_fname, unit=u.adu)
        overscan = np.median(o)
    else:
        overscan = 0
    # --> note scope violation of directory
    dark_fnames = [os.path.join(directory, f) for f in dark_fnames]
    satlevel = (np.uint16(-1) - overscan) * global_gain
    print('bias_level from bias immediately after dark sequence = ', overscan)
    print('saturation level (electrons) ', satlevel)
    lccds = [bias_subtract(d, bias_fname, '/tmp/test.fits', overscan=overscan)
         for d in dark_fnames]
    mem = psutil.virtual_memory()
    combined_average = \
        ccdproc.combine(lccds,
                        method='average',
                        sigma_clip=True,
                        sigma_clip_low_thresh=5,
                        sigma_clip_high_thresh=5,
                        sigma_clip_func=np.ma.median,
                        sigma_clip_dev_func=mad_std,
                        mem_limit=mem.available*0.6)
    im = combined_average
    std =  np.std(im)
    med =  np.median(im)
    mean = np.mean(im)
    tmin = np.min(im)
    tmax = np.max(im)
    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
    print('combined dark statistics')
    print('std, rdnoise, mean, med, min, max')
    print(std, rdnoise, mean, med, tmin, tmax)
    if show:
        impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
                          interpolation='none', vmin=med-std, vmax=med+std)
        plt.show()
        plt.close()
    return im
    

log.setLevel('DEBUG')

bias_combine('/data/io/IoIO/raw/20200711', show=False)

im = CCDData.read('/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits')
data = im.data.copy()

im = im.uncertainty.array.copy()
#im = data - im
std =  np.std(im)
med =  np.median(im)
#mean = np.asscalar(np.mean(im).data  )
mean = np.mean(im)
tmin = np.min(im)
tmax = np.max(im)
impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
                  interpolation='none', vmin=med-std, vmax=med+std)
im.dtype
plt.show()
plt.close()

## directory = '/data/io/IoIO/raw/20200711'
## bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
## 
## overscan_fname = os.path.join(directory, 'Bias-S005-R007-C001-B1.fts')
## dark_fnames = ['Dark-S005-R006-C001-B1.fts',
##                'Dark-S005-R006-C002-B1.fts',
##                'Dark-S005-R006-C003-B1.fts',
##                'Dark-S005-R006-C004-B1.fts',
##                'Dark-S005-R006-C005-B1.fts']               
## print("T100")
## T100 = dark_combine(dark_fnames, overscan_fname, show=True)
## overscan_fname = os.path.join(directory, 'Bias-S005-R008-C001-B1.fts')
## dark_fnames = ['Dark-S005-R007-C001-B1.fts',
##                'Dark-S005-R007-C002-B1.fts',
##                'Dark-S005-R007-C003-B1.fts']
## print("T300")
## T300 = dark_combine(dark_fnames, overscan_fname, show=True)
## overscan_fname = os.path.join(directory, 'Bias-S005-R009-C001-B1.fts')
## dark_fnames = ['Dark-S005-R008-C001-B1.fts']               
## print("T1000")
## T1000 = dark_combine(dark_fnames, overscan_fname, show=True)
## dscale = T100.multiply(3, handle_meta='first_found')
## im = T300.subtract(dscale)
## std =  np.std(im)
## med =  np.median(im)
## mean = np.mean(im)
## tmin = np.min(im)
## tmax = np.max(im)
## rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
## print('T300 - 3* T100')
## print('std, rdnoise, mean, med, min, max')
## print(std, rdnoise, mean, med, tmin, tmax)
## impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
##                   interpolation='none', vmin=med-std, vmax=med+std)
## plt.show()
## plt.close()
## dscale = T100.multiply(10, handle_meta='first_found')
## im = T1000.subtract(dscale)
## std =  np.std(im)
## med =  np.median(im)
## mean = np.mean(im)
## tmin = np.min(im)
## tmax = np.max(im)
## rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
## print('T1000 - 10* T100')
## print('std, rdnoise, mean, med, min, max')
## print(std, rdnoise, mean, med, tmin, tmax)
## impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
##                   interpolation='none', vmin=med-std, vmax=med+std)
## plt.show()
## plt.close()
## dscale = T300.multiply(10/3., handle_meta='first_found')
## im = T1000.subtract(dscale)
## std =  np.std(im)
## med =  np.median(im)
## mean = np.mean(im)
## tmin = np.min(im)
## tmax = np.max(im)
## rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
## print('T1000 - 3.33* T300')
## print('std, rdnoise, mean, med, min, max')
## print(std, rdnoise, mean, med, tmin, tmax)
## impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
##                   interpolation='none', vmin=med-std, vmax=med+std)
## plt.show()
## plt.close()
## 
## rdnoise = T1000.meta['RDNOISE']
## exptime = T1000.meta['EXPTIME']
## plot_dark_with_distributions(T1000.data, rdnoise, 0.002, exposure=exptime)




## 1000s at -5C shows only ~5 e- dark current but large std likely from
## cosmic rays
#dark_fname = os.path.join(directory, 'Dark-S005-R008-C001-B1.fts')
#print('self overscan-subtracted')
#dark = bias_subtract(dark_fname, bias_fname,
#                     '/tmp/test.fits')
#overscan_fname = os.path.join(directory, 'Bias-S005-R009-C001-B1.fts')
#o = CCDData.read(overscan_fname, unit=u.adu)
#overscan = np.median(o)
#print('bias_level from bias after read = ', overscan)
#dark = bias_subtract(dark_fname, bias_fname,
#                     '/tmp/test.fits', overscan=overscan)
#im = dark
#impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#                  interpolation='none', vmin=med-std, vmax=med+std)
#plt.show()
#plt.close()


#collection = ccdproc.ImageFileCollection(directory,
#                                         glob_include='Dark*')
#isdark = collection.summary['imagetyp'] == 'DARK'
#exps = collection.summary['EXPTIME'][isdark]
#dark_fnames = collection.summary['file'][isdark]



#c = CCDData.read('/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits')
#d = CCDData.read('/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts', unit=u.adu)
#
#ch = c.header
##print(ch)
#dh = d.header
#dh.update(ch)
#print(dh)
#print(c.header.update(d.header))

#print(d.meta.append(c.meta))

#result = c.subtract(10000.*u.adu)
#print(np.median(result))
##std =  np.asscalar(np.std(im).data   )
#std =  np.std(c).data

#bias_analyze()
#bulk_bias_combine()

#bias_combine('/data/io/IoIO/raw/20200711', show=True)
#bias_subtract('/data/io/IoIO/raw/20200711/Dark-S005-R008-C001-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)

#bias_subtract('/data/io/IoIO/raw/20200711/Bias-S005-R001-C005-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)


#bias_combine('/data/io/IoIO/raw/20200422', show=True)
#bias_subtract('/data/io/IoIO/raw/20200422/Bias-S002-R001-C001-B1.fts',
#bias_subtract('/data/io/IoIO/raw/20200422/Bias-S002-R002-C001-B1.fts',
#              '/data/io/IoIO/reduced/bias_dark/2020-04-22_ccdT_-20.25_combined_bias.fits',
#              '/tmp/test.fits',
#              show=True)

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


