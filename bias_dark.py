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
from scipy import signal, stats, interpolate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.transforms as transforms
import pandas as pd

from astropy import units as u
from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.time import Time, TimeDelta
from astropy.stats import mad_std, biweight_location
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils import Background2D, MedianBackground

import ccdproc as ccdp

from west_aux.west_aux import add_history
from ReduceCorObs import get_dirs
from IoIO import CorObsData

# Record in global variables Starlight Xpress Trius SX694 CCD
# characteristics.  Note that CCD was purchased in 2017 and is NOT the
# "Pro" version, which has a different gain but otherwise similar
# characteristics

sx694_camera_description = 'Starlight Xpress Trius SX694 mono, 2017 model version'

# 16-bit A/D converter, stored in SATLEVEL keyword
sx694_satlevel = 2**16-1
sx694_satlevel_comment = 'Saturation level (ADU)'

# Gain measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx.
# Value agrees well with Trius SX-694 advertised value (note, newer
# "PRO" model has a different gain value).  Stored in GAIN keyword
sx694_gain = 0.3
sx694_gain_comment = 'Measured gain (electron/ADU)'

# Readnoise measured as per ioio.notebk Tue Jul 10 12:13:33 2018 MCT
# jpmorgen@byted To be measured regularly as part of master bias
# creation.  Stored in RDNOISE keyword
# --> consider checking against this for sanity when
# constructing readnoises
sx694_example_readnoise = 15.475665 * sx694_gain
sx694_example_readnoise_comment = '2018-07-10 readnoise (electron)'

# Measurement in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx of
# when camera becomes non-linear.  Stored in NONLIN keyword.  Raw
# value of 42k was recorded with a typical overscan value.  Helps to
# remember ~40k is absolute max raw ADU to shoot for.  This is
# suspiciously close to the full-well depth in electrons of 17,000
# (web) - 18,000 (user's manual) provided by the manufacturer
# --> could do a better job of measuring the precise high end of this,
# since it could be as high as 50k
sx694_nonlin = 42000 - 1811
sx694_nonlin_comment = 'Measured nonlinearity point (ADU)'

# Exposure times at or below this value are counted on the camera and
# not in MaxIm.  There is a bug in the SX694 MaxIm driver seems to
# consistently add about sx694_exposure_correct seconds to the
# exposure time before asking the camera to read the CCD out.
# Measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx
# --> NEEDS TO BE VERIFIED WITH PHOTOMETRY FROM 2019 and 2020
# Corrected as part of local version of ccd_process
sx694_max_accurate_exposure = 0.7 # s
sx694_exposure_correct = 1.7 # s

def ccd_metadata(ccd,
                 camera_description=sx694_camera_description,
                 gain=sx694_gain,
                 gain_comment=sx694_gain_comment,
                 satlevel=sx694_satlevel,
                 satlevel_comment=sx694_satlevel_comment,
                 nonlin=sx694_nonlin,
                 nonlin_comment=sx694_nonlin_comment,
                 readnoise=sx694_example_readnoise,
                 readnoise_comment=sx694_example_readnoise_comment,
                 *args, **kwargs):
    """Record [SX694] CCD metadata for all the reductions"""
    if ccd.meta.get('camera') is not None:
        # We have been here before, so exit quietly
        return
    # Clean up double exposure time reference to avoid confusion
    if ccd.meta.get('exposure') is not None:
        del ccd.meta['EXPOSURE']
    ccd.meta.insert('INSTRUME',
                    ('CAMERA', camera_description),
                    after=True)
    ccd.meta['GAIN'] = (gain, gain_comment)
    # This gets used in ccdp.cosmicray_lacosmic
    ccd.meta['SATLEVEL'] = (satlevel, satlevel_comment)
    # This is where the CCD starts to become non-linear and is
    # used for things like rejecting flats recorded when
    # conditions were too bright
    ccd.meta['NONLIN'] = (nonlin, nonlin_comment)
    ccd.meta['RDNOISE'] = (readnoise, readnoise_comment)

def ccddata_read(fname_or_ccd, add_metadata=False, *args, **kwargs):
    """Convenience function to read a FITS file into a CCDData object.  

    Catches the case where the raw FITS file does not have a BUNIT
    keyword, which otherwise causes CCDData.read to crash.  In this
    case, ccddata_read assumes raw data are in units of ADU.  Optionally 
    supplements metadata with externally measured quantities such as
    gain, nonlinearity level, and readnoise

    Adds following FITS card if no BUNIT keyword present in metadata
        BUNIT = 'ADU' / physical units of the array values

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`

        If str, assumed to be a filename, which is read into a
        CCDData.  If ccddata, simply return the CCDData with BUNIT
        keyword possibly added

    add_metadata : bool
        If True, add [SX694] metadata.  Default: False
        
    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        CCDData with units set to ADU if none had been specified

    """
    if isinstance(fname_or_ccd, str):
        try:
            # This SOMETIMES fails if no units are specified
            ccd = CCDData.read(fname_or_ccd, *args, **kwargs)
        except Exception as e: 
            ccd = CCDData.read(fname_or_ccd, *args, unit="adu", **kwargs)
    else:
        ccd = fname_or_ccd
    assert isinstance(ccd, CCDData)
    if ccd.unit is None:
        log.warning('ccddata_read: CCDData.read read a file and did not assign any units to it.  Not sure why.  Setting units to ADU')
        ccd.unit = u.adu
    if ccd.meta.get('BUNIT') is None and ccd.unit is u.adu:
        # This comment is from the official FITS definition.  Not sure
        # why astropy stuff doesn't write it.  BUNIT is in the same
        # family is BZERO and BSCALE
        # https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html
        ccd.meta['BUNIT'] = ('ADU', 'physical units of the array values')
    if add_metadata:
        ccd_metadata(ccd, **kwargs)
    return ccd

def full_frame(im,
               naxis1 = 2750,
               naxis2 = 2200):
    """Returns true if image is a full frame, as defined by naxis1 and naxis2.

    Helps spot binned and guider images"""
    
    s = im.shape
    # Note Pythonic C index ordering
    if s != (naxis2, naxis1):
        return False
    return True    

def light_image(im, tolerance=2):
    
    """Returns True if light detected in image"""
    s = im.shape
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
    if (np.median(light_patch) - np.median(dark_patch) > 2):
        log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
        return True
    return False

def mask_above(ccd, key, margin=0.1):
    masklevel = ccd.meta[key]
    # Saturation level is subject to overscan subtraction and
    # multiplication by gain, so don't do strict = testing, but give
    # ourselves a little margin.
    mask = ccd.data >= masklevel - margin
    n_masked = np.count_nonzero(mask)
    if n_masked > 0:
        log.info('Masking {} pixels above {}'.format(n_masked, key))
    if len(key) > 6:
        h = 'HIERARCH '
    else:
        h = ''
    ccd.meta[h + 'N_' + key] \
        = (n_masked, 'number of pixels > {}'.format(key))
    if n_masked > 0:
        # Avoid creating a mask of all Falses
        ccd.mask = ccd.mask or mask
    return n_masked
    
#def mask_saturated(ccd):
#    """Record the number of saturated pixels in image metadata and log a warning if > 0"""
#    satlevel = ccd.meta['SATLEVEL']
#    # Saturation level is subject to overscan subtraction and multiplication by gain, so don't do strict = testing, but give ourselves a little margin.
#    mask = ccd.data >= satlevel - 0.1
#    n_saturated = np.count_nonzero(mask)
#    if n_saturated > 0:
#        log.warning('There are {} saturated pixels'.format(n_saturated))
#    ccd.meta['N_SAT'] \
#        = (n_saturated, 'number of saturated pixels')
#    return (n_saturated, mask)
#
#def mask_nonlin(ccd):
#    """Record the number of pixels > nonlinearity point in image metadata and log a warning if > 0"""
#    nonlin = ccd.meta['NONLIN']
#    mask = ccd.data > nonlin
#    n_nonlin = np.count_nonzero(mask)
#    if n_nonlin > 0:
#        log.warning('There are {} pixels > nonlinearity point'.format(n_nonlin))
#    ccd.meta['N_NONLIN'] \
#        = (n_nonlin, 'number of pixels > nonlinearity point')
#    return (n_nonlin, mask)

def fname_by_imagetyp_t_exp(directory=None,
                            collection=None,
                            subdirs=None,
                            imagetyp=None,
                            glob_include=None,
                            temperature_tolerance=0.5):
    """For a given IMAGETYP, returns a list of dictionaries with keys T (CCD-TEMP), EXPTIME, and fnames"""
    assert imagetyp is not None
    if subdirs is None:
        subdirs = []
    if glob_include is None:
        # Trick to make loop on glob_include, below, pass None to
        # ccdp.ImageFileCollection
        glob_include = [None]
    fdict_list = []
    if collection is None:
        # Prepare to call ourselves recursively to build up a list of
        # fnames in the provided directory and optional subdirectories
        if not os.path.isdir(directory):
            # This is the end of our recursive line
            return fdict_list
        for sd in subdirs:
            subdir = os.path.join(directory, sd)
            sub_fdict_list = fname_by_imagetyp_t_exp \
                (subdir,
                 imagetyp=imagetyp,
                 glob_include=glob_include,
                 temperature_tolerance=temperature_tolerance)
            for sl in sub_fdict_list:
                fdict_list.append(sl)
        # After processing our subdirs, process 'directory.'
        for gi in glob_include:
            # Speed things up considerably by allowing globbing.  As
            # per comment above, if None passed to glob_include, this
            # runs once with None passed to ccdp.ImageFileCollection's
            # glob_include
            collection = ccdp.ImageFileCollection(directory,
                                                  glob_include=gi)
            # Call ourselves recursively, but using the code below,
            # since collection is now defined
            gi_fdict_list = fname_by_imagetyp_t_exp \
                (collection=collection,
                 imagetyp=imagetyp,
                 temperature_tolerance=temperature_tolerance)
            for gi in gi_fdict_list:
                fdict_list.append(gi)
        # Here is the end of our recursive line if directory and
        # optional subdirs were specified
        return fdict_list
    if collection.summary is None:
        # We were probably called on a glob_include that yielded no results
        return fdict_list
    # If we made it here, we have a collection, possibly from calling
    # ourselves recursively
    our_imagetyp = collection.summary['imagetyp'] == imagetyp
    narrow_to_imagetyp = collection.summary[our_imagetyp]
    ts = narrow_to_imagetyp['ccd-temp']
    # ccd-temp is recorded as a string.  Convert it to a number so
    # we can sort +/- values properly
    ts = np.asarray(ts)
    # Get the sort indices so we can extract fnames in proper order
    tsort_idx = np.argsort(ts)
    ts = ts[tsort_idx]
    # Spot jumps in t and translate them into slices into ts
    dts = ts[1:] - ts[0:-1]
    jump = np.flatnonzero(dts > temperature_tolerance)
    tslices = np.append(0, jump+1)
    tslices = np.append(tslices, -1)
    fdict_list = []
    for it in range(len(tslices)-1):
        these_ts = ts[tslices[it]:tslices[it+1]]
        mean_t = np.mean(these_ts)
        # Create a new summary Table that inlcudes just these Ts
        narrow_to_t = narrow_to_imagetyp[tslices[it]:tslices[it+1]]
        exps = narrow_to_t['exptime']
        # These are sorted by increasing exposure time
        ues = np.unique(exps)
        for ue in ues:
            exp_idx = np.flatnonzero(exps == ue)
            files = narrow_to_t['file'][exp_idx]
            full_files = [os.path.join(collection.location, f) for f in files]
            fdict_list.append({'T': mean_t,
                               'EXPTIME': ue,
                               'fnames': full_files})
    return fdict_list

def bias_combine(directory=None,
                 collection=None,
                 subdirs=['Calibration'],
                 glob_include=['Bias*', '*_bias.fit'],
                 temperature_tolerance=0.5,
                 outdir='/data/io/IoIO/reduced/bias_dark',
                 show=False,
                 camera_description=sx694_camera_description,
                 gain=sx694_gain,
                 satlevel=sx694_satlevel,
                 readnoise=sx694_example_readnoise,
                 readnoise_tolerance=0.5, # units of readnoise
                 gain_correct=False):
    """Play with biases in a directory

    Parameters
    ----------
    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of ADU.  Default: False
    """
    fdict_list = \
        fname_by_imagetyp_t_exp(directory=directory,
                                collection=collection,
                                subdirs=subdirs,
                                imagetyp='BIAS',
                                glob_include=glob_include,
                                temperature_tolerance=temperature_tolerance)
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    if len(fdict_list) == 0:
        log.debug('No biases found in: ' + directory)
        return False
    # Loop through each member of our fdict_list, preparing summary
    # plot and combining biases
    for fdict in fdict_list:
        lccds = []
        stats = []
        jds = []
        for fname in fdict['fnames']:
            ccd = ccddata_read(fname, add_metadata=True)
            if not full_frame(ccd):
                log.debug('bias wrong shape: ' + fname)
                continue
            if light_image(ccd):
                log.debug('bias recorded during light conditions: ' +
                          fname)
                continue
            im = ccd.data
            # Create uncertainty image
            diffs2 = (im[1:] - im[0:-1])**2
            rdnoise = np.sqrt(biweight_location(diffs2))
            uncertainty = np.multiply(rdnoise, np.ones(im.shape))
            ccd.uncertainty = StdDevUncertainty(uncertainty)
            lccds.append(ccd)
            # Prepare to create a pandas data frame to track relevant
            # quantities
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            ccdt = ccd.meta['CCD-TEMP']
            tt = tm.tt.datetime
            jds.append(tm.jd)
            stats.append({'time': tt,
                          'ccdt': ccdt,
                          'median': np.median(im),
                          'mean': np.mean(im),
                          'std': np.std(im)*gain,
                          'rdnoise': rdnoise*gain,
                          'min': np.min(im),  
                          'max': np.max(im)})
            mean_t = fdict['T']
        if len(stats) < 3:
            log.debug('Not enough good biases found at CCDT = {} C in {}'.format(mean_t, collection.location))
            continue
        df = pd.DataFrame(stats)
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        this_ccdt = '{:.2f}'.format(mean_t)
        f = plt.figure(figsize=[8.5, 11])

        # In the absence of a formal overscan region, this is the best
        # I can do
        medians = df['median']
        overscan = np.mean(medians)

        ax = plt.subplot(5, 1, 1)
        plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        plt.plot(df['time'], df['max'], 'k.')
        plt.ylabel('max (ADU)')

        ax = plt.subplot(5, 1, 2)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=False)
        plt.plot(df['time'], df['median'], 'k.')
        plt.plot(df['time'], df['mean'], 'r.')
        plt.ylabel('median & mean (ADU)')
        plt.legend(['median', 'mean'])
        secax = ax.secondary_yaxis \
            ('right',
             functions=(lambda adu: (adu - overscan)*gain,
                        lambda e: e/gain + overscan))
        secax.set_ylabel('Electrons')

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

        # At the 0.5 deg level, there seems to be no correlation between T and bias level
        #plt.plot(df['ccdt'], df['mean'], 'k.')
        #plt.xlabel('ccdt')
        #plt.ylabel('mean')
        #plt.show()
            
        # Make sure outdir exists
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fbase = os.path.join(outdir, this_dateb + '_ccdT_' + this_ccdt)
        fname = fbase + '_combined_bias.fits'
        plt.savefig((fbase + '_vs_time.png'), transparent=True)
        if show:
            plt.show()
        plt.close()

        # Do a sanity check of readnoise
        av_rdnoise = np.mean(df['rdnoise'])            
        if (np.abs(av_rdnoise/sx694_example_readnoise - 1)
            > readnoise_tolerance):
            log.warning('High readnoise {}, skipping {}'.format(av_rdnoise, fname))
            continue

        # Go through each image and subtract the median, since that
        # value wanders as a function of ambient (not CCD)
        # temperature.  To use ccd.subtract, type must match type of
        # array.  And they are integers anyway

        # We need another list to keep track of the
        # overscan-subtracted ccds, since ccd.subtract returns a copy
        # which doesn't get put back into the original lccd.  And when
        # I tried to put it in the original lccd wierd things happened
        # anyway
        os_lccds = []
        for ccd, m in zip(lccds, medians):
            ccd = ccd.subtract(m*u.adu, handle_meta='first_found')
            os_lccds.append(ccd)
            lccds = os_lccds

        ### Use ccdproc Combiner object iteratively as per example to
        ### mask out bad pixels
        ##combiner = ccdp.Combiner(lccds)
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

        # Use ccdp.combine since it enables memory management by
        # breaking up images to smaller chunks (better than throwing
        # images away)
        mem = psutil.virtual_memory()
        im = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*0.6)
        ccd_metadata(im)
        if gain_correct:
            im = ccdp.gain_correct(im, gain*u.electron/u.adu)
            gain = 1
        mask_above(im, 'SATLEVEL')
        mask_above(im, 'NONLIN')
            
        # Collect image metadata.  For some reason, masked pixels
        # aren't ignored by std, etc. even though they output masked
        # arrays (which is annoying in its own right -- see example
        # commented mean).  So just create a new array, if needed, and
        # only put into it the good pixels
        if im.mask is None:
            tim = im
        else:
            tim = im.data[im.mask == 0]
        std =  np.std(tim)*gain
        med =  np.median(tim)*gain
        #mean = np.asscalar(np.mean(tim).data  )
        mean = np.mean(tim)*gain
        tmin = np.min(tim)*gain
        tmax = np.max(tim)*gain
        print('std, mean, med, tmin, tmax (electron)')
        print(std, mean, med, tmin, tmax)
        im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
        im.meta['CCD-TEMP'] = (mean_t, 'Average CCD temperature for combined biases')
        im.meta['RDNOISE'] = (av_rdnoise, 'Measured readnoise (electron)')
        im.meta['STD'] = (std, 'Standard deviation of image (electron)')
        im.meta['MEDIAN'] = (med, 'Median of image (electron)')
        im.meta['MEAN'] = (mean, 'Mean of image (electron)')
        im.meta['MIN'] = (tmin, 'Min of image (electron)')
        im.meta['MAX'] = (tmax, 'Max of image (electron)')
        im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (ADU)')
        im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
        im.meta['NCOMBINE'] = (len(lccds), 'Number of biases combined')
        # Record each filename
        for i, f in enumerate(fdict['fnames']):
            im.meta['FILE{0:02}'.format(i)] = f
        add_history(im.meta,
                    'Combining NCOMBINE biases indicated in FILENN')
        add_history(im.meta,
                    'SATLEVEL and NONLIN apply to pre-overscan subtraction')
        # Leave these large for fast calculations downstream and make
        # final results that primarily sit on disk in bulk small
        #im.data = im.data.astype('float32')
        #im.uncertainty.array = im.uncertainty.array.astype('float32')
        im.write(fname, overwrite=True)
        # Always display image in electrons
        impl = plt.imshow(im.multiply(gain), origin='lower', cmap=plt.cm.gray,
                          filternorm=0, interpolation='none',
                          vmin=med-std, vmax=med+std)
        plt.title('CCDT = {} C on {} (electrons)'.format(this_ccdt, this_dateb))
        plt.savefig((fbase + '_combined_bias.png'), transparent=True)
        if show:
            plt.show()
        plt.close()

def hist_of_im(im, binsize=1, show=False):
    """Returns a tuple of the histogram of image and index into *centers* of
bins."""
    # Code from west_aux.py, maskgen.
    # Histogram bin size should be related to readnoise
    hrange = (im.data.min(), im.data.max())
    nbins = int((hrange[1] - hrange[0]) / binsize)
    hist, edges = np.histogram(im, bins=nbins,
                               range=hrange, density=False)
    # Convert edges of histogram bins to centers
    centers = (edges[0:-1] + edges[1:])/2
    if show:
        plt.plot(centers, hist)
        plt.show()
        plt.close()
    return (hist, centers)

def overscan_estimate(ccd_in, meta=None, master_bias=None,
                      readnoise=sx694_example_readnoise,
                      gain=sx694_gain, binsize=None,
                      min_width=1, max_width=8, box_size=100,
                      min_hist_val=10,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    Uses the minimum of: (1) the first peak in the histogram of the
    image or (2) the minimum of the median of four boxes at the
    corners of the image.

    Works best if bias shape (particularly bias ramp) is subtracted
    first.  Will subtract bias if bias is supplied and has not been
    subtracted.

    Parameters
    ----------
    ccd_in : `~astropy.nddata.CCDData` or filename
        Image from which to extract overscan estimate

    meta : `astropy.io.fits.header` or None
        referece to metadata of ccd into which to write OVERSCAN_* cards

    master_bias : `~astropy.nddata.CCDData`, filename, or None
        Bias to subtract from ccd before estimate is calculated.
        Improves accruacy by removing bias ramp.  Bias can be in units
        of ADU or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

    readnoise : float
        If bias supplied, its value of the RDNOISE keyword is used
        Default = ``sx694_example_readnoise``.

    gain :  float
        If bias supplied, its value of the GAIN keyword is used
        Default = ``sx_gain``.

    binsize: float or None, optional
        The binsize to use for the histogram.  If None, binsize is 
        (readnoise in ADU)/4.  Default = None

    min_width : int, optional
        Minimum width peak to search for in histogram.  Keep in mind
        histogram bins are binsize ADU wide.  Default = 1

    max_width : int, optional
        See min_width.  Default = 8

    box_size : int
        Edge size of square box used to extract biweight median location
        from the corners of the image for this method of  overscan
        estimation.  Default = 100

    show : boolean
       Show image with min/max set to highlight overscan pixels and
       histogram with overscan chopped  histogram.  Default is False [consider making this boolean or name of plot file]

    """
    # Originally in IoIO.py as back_level
    ccd = ccddata_read(ccd_in, add_metadata=True)
    if meta is None:
        meta = ccd.meta
    # For now don't get fancy with unit conversion
    assert ccd.unit is u.adu
    if master_bias is None:
        bias = None
    elif isinstance(master_bias, CCDData):
        # Make a copy because we are going to mess with it
        bias = master_bias.copy()
    else:
        bias = ccddata_read(master_bias)
    if isinstance(bias, CCDData):
        readnoise = bias.meta['RDNOISE']
        gain = bias.meta['GAIN']
    if isinstance(bias, CCDData):
        # Make sure bias hasn't been subtracted before
        if ccd.meta.get('subtract_bias') is None:
            if bias.unit is u.electron:
                # Convert bias back to ADU for subtraction, if needed
                bias = bias.divide(gain*u.electron/u.adu)
            ccd = ccdp.subtract_bias(ccd, bias)
    elif ccd.meta.get('subtract_bias') is None:
        log.warning('overscan_estimate: bias has not been subtracted, which can lead to inaccuracy of overscan estimate')
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.
    # Corners method
    s = ccd.shape
    bs = box_size
    c00 = biweight_location(ccd[0:bs,0:bs])
    c10 = biweight_location(ccd[s[0]-bs:s[0],0:bs])
    c01 = biweight_location(ccd[0:bs,s[1]-bs:s[1]])
    c11 = biweight_location(ccd[s[0]-bs:s[0],s[1]-bs:s[1]])
    corners_method = min(c00, c10, c01, c11)
    # Histogram method.  The first peak is the bias, the second is the
    # ND filter.  Note that the 1.25" filters do a better job at this
    # than the 2" filters but with carefully chosen parameters, the
    # first small peak can be spotted.
    if binsize is None:
        # Calculate binsize based on readnoise in ADU, but oversample
        # by 4.  Note need to convert from Quantity to float
        binsize = readnoise/gain/4.
    im_hist, im_hist_centers = hist_of_im(ccd, binsize)
    # Note that after bias subtraction, there is sometimes some noise
    # at low counts.  We expect a lot of pixels in the histogram, so filter
    good_idx = np.flatnonzero(im_hist > min_hist_val)
    im_hist = im_hist[good_idx]
    im_hist_centers = im_hist_centers[good_idx]
    # The arguments to linspace are the critical parameters I played
    # with together with binsize to get the first small peak to be recognized
    im_peak_idx = signal.find_peaks_cwt(im_hist,
                                        np.linspace(min_width, max_width))
    hist_method = im_hist_centers[im_peak_idx[0]]
    overscan_methods = ['corners', 'histogram']
    overscan_values = np.asarray((corners_method, hist_method))
    meta['HIERARCH OVERSCAN_CORNERS'] = (corners_method, 'ADU')
    meta['HIERARCH OVERSCAN_HISTOGRAM'] = (hist_method, 'ADU')
    o_idx = np.argmin(overscan_values)
    overscan = overscan_values[o_idx]
    meta['HIERARCH OVERSCAN_METHOD'] = (overscan_methods[o_idx],
                                       'Method used for overscan estimation')
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9))
        ccds = ccd.subtract(1000*u.adu)
        range = 5*readnoise/gain
        vmin = overscan - range - 1000
        vmax = overscan + range - 1000
        ax1.imshow(ccds, origin='lower', cmap=plt.cm.gray, filternorm=0,
                   interpolation='none', vmin=vmin, vmax=vmax)
        ax1.set_title('Image minus 1000 ADU')
        ax2.plot(im_hist_centers, im_hist)
        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.axvline(overscan, color='r')
        # https://stackoverflow.com/questions/13413112/creating-labels-where-line-appears-in-matplotlib-figure
        # the x coords of this transformation are data, and the
        # y coord are axes
        trans = transforms.blended_transform_factory(
            ax2.transData, ax2.transAxes)
        ax2.set_title('Histogram')
        ax2.text(overscan+20, 0.05, overscan_methods[o_idx]
                 + ' overscan = {:.2f}'.format(overscan),
                 rotation=90, transform=trans,
                 verticalalignment='bottom')
        plt.show()
    return overscan

def subtract_overscan(ccd_in, *args, **kwargs):
    """Subtract overscan for IoIO coronagraph and add CCD metadata.
    Overscan value is subtracted from the SATLEVEL keyword

    This is a wrapper around overscan_estimate in case I want to make
    overscan estimation more complicated by linking files within a
    directory.  Note: ccdproc's native subtract_overscan function can't be
    used because it assumes the overscan region is specified by a
    simple rectangle.

    All processing except bias_combine needs to run through this
    point, so this is a good place to add our common metadata

    """
    ccd = ccddata_read(ccd_in, add_metadata=True)
    ccd_metadata(ccd)
    overscan = overscan_estimate(ccd, meta=ccd.meta, *args, **kwargs)
    ccd = ccd.subtract(overscan*u.adu, handle_meta='first_found')
    ccd.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'overscan value subtracted (ADU)')
    ccd.meta['HIERARCH SUBTRACT_OVERSCAN'] \
        = (True, 'Overscan has been subtracted')
    # Keep track of our precise saturation level
    satlevel = ccd.meta['SATLEVEL']
    satlevel -= overscan
    ccd.meta['SATLEVEL'] = satlevel # still in ADU
    ccd.meta
    return ccd

# Copy and tweak ccdp.ccd_process
#def ccd_process(ccd, oscan=None, trim=None, error=False, master_bias=None,
#                dark_frame=None, master_flat=None, bad_pixel_mask=None,
#                gain=None, readnoise=None, oscan_median=True, oscan_model=None,
#                min_value=None, dark_exposure=None, data_exposure=None,
#                exposure_key=None, exposure_unit=None,
#                dark_scale=False, gain_corrected=True):
def ccd_process(ccd, oscan=None, error=False, master_bias=None,
                gain=None, readnoise=None, dark_frame=None, master_flat=None,
                *args, **kwargs):
    """Perform basic processing on IoIO ccd data.  Uses ccd_process
    for all steps except overscan subtraction

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

    exposure_key : `~ccdp.Keyword`, str or None, optional
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

    # Put in our common metadata
    ccd_metadata(nccd)

    # Correct exposure time
    # --> REFINE THIS ESTIMATE BASED ON MORE MEASUREMENTS
    exptime = nccd.meta['EXPTIME']
    if exptime > sx694_max_accurate_exposure:
        nccd.meta.insert('EXPTIME', 
                         ('OEXPTIME', exptime, 'original exposure time (seconds)'),
                         after=True)
        exptime += sx694_exposure_correct
        nccd.meta['EXPTIME'] = (exptime, 'corrected exposure time (seconds)')
        nccd.meta.insert('OEXPTIME', 
                         ('HIERARCH EXPTIME_CORRECTION',
                          sx694_exposure_correct, '(seconds)'),
                         after=True)
        add_history(nccd.meta,
                    'Corrected exposure time for SX694 MaxIm driver bug')

    # Apply overscan correction unique to the IoIO SX694 CCD.  This
    # also adds our CCD metadata
    if oscan is True:
        nccd = subtract_overscan(nccd, master_bias=master_bias,
                                 *args, **kwargs)
    elif oscan is None or oscan is False:
        pass
    else:
        raise TypeError('oscan is not None, True or False')

    if master_bias is not None:
        nccd.meta['RDNOISE'] = master_bias.meta['RDNOISE']
        nccd.meta.comments['RDNOISE'] = master_bias.meta.comments['RDNOISE']
        # Extract values used for further ccd_process
        # --> document these True options in help
        if gain is True:
            gain = master_bias.meta['GAIN']
            if master_bias.unit == u.electron/u.adu:
                gain_corrected = True
        if error is True:
            readnoise = master_bias.meta['RDNOISE']

    # Correct our SATLEVEL and NONLIN units if we are going to gain
    # correct
    if gain is not None:
        nccd.meta['SATLEVEL'] = nccd.meta['SATLEVEL'] * gain
        nccd.meta.comments['SATLEVEL'] = 'saturation level (electron)'
        nccd.meta['NONLIN'] = nccd.meta['NONLIN'] * gain
        nccd.meta.comments['NONLIN'] = 'Measured nonlinearity point (electron)'
            
    # Create the error frame.  I can't trim my overscan, so there are
    # lots of pixels at the overscan level.  After overscan and bias
    # subtraction, many of them that are probably normal statitical
    # outliers are negative enough to overwhelm the readnoise in the
    # deviation calculation.  But I don't want the error estimate on
    # them to be NaN, since the error is really the readnoise.
    if error and gain is not None and readnoise is not None:
        nccd = ccdp.create_deviation(nccd, gain=gain*u.electron/u.adu,
                                     readnoise=readnoise*u.electron,
                                     disregard_nan=True)
    elif error and (gain is None or readnoise is None):
        raise ValueError(
            'gain and readnoise must be specified to create error frame.')
    
    # Make it convenient to just specify dark_frame to do dark
    # subtraction the way I want
    if isinstance(dark_frame, CCDData):
        exposure_key = ccdp.Keyword('EXPTIME', u.s)
        dark_scale = True
    else:
        exposure_key=None
        dark_scale=False

    nccd = ccdp.ccd_process(nccd, master_bias=master_bias,
                            gain=gain*u.electron/u.adu,
                            dark_frame=dark_frame,
                            exposure_key=exposure_key,
                            dark_scale=dark_scale,
                            *args, **kwargs)
    if master_flat is None:
        return nccd
    min_value = master_flat.meta['FLAT_CUT']
    nccd = ccdp.flat_correct(nccd, master_flat,
                             min_value=min_value, norm_value=1)
    # My flats look different that most, so just divide and call that
    # good enough
    #master_flat.mask=None
    #nccd = nccd.divide(master_flat, handle_meta='first_found')
    #nccd.meta['HIERARCH FLAT_CORRECT'] = True
    return nccd
    

    #### make a copy of the object
    ###nccd = ccd.copy()

    #### apply the overscan correction
    ###if isinstance(oscan, CCDData):
    ###    nccd = subtract_overscan(nccd, overscan=oscan,
    ###                             median=oscan_median,
    ###                             model=oscan_model)
    ###elif isinstance(oscan, str):
    ###    nccd = subtract_overscan(nccd, fits_section=oscan,
    ###                             median=oscan_median,
    ###                             model=oscan_model)
    ###elif oscan is None:
    ###    pass
    ###else:
    ###    raise TypeError('oscan is not None, a string, or CCDData object.')
    ###
    #### apply the trim correction
    ###if isinstance(trim, str):
    ###    nccd = trim_image(nccd, fits_section=trim)
    ###elif trim is None:
    ###    pass
    ###else:
    ###    raise TypeError('trim is not None or a string.')
    ###
    #### create the error frame
    ###if error and gain is not None and readnoise is not None:
    ###    nccd = create_deviation(nccd, gain=gain, readnoise=readnoise)
    ###elif error and (gain is None or readnoise is None):
    ###    raise ValueError(
    ###        'gain and readnoise must be specified to create error frame.')
    ###
    #### apply the bad pixel mask
    ###if isinstance(bad_pixel_mask, np.ndarray):
    ###    nccd.mask = bad_pixel_mask
    ###elif bad_pixel_mask is None:
    ###    pass
    ###else:
    ###    raise TypeError('bad_pixel_mask is not None or numpy.ndarray.')
    ###
    #### apply the gain correction
    ###if not (gain is None or isinstance(gain, Quantity)):
    ###    raise TypeError('gain is not None or astropy.units.Quantity.')
    ###
    ###if gain is not None and gain_corrected:
    ###    nccd = gain_correct(nccd, gain)
    ###
    #### subtracting the master bias
    ###if isinstance(master_bias, CCDData):
    ###    nccd = subtract_bias(nccd, master_bias)
    ###elif master_bias is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'master_bias is not None or a CCDData object.')
    ###
    #### subtract the dark frame
    ###if isinstance(dark_frame, CCDData):
    ###    nccd = subtract_dark(nccd, dark_frame, dark_exposure=dark_exposure,
    ###                         data_exposure=data_exposure,
    ###                         exposure_time=exposure_key,
    ###                         exposure_unit=exposure_unit,
    ###                         scale=dark_scale)
    ###elif dark_frame is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'dark_frame is not None or a CCDData object.')
    ###
    #### test dividing the master flat
    ###if isinstance(master_flat, CCDData):
    ###    nccd = flat_correct(nccd, master_flat, min_value=min_value)
    ###elif master_flat is None:
    ###    pass
    ###else:
    ###    raise TypeError(
    ###        'master_flat is not None or a CCDData object.')
    ###
    #### apply the gain correction only at the end if gain_corrected is False
    ###if gain is not None and not gain_corrected:
    ###    nccd = gain_correct(nccd, gain)
    ###
    ###return nccd

def dark_combine(directory=None,
                 collection=None,
                 subdirs=['Calibration'],
                 glob_include=['Dark*', '*_dark.fit'],
                 master_bias=None, # This is going to have to be True or something like that to trigger search for optimum
                 outdir='/data/io/IoIO/reduced/bias_dark', 
                 show=False,
                 temperature_tolerance=0.5,
                 mask_threshold=3): #Units of readnoise
    fdict_list = \
        fname_by_imagetyp_t_exp(directory=directory,
                                collection=collection,
                                subdirs=subdirs,
                                imagetyp='DARK',
                                glob_include=glob_include,
                                temperature_tolerance=temperature_tolerance)
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    if len(fdict_list) == 0:
            log.debug('No darks found in: ' + directory)
            return False
    log.debug('Darks found in ' + directory)

    # Find the distinct sets of temperatures within temperature_tolerance
    # --> doing this for the second time.  Might want to make it some
    # sort of function.  For darks I also need to collect times
    
    for fdict in fdict_list:
        lccds = []
        jds = []
        for fname in fdict['fnames']:
            ccd = ccddata_read(fname, add_metadata=True)
            if not full_frame(ccd):
                log.debug('dark wrong shape: ' + fname)
                continue
            if light_image(ccd):
                log.debug('dark recorded during light conditions: ' +
                          fname)
                continue
            ccd = ccd_process(ccd, oscan=True, master_bias=master_bias,
                              gain=True, error=True)
            lccds.append(ccd)
            # Get ready to capture the mean DATE-OBS
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
        mean_t = fdict['T']
        exptime = fdict['EXPTIME']
        if len(jds) < 3:
            log.debug('Not enough good darks found at CCDT = {} C EXPTIME = {} in {}'.format(mean_t, exptime, collection.location))
            continue
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        this_ccdt = '{:.2f}'.format(mean_t)
        fbase = '{}_ccdT_{}_exptime_{}s'.format(
            this_dateb, this_ccdt, exptime)

        mem = psutil.virtual_memory()
        im = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*0.6)
        mask_above(im, 'SATLEVEL')
        mask_above(im, 'NONLIN')

        # Create a mask that blanks out all our pixels that are just
        # readnoise.  Multiply this in as zeros, not a formal mask,
        # otherwise subsequent operations with the dark will mask out
        # all but the dark current-affected pixels!
        measured_readnoise = im.meta['RDNOISE']
        is_dark_mask = im.data > measured_readnoise * mask_threshold
        n_dark_pix = np.count_nonzero(is_dark_mask)
        im.meta['NDARKPIX'] \
            = (n_dark_pix, 'number of pixels with dark current')
        im = im.multiply(is_dark_mask, handle_meta='first_found')

        # Collect image metadata.  For some reason, masked pixels
        # aren't ignored by std, etc. even though they output masked
        # arrays (which is annoying in its own right -- see example
        # commented mean).  So just create a new array, if needed, and
        # only put into it the good pixels
        bad_mask = is_dark_mask == 0
        if im.mask is not None:
            bad_mask = bad_mask | im.mask
        # Flip bad mask around so we get only the dark pixels in the
        # linear range
        tim = im.data[bad_mask == 0]

        std =  np.std(tim)
        #std =  np.asscalar(std.data  )
        med =  np.median(tim)
        #med =  np.asscalar(med.data  )
        mean = np.mean(tim)
        #mean = np.asscalar(mean.data  )
        tmin = np.min(tim)
        tmax = np.max(tim)
        rdnoise = np.sqrt(np.median((tim[1:] - tim[0:-1])**2))
        print('combined dark statistics for ' + fbase)
        print('std, rdnoise, mean, med, min, max, n_dark_pix')
        print(std, rdnoise, mean, med, tmin, tmax, n_dark_pix)
        im.meta['STD'] = (std, 'Standard deviation of image (electron)')
        im.meta['MEDIAN'] = (med, 'Median of image (electron)')
        im.meta['MEAN'] = (mean, 'Mean of image (electron)')
        im.meta['MIN'] = (tmin, 'Min of image (electron)')
        im.meta['MAX'] = (tmax, 'Max of image (electron)')
        im.meta['NCOMBINE'] = (len(lccds), 'Number of darks combined')
        add_history(im.meta,
                    'Combining NCOMBINE biases indicated in FILENN')
        im.meta['HIERARCH MASK_THRESHOLD'] \
            = (mask_threshold, 'Units of readnoise')
        add_history(im.meta,
                    'Setting pixes below MASK_THRESHOLD to zero; prevents subtraction noise')
        # Record each filename
        for i, f in enumerate(fdict['fnames']):
            im.meta['FILE{0:02}'.format(i)] = f
        # Prepare to write
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fbase = os.path.join(outdir, fbase)
        fname = fbase + '_combined_dark.fits'
        # Leave these large for fast calculations downstream and make
        # final results that primarily sit on disk in bulk small
        #im.data = im.data.astype('float32')
        #im.uncertainty.array = im.uncertainty.array.astype('float32')
        im.write(fname, overwrite=True)
        if show:
            impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray,
                              filternorm=0, interpolation='none',
                              vmin=med-std, vmax=med+std)
            plt.show()
            plt.close()

def flat_combine(directory=None,
                 collection=None,
                 subdir='AutoFlat', # not a list
                 glob_include='*Flat*', # not a list
                 master_bias=None, # This is going to have to be True or something like that to trigger search for optimum
                 dark_frame=None, # similarly here
                 outdir='/data/io/IoIO/reduced/bias_dark', # --> change this
                 show=False,
                 flat_cut=0.75,
                 init_threshold=100, # units of readnoise
                 edge_mask=-40, # CorObsData parameter for ND filter coordinates
                 temperature_tolerance=0.5):
    if collection is None:
        if not os.path.isdir(directory):
            log.debug('No directory ' + directory)
            return False
        collection = ccdp.ImageFileCollection(directory,
                                              glob_include=glob_include)
    directory = collection.location
    if collection.summary is None:
        if subdir is not None:
            newdir = os.path.join(directory, subdir)
            return flat_combine(newdir,
                                subdir=None,
                                glob_include=glob_include,
                                master_bias=master_bias,
                                dark_frame=dark_frame, # similarly here
                                outdir=outdir,
                                show=show,
                                temperature_tolerance=temperature_tolerance)
        log.debug('No [matching] FITS files found in  ' + directory)
        return False
    # If we made it here, we have a collection with files in it
    filters = np.unique(collection.summary['filter'])
    for this_filter in filters:
        flat_fnames = collection.files_filtered(imagetyp='FLAT',
                                                filter=this_filter,
                                                include_path=True)
        lccds = []
        jds = []
        for fname in flat_fnames:
            ccd = ccddata_read(fname, add_metadata=True)
            ccd = ccd_process(ccd,
                              oscan=True,
                              master_bias=master_bias,
                              gain=True,
                              error=True,
                              dark_frame=dark_frame)
            # Use photutils.Background2D to smooth each flat and get a
            # good maximum value.  Mask edges and ND filter so as to
            # increase quality of background map
            mask = np.zeros(ccd.shape, bool)
            # Use the CorObsData ND filter stuff with a negative
            # edge_mask to blank out all of the fuzz from the ND filter cut
            obs_data = CorObsData(ccd.to_hdu(), edge_mask=edge_mask)
            mask[obs_data.ND_coords] = True
            rdnoise = ccd.meta['RDNOISE']
            mask[ccd.data < rdnoise * init_threshold] = True
            ccd.mask = mask
            bkg_estimator = MedianBackground()
            b = Background2D(ccd, 20, mask=mask, filter_size=5,
                             bkg_estimator=bkg_estimator)
            max_flat = np.max(b.background)
            if max_flat > ccd.meta['NONLIN']:
                log.warning('flat max value of {} too bright: {}'.format(
                    max_flat, fname))
                continue
            ccd.mask = None
            ccd = ccd.divide(max_flat, handle_meta='first_found')
            lccds.append(ccd)
            # Get ready to capture the mean DATE-OBS
            tm = Time(ccd.meta['DATE-OBS'], format='fits')
            tt = tm.tt.datetime
            jds.append(tm.jd)
        
        if len(jds) < 3:
            log.debug('Not enough good flats found for filter = {} in {}'.format(this_filter, collection.location))
            continue
        # Combine our flats
        mem = psutil.virtual_memory()
        im = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*0.6)
        # Interpolate over our ND filter
        hdul = ccd.to_hdu()
        obs_data = CorObsData(hdul, edge_mask=edge_mask)
        # Capture our ND filter metadata
        im.meta = hdul[0].header
        good_pix = np.ones(ccd.shape, bool)
        good_pix[obs_data.ND_coords] = False
        points = np.where(good_pix)
        values = im[points]
        xi = obs_data.ND_coords
        # Linear behaved much better
        nd_replacement = interpolate.griddata(points,
                                              values,
                                              xi,
                                              method='linear')
                                              #method='cubic')
        im.data[xi] = nd_replacement
        # Do one last smoothing and renormalization
        bkg_estimator = MedianBackground()
        b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
                         bkg_estimator=bkg_estimator)
        max_flat = np.max(b.background)
        im = im.divide(max_flat, handle_meta='first_found')
        im.mask = im.data < flat_cut
        im.meta['FLAT_CUT'] = (flat_cut, 'Value below which flat is masked')

        # Prepare to write
        tm = Time(np.mean(jds), format='jd')
        this_date = tm.fits
        this_dateb = this_date.split('T')[0]
        fbase = '{}_{}'.format(this_dateb, this_filter)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fbase = os.path.join(outdir, fbase)
        fname = fbase + '_flat.fits'
        im.write(fname, overwrite=True)
        if show:
            impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray)
            plt.show()


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
    collection = ccdp.ImageFileCollection(directory)
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
    
    #h = plt.hist(image.flatten(), bins=20, align='mid', 
    #             density=True, label="Dark frame");
    h = plt.hist(image.flatten(), bins=20000, align='mid', 
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

def quick_show(im):
    std =  np.std(im)
    med =  np.median(im)
    mean = np.mean(im)
    tmin = np.min(im)
    tmax = np.max(im)
    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
    print('image statistics')
    print('std, rdnoise, mean, med, min, max')
    print(std, rdnoise, mean, med, tmin, tmax)
    #impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
    #                  interpolation='none', vmin=med-std, vmax=med+std)
    impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray, filternorm=0,
                      interpolation='none', vmin=med-std, vmax=med+std)
    if isinstance(im, CCDData):
        plt.title(im.unit)
    else:
        plt.title('units not specified')    
    plt.show()
    plt.close()
    

log.setLevel('DEBUG')

# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
# master_bias = ccddata_read(bias_fname)
# #dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
# dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_10.0s_combined_dark.fits'
# dark_frame = ccddata_read(dark_fname)
# #flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-07_Na_on_flat.fits'
# #master_flat = ccddata_read(flat_fname)
# # --> remove this
# #master_flat.meta['FLAT_CUT'] = 0.75
# fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
# raw = ccddata_read(fname)
# ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
#                   gain=True, error=True, dark_frame=dark_frame)
# 
# readnoise = ccd.meta['RDNOISE']
# binsize = readnoise/4.
# im_hist, im_hist_centers = hist_of_im(ccd, binsize)
# min_hist_val=10
# min_width=10
# max_width=20
# good_idx = np.flatnonzero(im_hist > min_hist_val)
# im_hist = im_hist[good_idx]
# im_hist_centers = im_hist_centers[good_idx]
# # The arguments to linspace are the critical parameters I played
# # with together with binsize to get the first small peak to be recognized
# im_peak_idx = signal.find_peaks_cwt(im_hist,
#                                     np.linspace(min_width, max_width))
# print(im_hist_centers)
# sky = im_hist_centers[im_peak_idx[1]]
# print(sky) # expecting ~90
# mask = ccd.data < 0.25 * sky
# ccd.mask = mask
#ccd.write('/tmp/test.fits', overwrite=True)
#impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
#                      interpolation='none', vmin=0, vmax=500)
#plt.show()
#quick_show(ccd)

bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
master_bias = ccddata_read(bias_fname)
dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_10.0s_combined_dark.fits'
dark_frame = ccddata_read(dark_fname)
flat_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-07_Na_on_flat.fits'
master_flat = ccddata_read(flat_fname)
# --> remove this
#master_flat.meta['FLAT_CUT'] = 0.75
fname = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
raw = ccddata_read(fname)
ccd = ccd_process(raw, oscan=True, master_bias=master_bias,
                  gain=True, error=True, dark_frame=dark_frame,
                  master_flat=master_flat)
ccd.write('/tmp/test.fits', overwrite=True)
impl = plt.imshow(ccd, origin='upper', cmap=plt.cm.gray, filternorm=0,
                      interpolation='none', vmin=0, vmax=500)
plt.show()
#quick_show(ccd)


# bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
# master_bias = ccddata_read(bias_fname)
# dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
# dark_frame = ccddata_read(dark_fname)
# flat_combine('/data/io/IoIO/raw/2020-07-07/',
#              master_bias=master_bias,
#              dark_frame=dark_frame,
#              show=False)


#fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
#master_bias = ccddata_read(bias_fname)
#ccd = ccddata_read(fname)
#b = ccd_process(ccd, oscan=True, master_bias=master_bias,
#                gain=True, error=True)
#quick_show(b)


#dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
##dark_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.25_exptime_300.0s_combined_dark.fits'
#ccd = ccddata_read(dark_fname)
#rn = ccd.meta['RDNOISE']
#exptime = ccd.meta['EXPTIME']
#n_images = ccd.meta['NCOMBINE']
#plt.figure(figsize=(10, 8))
#image=ccd.data
##h = plt.hist(image.flatten(), bins=20000, align='mid', 
##             density=True, label="Dark frame");
#plot_dark_with_distributions(image, rn, 1E-4, exposure=exptime,
#                             show_poisson=True, show_gaussian=True,
#                             n_images=n_images)
##plt.xlim(-20, 30)
#plt.show()

#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=False)
#bias_combine('/data/io/IoIO/raw/20200711', show=False, gain_correct=False)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#bias = ccddata_read(bias_fname)
#dark_dir = '/data/io/IoIO/raw/20200711'
##dark_combine(dark_dir, master_bias=bias, show=True)
#dark_combine(dark_dir, master_bias=bias, show=False)


## 
## im = CCDData.read('/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits')
## data = im.data.copy()
## 
## im = im.uncertainty.array.copy()
## #im = data - im
#std =  np.std(im)
#med =  np.median(im)
##mean = np.asscalar(np.mean(im).data  )
#mean = np.mean(im)
#tmin = np.min(im)
#tmax = np.max(im)
#impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#                  interpolation='none', vmin=med-std, vmax=med+std)
#im.dtype
#plt.show()
#plt.close()

