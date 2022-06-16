#!/usr/bin/python3

"""IoIO CCD bias, dark and flat calibration system"""

import os
import re
import psutil
import datetime
import glob
import argparse
from pathlib import Path

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy import log
from astropy import units as u
from astropy.io.fits import Header, getheader
from astropy.table import QTable
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.stats import mad_std, biweight_location

from photutils import Background2D, MedianBackground
import ccdproc as ccdp

from bigmultipipe import WorkerWithKwargs, NestablePool
from bigmultipipe import assure_list, num_can_process, prune_pout

import IoIO.sx694 as sx694
from IoIO.utils import (Lockfile, get_dirs_dates, add_history,
                        im_med_min_max, savefig_overwrite)
from IoIO.cordata_base import CorDataBase, CorDataNDparams
from IoIO.cor_process import standardize_filt_name
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               MAX_NUM_PROCESSES, MAX_CCDDATA_BITPIX,
                               MAX_MEM_FRAC, COR_PROCESS_EXPAND_FACTOR,
                               ND_EDGE_EXPAND, CorMultiPipeBase,
                               CorArgparseHandler,
                               light_image, mask_above_key)


# These are MaxIm and ACP day-level raw data directories, respectively
CALIBRATION_ROOT = os.path.join(IoIO_ROOT, 'Calibration')
CALIBRATION_SCRATCH = os.path.join(CALIBRATION_ROOT, 'scratch')

# Lockfiles to prevent multiple upstream parallel processes from
# simultanously autoreducing calibration data
LOCKFILE = '/tmp/calibration_reduce.lock'

# Raw (and reduced) data are stored in directories by UT date, but
# some have subdirectories that contain calibration files.
CALIBRATION_SUBDIRS = ['Calibration', 'AutoFlat']

# Put the regular expressions for the biases, darks, and flats here so
# that they can be found quickly without having to do a ccd.Collection
# on a whold directory.  The later is the rock solid most reliable,
# but slow in big directories, since ccdproc.Collection has to read
# each file
BIAS_GLOB = ['Bias*', '*_bias.fit']
DARK_GLOB = ['Dark*', '*_dark.fit']
FLAT_GLOB = '*Flat*'

# During the creation of master biases and darks files are grouped by
# CCD temperature.  This is the change in temperature seen as a
# function of time that is used to trigger the creation of a new group
DCCDT_TOLERANCE = 0.5
# During reduction of files, biases and darks need to be matched to
# each file by temperature.  This is the tolerance for that matching
CCDT_TOLERANCE = 2
# When I was experimenting with bias collection on a per-night basis,
# I got lots of nights with a smattering of biases.  Discard these
MIN_NUM_BIASES = 7
MIN_NUM_FLATS = 3

# Accept as match darks with this much more exposure time
DARK_EXP_MARGIN = 3

FLAT_CUT = 0.75
GRIDDATA_EXPAND_FACTOR = 20

# These are use to optimize parallelization until such time as
# ccdproc.combiner can be parallelized
NUM_CCDTS = int((35 - (-10)) / 5)
NUM_DARK_EXPTIMES = 8
NUM_FILTS = 9
NUM_CALIBRATION_FILES = 11

# Date after which flats were taken in a uniform manner (~60 degrees
# from Sun, ~2 hr after sunrise), leading to more stable results
# --> I am less convinced this is true, after looking at the early Na
# data
#STABLE_FLAT_DATE = '2020-01-01'
STABLE_FLAT_DATE = '1000-01-01'

# Tue Mar 01 07:59:13 2022 EST  jpmorgen@snipe
# Value on the date above for sanity checks
#FLAT_RATIO_CHECK_LIST = \
#    [{'band': 'Na',
#      'biweight_ratio': 4.752328783054392,
#      'mad_std_ratio': 0.1590846222357597,
#      'med_ratio': 4.716620841458621,
#      'std_ratio': 0.15049131587890988},
#     {'band': 'SII',
#      'biweight_ratio': 4.879239156541213,
#      'mad_std_ratio': 0.04906457267060454,
#      'med_ratio': 4.879915705039635,
#      'std_ratio': 0.06540676680077083}]

# Decide not to limit to just later data because earlier data is
# clearly higher.  May have a time dependence
FLAT_RATIO_CHECK_LIST = \
    [{'band': 'Na',
      'biweight_ratio': 4.81,
      'mad_std_ratio': 0.31,
      'med_ratio': 4.716620841458621, # Didn't bother with these
      'std_ratio': 0.15049131587890988},
     {'band': 'SII',
      'biweight_ratio': 4.80,
      'mad_std_ratio': 0.23,
      'med_ratio': 4.879915705039635, # Didn't bother with these
      'std_ratio': 0.06540676680077083}]

def jd_meta(ccd, bmp_meta=None, **kwargs):
    """CorMultiPipe post-processing routine to return JD.  Assumes PGData
    """
    if bmp_meta is not None:
        bmp_meta['jd'] = ccd.tavg.jd
    return ccd

def fdict_list_collector(fdict_list_creator,
                         directory=None,
                         collection=None,
                         subdirs=None,
                         glob_include=None,
                         imagetyp=None,
                         **kwargs):

    if subdirs is None:
        subdirs = []
    glob_include = assure_list(glob_include)
    if not isinstance(glob_include, list):
        glob_include = [glob_include]
    fdict_list = []
    if collection is None:
        # Prepare to call ourselves recursively to build up a list of
        # fnames in the provided directory and optional subdirectories
        if not os.path.isdir(directory):
            # This is the end of our recursive line
            return fdict_list
        for sd in subdirs:
            subdir = os.path.join(directory, sd)
            sub_fdict_list = fdict_list_collector \
                (fdict_list_creator,
                 subdir,
                 imagetyp=imagetyp,
                 glob_include=glob_include,
                 **kwargs)
            for sl in sub_fdict_list:
                fdict_list.append(sl)
        # After processing our subdirs, process 'directory.'
        # Make loop runs at least once
        if len(glob_include) == 0:
            glob_include = [None]
        for gi in glob_include:
            # Speed things up considerably by allowing globbing.  As
            # per comment above, if None passed to glob_include, this
            # runs once with None passed to ccdp.ImageFileCollection's
            # glob_include
            # Avoid anoying warning about empty collection
            flist = glob.glob(os.path.join(directory, gi))
            # Tricky!  Catch the case where AutoFlat is a subdir AND
            # matches glob_include
            flist = [f for f in flist if os.path.basename(f) not in subdirs]
            if len(flist) == 0:
                continue
            collection = ccdp.ImageFileCollection(directory,
                                                  filenames=flist)
            # Call ourselves recursively, but using the code below,
            # since collection is now defined
            gi_fdict_list = fdict_list_collector \
                (fdict_list_creator,
                 collection=collection,
                 imagetyp=imagetyp,
                 **kwargs)
            for gi in gi_fdict_list:
                fdict_list.append(gi)
        # Here is the end of our recursive line if directory and
        # optional subdirs were specified
        return fdict_list
    if collection.summary is None:
        # We were probably called on a glob_include that yielded no results
        return fdict_list

    # If we made it here, we have a collection, possibly from calling
    # ourselves recursively.  Hand off to our fdict_list_creator to do
    # all the work
    return fdict_list_creator(collection, imagetyp=imagetyp, **kwargs)

def bias_dark_fdict_creator(collection,
                            imagetyp=None,
                            dccdt_tolerance=DCCDT_TOLERANCE,
                            debug=False):
    # Create a new collection narrowed to our imagetyp.  
    directory = collection.location
    # We require imagetyp designation and are not polite in its absence 
    collection = collection.filter(imagetyp=imagetyp)
    # --> Oops, this recycles binned biaes I took for a while to just
    # waste some time.  For now, let them be rejected later on
    #
    # Reject binned and non-full frame images, such as I took early
    # on.  Note, this currently doesn't leave the directory with a
    # "bad" marker.  To do that, just uncomment this code and the non
    # full-frame shapes will be caught later.  If we really wanted to
    # process other modes properly, we would add ccd.shape and binning
    # info to the filenames.  
    #try:
    #    collection = collection.filter(naxis1=sx694.naxis1,
    #                                   naxis2=sx694.naxis2)
    #except Exception as e:
    #    log.error(f'Problem collecting full-frame files of imagetyp {imagetyp}  in {directory}: {e}')
    #    return []
    # Guide camera biases would add another layer of complexity with
    # no CCD-TEMP
    if 'ccd-temp' not in collection.keywords:
        log.error(f'CCD-TEMP not found in any {imagetyp} files in {directory}')
        return []
    # Create a summary table narrowed to our imagetyp
    narrow_to_imagetyp = collection.summary
    ts = narrow_to_imagetyp['ccd-temp']
    # ccd-temp is recorded as a string.  Convert it to a number so
    # we can sort +/- values properly
    ts = np.asarray(ts)
    # If some Lodestar guide camera biases snuck in, filter them
    # here
    tidx = np.flatnonzero(ts != None)
    narrow_to_imagetyp = narrow_to_imagetyp[tidx]
    ts = ts[tidx]
    # Get the sort indices so we can extract fnames in proper order
    tsort_idx = np.argsort(ts)
    # For ease of use, re-order everything in terms of tsort
    ts = ts[tsort_idx]
    narrow_to_imagetyp = narrow_to_imagetyp[tsort_idx]    
    # Spot jumps in t and translate them into slices into ts
    dts = ts[1:] - ts[0:-1]
    jump = np.flatnonzero(dts > dccdt_tolerance)
    # Note, when jump if an empty array, this just returns [0]
    tslices = np.append(0, jump+1)
    # Whew!  This was a tricky one!
    # https://stackoverflow.com/questions/509211/understanding-slice-notation
    # showed that I needed None and explicit call to slice(), below,
    # to be able to generate an array element in tslices that referred
    # to the last array element in ts.  :-1 is the next to the last
    # element because of how slices work.  Appending just None to an
    # array avoids depreciation complaint from numpy if you try to do
    # np.append(0, [jump+1, None])
    tslices = np.append(tslices, None)
    if debug:
        print(ts)
        print(dts)
        print(tslices)
    fdict_list = []
    for it in range(len(tslices)-1):
        these_ts = ts[slice(tslices[it], tslices[it+1])]
        mean_ccdt = np.mean(these_ts)
        # Create a new summary Table that inlcudes just these Ts
        narrow_to_t = narrow_to_imagetyp[tslices[it]:tslices[it+1]]
        exps = narrow_to_t['exptime']
        # These are sorted by increasing exposure time
        ues = np.unique(exps)
        for ue in ues:
            exp_idx = np.flatnonzero(exps == ue)
            files = narrow_to_t['file'][exp_idx]
            full_files = [os.path.join(directory, f) for f in files]
            fdict_list.append({'directory': directory,
                               'CCDT': mean_ccdt,
                               'EXPTIME': ue,
                               'fnames': full_files})
    return fdict_list

def discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate):
    if not keep_intermediate:
        for f in out_fnames:
            try:
                os.remove(f)
            except Exception as e:
                # We do not expect this, since we created these with
                # our local process
                log.error(f'Unexpected!  Remove {f} failed: ' + str(e))
        # These we expect to fail until all of our other parallel
        # processes have finished
        try:
            os.rmdir(sdir)
        except Exception as e:
            pass
        try:
            os.rmdir(calibration_scratch)
        except Exception as e:
            pass

def bias_stats(ccd, bmp_meta=None, gain=sx694.gain, **kwargs):
    """CorMultiPipe post-processing routine for bias_combine
    Returns dictionary of bias statistics for pandas dataframe
    """
    im = ccd.data
    hdr = ccd.meta
    # Calculate readnoise.  This is time-consuming
    diffs2 = (im[1:] - im[0:-1])**2
    rdnoise = np.sqrt(biweight_location(diffs2))
    # Skip uncertainty creation, since it is not used in any
    # subsequent calcs
    #uncertainty = np.multiply(rdnoise, np.ones(im.shape))
    #ccd.uncertainty = StdDevUncertainty(uncertainty)
    # Prepare to create a pandas data frame to track relevant
    # quantities
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    ccdt = ccd.meta['CCD-TEMP']
    tt = tm.tt.datetime
    # We have already subtracted overscan, so add it back in where
    # appropriate
    median = hdr['OVERSCAN_MEDIAN']
    stats = {'time': tt,
             'ccdt': ccdt,
             'median': median,
             'mean': np.mean(im) + median,
             'std': np.std(im)*gain,
             'rdnoise': rdnoise*gain,
             'min': np.min(im) + median,  
             'max': np.max(im) + median}
    if bmp_meta is not None:
        bmp_meta['bias_stats'] = stats
    return ccd

def bias_combine_one_fdict(fdict,
                           outdir=CALIBRATION_ROOT,
                           calibration_scratch=CALIBRATION_SCRATCH,
                           keep_intermediate=False,
                           show=False,
                           min_num_biases=MIN_NUM_BIASES,
                           gain=sx694.gain,
                           satlevel=sx694.satlevel,
                           readnoise=sx694.example_readnoise,
                           readnoise_tolerance=sx694.readnoise_tolerance,
                           gain_correct=False,
                           num_processes=MAX_NUM_PROCESSES,
                           mem_frac=MAX_MEM_FRAC,
                           naxis1=sx694.naxis1,
                           naxis2=sx694.naxis2,
                           bitpix=MAX_CCDDATA_BITPIX,
                           **kwargs):

    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.


    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of adu.  Default: False
    """

    fnames = fdict['fnames']
    num_files = len(fnames)
    mean_ccdt = fdict['CCDT']
    directory = fdict['directory']
    # Avoid annoying WCS warning messages
    hdr = getheader(fnames[0])
    tm = hdr['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    outbase = os.path.join(outdir, this_dateb1)
    bad_fname = outbase + '_ccdT_XXX' + '_bias_combined_bad.fits'
    if num_files < min_num_biases:
        log.warning(f"Not enough good biases found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    sdir = os.path.join(calibration_scratch, this_dateb1)

    #mem = psutil.virtual_memory()
    #num_files_can_fit = \
    #    int(min(num_files,
    #            mem.available*mem_frac/ccddata_size))
    #num_can_process = min(num_processes, num_files_can_fit)
    #print('bias_combine_one_fdict: num_processes = {}, mem_frac = {}, num_files= {}, num_files_can_fit = {}, num_can_process = {}'.format(num_processes, mem_frac, num_files, num_files_can_fit, num_can_process))

    # Use CorMultiPipe to subtract the median from each bias and
    # create a dict of stats for a pandas dataframe
    cmp = CorMultiPipeBase(num_processes=num_processes,
                           mem_frac=mem_frac,
                           naxis1=naxis1,
                           naxis2=naxis2,
                           bitpix=bitpix,
                           outdir=sdir,
                           create_outdir=True,
                           overwrite=True,
                           fits_fixed_ignore=True, 
                           pre_process_list=[light_image],
                           post_process_list=[bias_stats, jd_meta])
    #combined_base = outbase + '_bias_combined'
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"Not enough good biases {len(pout)} found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False
    out_fnames, pipe_meta = zip(*pout)
    if len(out_fnames) < min_num_biases:
        log.warning(f"Not enough good biases {len(pout)} found at CCDT = {mean_ccdt} C in {directory}")
        discard_intermediate(out_fnames, sdir,
                             calibration_scratch, keep_intermediate)
        Path(bad_fname).touch()
        return False

    stats = [m['bias_stats'] for m in pipe_meta]
    jds = [m['jd'] for m in pipe_meta]

    tbl = QTable(rows=stats)
    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first bias is on {this_dateb1} but average is {this_dateb}")

    this_ccdt = '{:.1f}'.format(mean_ccdt)
    f = plt.figure(figsize=[8.5, 11])

    # In the absence of a formal overscan region, this is the best
    # I can do
    medians = tbl['median']
    overscan = np.mean(medians)

    ax = plt.subplot(6, 1, 1)
    plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(tbl['time'], tbl['ccdt'], 'k.')
    plt.ylabel('CCDT (C)')

    ax = plt.subplot(6, 1, 2)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(tbl['time'], tbl['max'], 'k.')
    plt.ylabel('max (adu)')

    ax = plt.subplot(6, 1, 3)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=False)
    plt.plot(tbl['time'], tbl['median'], 'k.')
    plt.plot(tbl['time'], tbl['mean'], 'r.')
    plt.ylabel('median & mean (adu)')
    plt.legend(['median', 'mean'])
    secax = ax.secondary_yaxis \
        ('right',
         functions=(lambda adu: (adu - overscan)*gain,
                    lambda e: e/gain + overscan))
    secax.set_ylabel('Electrons')

    ax=plt.subplot(6, 1, 4)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(tbl['time'], tbl['min'], 'k.')
    plt.ylabel('min (adu)')

    ax=plt.subplot(6, 1, 5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(tbl['time'], tbl['std'], 'k.')
    plt.ylabel('std (electron)')

    ax=plt.subplot(6, 1, 6)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    plt.plot(tbl['time'], tbl['rdnoise'], 'k.')
    plt.ylabel('rdnoise (electron)')

    plt.gcf().autofmt_xdate()

    # At the 0.5 deg level, there seems to be no correlation between T and bias level
    #plt.plot(tbl['ccdt'], tbl['mean'], 'k.')
    #plt.xlabel('ccdt')
    #plt.ylabel('mean')
    #plt.show()
        
    # Make sure outdir exists
    os.makedirs(outdir, exist_ok=True)
    outbase = os.path.join(outdir, this_dateb + '_ccdT_' + this_ccdt)
    out_fname = outbase + '_bias_combined.fits'
    savefig_overwrite((outbase + '_bias_vs_time.png'), transparent=True)
    if show:
        plt.show()
    plt.close()

    # Do a sanity check of readnoise
    av_rdnoise = np.mean(tbl['rdnoise'])            
    if (np.abs(av_rdnoise/sx694.example_readnoise - 1)
        > readnoise_tolerance):
        log.warning('High readnoise {}, skipping {}'.format(av_rdnoise, out_fname))
        Path(outbase + '_bad.fits').touch()
        return False

    # Use ccdp.combine since it enables memory management by breaking
    # up images to smaller chunks (better than throwing images away).
    # --> eventually it would be great to parallelize this primitive,
    # since it is very slow.  In the mean time I have parallelized all
    # the higher steps!
    mem = psutil.virtual_memory()
    im = \
        ccdp.combine(list(out_fnames),
                     method='average',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=mem.available*mem_frac)
    im.meta = sx694.metadata(im.meta)
    if gain_correct:
        im = ccdp.gain_correct(im, gain*u.electron/u.adu)
        im_gain = 1
    else:
        im_gain = gain
    im = mask_above_key(im, key='SATLEVEL')
    im = mask_above_key(im, key='NONLIN')
        
    # Collect image metadata.  For some reason, masked pixels
    # aren't ignored by std, etc. even though they output masked
    # arrays (which is annoying in its own right -- see example
    # commented mean).  So just create a new array, if needed, and
    # only put into it the good pixels
    if im.mask is None:
        # This is not a copy!  But don't worry, we don't change tim,
        # just collect info from it
        tim = im
    else:
        # This is a new array with fewer elements.  We will collect
        # stats and write the original im, below
        tim = im.data[im.mask == 0]
    std =  np.std(tim)*im_gain
    med =  np.median(tim)*im_gain
    #mean = np.asscalar(np.mean(tim).data  )
    mean = np.mean(tim)*im_gain
    tmin = np.min(tim)*im_gain
    tmax = np.max(tim)*im_gain
    log.info(f'combine bias statistics for {outbase}')
    log.info(f'std = {std:.2f}; mean = {mean:.2f}, med = {med:.2f}; min = {tmin:.2f}; max = {tmax:.2f}')
    im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
    im.meta['CCD-TEMP'] = (mean_ccdt, 'Average CCD temperature for combined biases')
    im.meta['RDNOISE'] = (av_rdnoise, 'Measured readnoise (electron)')
    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
    im.meta['MIN'] = (tmin, 'Min of image (electron)')
    im.meta['MAX'] = (tmax, 'Max of image (electron)')
    im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (adu)')
    im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of biases combined')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')
    add_history(im.meta,
                'SATLEVEL and NONLIN apply to pre-overscan subtraction')
    # Leave these large for fast calculations downstream and make
    # final results that primarily sit on disk in bulk small
    #im.data = im.data.astype('float32')
    #im.uncertainty.array = im.uncertainty.array.astype('float32')
    im.write(out_fname, overwrite=True)
    # Always display image in electrons
    impl = plt.imshow(im.multiply(im_gain), origin='lower',
                      cmap=plt.cm.gray,
                      filternorm=0, interpolation='none',
                      vmin=med-std, vmax=med+std)
    plt.title('CCDT = {} C on {} (electrons)'.format(this_ccdt, this_dateb))
    savefig_overwrite((outbase + '_bias_combined.png'), transparent=True)
    if show:
        plt.show()
    plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)
                
def bias_combine(directory=None,
                 collection=None,
                 subdirs=CALIBRATION_SUBDIRS,
                 glob_include=BIAS_GLOB,
                 dccdt_tolerance=DCCDT_TOLERANCE,
                 num_processes=MAX_NUM_PROCESSES,
                 mem_frac=MAX_MEM_FRAC,
                 num_calibration_files=NUM_CALIBRATION_FILES,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=MAX_CCDDATA_BITPIX,
                 process_expand_factor=COR_PROCESS_EXPAND_FACTOR,
                 **kwargs):
    """Combine biases in a directory

    Parameters
    ----------
    directory : string
        Directory in which to find biases.  Default: ``None``

    collection : ccdp.Collection
        Collection of directory in which to find calibration data.
        Default: ``None``

    subdirs : list
        List of subdirectories in which to search for calibration
        data.  Default: :value:`CALIBRATION_SUBDIRS`

    glob_include : list
        List of `glob` expressions for calibration filenames

    dccdt_tolerance : float
        During the creation of master biases and darks files, are
        grouped by CCD temperature (ccdt).  This is the change in
        temperature seen as a function of time that is used to trigger
        the creation of a new group

    num_processes : int
        Number of processes available to this task for
        multiprocessing.  Default: :value:`MAX_NUM_PROCESSES`

    mem_frac : float
        Fraction of memory available to this task.  Default:
        :value:`MAX_MEM_FRAC`

    **kwargs passed to bias_combine_one_fdict

    """
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    log.info(f'bias_combine directory: {directory}')
    fdict_list = \
        fdict_list_collector(bias_dark_fdict_creator,
                             directory=directory,
                             collection=collection,
                             subdirs=subdirs,
                             imagetyp='BIAS',
                             glob_include=glob_include,
                             dccdt_tolerance=DCCDT_TOLERANCE)
    nfdicts = len(fdict_list)
    if nfdicts == 0:
        log.debug('No usable biases found in: ' + directory)
        return False

    one_fdict_size = (num_calibration_files
                      * naxis1 * naxis2
                      * bitpix/8
                      * process_expand_factor)

    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)

    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug(f'bias_combine: {directory} nfdicts = {nfdicts}, num_processes = {num_processes}, mem_frac = {mem_frac}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')

    wwk = WorkerWithKwargs(bias_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                wwk.worker(fdict)
    else:
        with NestablePool(processes=our_num_processes) as p:
            p.map(wwk.worker, fdict_list)

def dark_combine_one_fdict(fdict,
                           outdir=CALIBRATION_ROOT,
                           calibration_scratch=CALIBRATION_SCRATCH,
                           keep_intermediate=False,
                           show=False,
                           mask_threshold=sx694.dark_mask_threshold,
                           num_processes=MAX_NUM_PROCESSES,
                           mem_frac=MAX_MEM_FRAC,
                           naxis1=sx694.naxis1,
                           naxis2=sx694.naxis2,
                           bitpix=MAX_CCDDATA_BITPIX,
                           **kwargs):
    """Worker that allows the parallelization of calibrations taken at one
    temperature, exposure time, filter, etc.

    """

    fnames = fdict['fnames']
    mean_ccdt = fdict['CCDT']
    exptime = fdict['EXPTIME']
    directory = fdict['directory']
    # Avoid annoying WCS warning messages
    hdr = getheader(fnames[0])
    tm = hdr['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    badbase = '{}_ccdT_{:.1f}_exptime_{}s'.format(
        this_dateb1, mean_ccdt, exptime)
    badbase = os.path.join(outdir, badbase)
    bad_fname = badbase + '_dark_combined_bad.fits'

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    sdir = os.path.join(calibration_scratch, this_dateb1)

    cmp = CorMultiPipeBase(num_processes=num_processes,
                           mem_frac=mem_frac,
                           naxis1=naxis1,
                           naxis2=naxis2,
                           bitpix=bitpix,
                           outdir=sdir,
                           create_outdir=True,
                           overwrite=True,
                           fits_fixed_ignore=True, 
                           pre_process_list=[light_image],
                           post_process_list=[jd_meta])
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"No good darks found at CCDT = {mean_ccdt} C in {directory}")
        Path(bad_fname).touch()
        return False

    out_fnames, pipe_meta = zip(*pout)
    jds = [m['jd'] for m in pipe_meta]

    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first dark is on {this_dateb1} but average is {this_dateb}")
    this_ccdt = '{:.1f}'.format(mean_ccdt)
    outbase = '{}_ccdT_{}_exptime_{}s'.format(
        this_dateb, this_ccdt, exptime)
    
    mem = psutil.virtual_memory()
    out_fnames = list(out_fnames)
    if len(out_fnames) == 1:
        log.debug(f'single out_fname = {out_fnames}')
        im = CorDataBase.read(out_fnames[0])
    else:
        im = \
            ccdp.combine(out_fnames,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*mem_frac)
    im = mask_above_key(im, key='SATLEVEL')
    im = mask_above_key(im, key='NONLIN')

    # Create a mask that blanks out all our pixels that are just
    # readnoise.  Multiply this in as zeros, not a formal mask,
    # otherwise subsequent operations with the dark will mask out
    # all but the dark current-affected pixels!
    measured_readnoise = im.meta['RDNOISE']
    is_dark_mask = im.data > measured_readnoise * mask_threshold
    n_dark_pix = np.count_nonzero(is_dark_mask)
    im.meta['NDARKPIX'] \
        = (n_dark_pix, 'number of pixels with dark current')
    if n_dark_pix > 0:
        im.data = im.data * is_dark_mask
        if im.uncertainty is not None:
            im.uncertainty.array = im.uncertainty.array*is_dark_mask

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
    log.info(f'combined dark statistics for {outbase}')
    log.info(f'std = {std:.2f}; rdnoise = {rdnoise:.2f}; mean = {mean:.2f}; med = {med:.2f}; min = {tmin:.2f}; max = {tmax:.2f}; n_dark_pix = {n_dark_pix}')
    im.meta['STD'] = (std, 'Standard deviation of image (electron)')
    im.meta['MEDIAN'] = (med, 'Median of image (electron)')
    im.meta['MEAN'] = (mean, 'Mean of image (electron)')
    im.meta['MIN'] = (tmin, 'Min of image (electron)')
    im.meta['MAX'] = (tmax, 'Max of image (electron)')
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of darks combined')
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')
    im.meta['HIERARCH MASK_THRESHOLD'] \
        = (mask_threshold, '*RDNOISE (electron)')
    add_history(im.meta,
                'Setting pixes below MASK_THRESHOLD to zero; prevents subtraction noise')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    # Prepare to write
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outbase = os.path.join(outdir, outbase)
    out_fname = outbase + '_dark_combined.fits'
    # Leave these large for fast calculations downstream and make
    # final results that primarily sit on disk in bulk small
    #im.data = im.data.astype('float32')
    #im.uncertainty.array = im.uncertainty.array.astype('float32')
    im.write(out_fname, overwrite=True)
    if show:
        impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray,
                          filternorm=0, interpolation='none',
                          vmin=med-std, vmax=med+std)
        plt.show()
        plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)

def dark_combine(directory=None,
                 collection=None,
                 subdirs=CALIBRATION_SUBDIRS,
                 glob_include=DARK_GLOB,
                 dccdt_tolerance=DCCDT_TOLERANCE,
                 num_processes=MAX_NUM_PROCESSES,
                 mem_frac=MAX_MEM_FRAC,
                 num_calibration_files=NUM_CALIBRATION_FILES,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=MAX_CCDDATA_BITPIX,
                 process_expand_factor=COR_PROCESS_EXPAND_FACTOR,
                 **kwargs):
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    log.info(f'dark_combine directory: {directory}')
    fdict_list = \
        fdict_list_collector(bias_dark_fdict_creator,
                             directory=directory,
                             collection=collection,
                             subdirs=subdirs,
                             imagetyp='DARK',
                             glob_include=glob_include,
                             dccdt_tolerance=dccdt_tolerance)
    nfdicts = len(fdict_list)
    if len(fdict_list) == 0:
        log.debug('No usable darks found in: ' + directory)
        return False

    one_fdict_size = (num_calibration_files
                      * naxis1 * naxis2
                      * bitpix/8
                      * process_expand_factor)

    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_fdict_size)
    num_subprocesses = int(num_processes / our_num_processes)
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug(f'dark_combine: {directory} num_processes = {num_processes}, mem_frac = {mem_frac:.0f}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')

    wwk = WorkerWithKwargs(dark_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                wwk.worker(fdict)
    else:
        with NestablePool(processes=our_num_processes) as p:
            p.map(wwk.worker, fdict_list)

def flat_fdict_creator(collection,
                       imagetyp=None):
    # Create a new collection narrowed to our imagetyp
    directory = collection.location
    collection = collection.filter(imagetyp=imagetyp)
    # --> Oops, this recycles wrong-sized flats which are better
    # rejected later
    #try:
    #    collection = collection.filter(naxis1=sx694.naxis1,
    #                                   naxis2=sx694.naxis2)
    #except Exception as e:
    #    log.error(f'Problem collecting full-frame files of imagetyp {imagetyp}  in {directory}: {e}')
    #    return []
    if 'filter' not in collection.keywords:
        log.error(f'filter not found in any {imagetyp} files in {directory}')
        return []
    # Keep in mind filters will have our old names
    standardize_filt_name(collection)
    filters = collection.values('filter', unique=True)
    fdict_list = []
    for filt in filters:
        fcollection = collection.filter(filter=filt)
        fnames = fcollection.files_filtered(include_path=True)
        fdict_list.append({'directory': directory,
                           'filter': filt,
                           'fnames': fnames})
    return fdict_list

def flat_process(ccd, bmp_meta=None,
                 init_threshold=100, # units of readnoise
                 nd_edge_expand=ND_EDGE_EXPAND,
                 in_name=None,
                 **kwargs):
    if ccd.meta.get('flatdiv') is not None:
        raise ValueError('Trying to reprocess a processed flat')
    # Use basic patch medians to spot pathological cases
    mdp, mlp = im_med_min_max(ccd)
    if mlp < 1000:
        log.warning(f'flat median of {mlp} {ccd.unit} too low {in_name}')
        return None
    if mlp > ccd.meta['NONLIN']:
        log.warning(f'flat median of {mlp} {ccd.unit} too high {in_name}')
        return None
    # Use photutils.Background2D to smooth each flat and get a
    # good maximum value.  Mask edges and ND filter so as to
    # increase quality of background map
    mask = np.zeros(ccd.shape, bool)
    # Return a copy of ccd with the edge_mask property adjusted.  Do
    # it this way to keep ccd's ND filt parameters intact
    emccd = CorDataNDparams(ccd, edge_mask=-nd_edge_expand)
    try:
        mask[emccd.ND_coords] = True
    except Exception as e:
        # We should have caught all nasty cases above
        log.error(f'ND_coords gave error: {e} for {in_name}')
        return None
    del emccd
    rdnoise = ccd.meta['RDNOISE']
    mask[ccd.data < rdnoise * init_threshold] = True
    ccd.mask = mask
    bkg_estimator = MedianBackground()
    b = Background2D(ccd, 20, mask=mask, filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    if max_flat > ccd.meta['NONLIN']*ccd.unit:
        log.debug(f'flat max value of {max_flat.value} {max_flat.unit} too bright: {in_name}')
        return None
    ccd.mask = None
    ccd = ccd.divide(max_flat, handle_meta='first_found')
    # --> This will get better if Card units are implemented
    ccd.meta['FLATDIV'] = (max_flat.value, f'Normalization value (smoothed max) ({max_flat.unit})')
    # Get ready to capture the mean DATE-OBS
    tm = Time(ccd.meta['DATE-OBS'], format='fits')
    if bmp_meta is not None:
        bmp_meta['jd'] = tm.jd 
    return ccd

def flat_combine_one_fdict(fdict,
                           outdir=CALIBRATION_ROOT,
                           calibration_scratch=CALIBRATION_SCRATCH,
                           keep_intermediate=False,
                           min_num_flats=MIN_NUM_FLATS,
                           num_processes=MAX_NUM_PROCESSES,
                           mem_frac=MAX_MEM_FRAC,
                           naxis1=sx694.naxis1,
                           naxis2=sx694.naxis2,
                           bitpix=MAX_CCDDATA_BITPIX,
                           show=False,
                           nd_edge_expand=ND_EDGE_EXPAND,
                           flat_cut=FLAT_CUT,
                           **kwargs):
    fnames = fdict['fnames']
    num_files = len(fnames)
    this_filter = fdict['filter']
    directory = fdict['directory']
    # Avoid annoying WCS warning messages
    hdr = getheader(fnames[0])
    tm = hdr['DATE-OBS']
    this_dateb1, _ = tm.split('T')
    outbase = os.path.join(outdir, this_dateb1)
    bad_fname = outbase + '_' + this_filter + '_flat_bad.fits'

    if len(fnames) < min_num_flats:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        Path(bad_fname).touch()
        return False

    # Make a scratch directory that is the date of the first file.
    # Not as fancy as the biases, but, hey, it is a scratch directory
    sdir = os.path.join(calibration_scratch, this_dateb1)

    cmp = CorMultiPipeBase(ccddata_cls=CorDataNDparams,
                           num_processes=num_processes,
                           mem_frac=mem_frac,
                           naxis1=naxis1,
                           naxis2=naxis2,
                           bitpix=bitpix,
                           outdir=sdir,
                           create_outdir=True,
                           overwrite=True,
                           fits_fixed_ignore=True, 
                           post_process_list=[flat_process, jd_meta],
                           **kwargs)
    pout = cmp.pipeline(fnames, **kwargs)
    pout, fnames = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        Path(bad_fname).touch()
        return False
    out_fnames, pipe_meta = zip(*pout)
    if len(out_fnames) < min_num_flats:
        log.warning(f"Not enough good flats found for filter {this_filter} in {directory}")
        discard_intermediate(out_fnames, sdir,
                             calibration_scratch, keep_intermediate)
        Path(bad_fname).touch()
        return False

    jds = [m['jd'] for m in pipe_meta]

    # Combine our flats
    mem = psutil.virtual_memory()
    #print(f'flat_combine_one_filt: mem_frac {mem_frac}; num_processes {num_processes}')
    #print(f'flat_combine_one_filt: mem_limit {mem.available*mem_frac/2**20}')

    im = \
        ccdp.combine(list(out_fnames),
                     method='average',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=mem.available*mem_frac)
    im.meta['NCOMBINE'] = (len(out_fnames), 'Number of flats combined')
    # Record each filename
    for i, f in enumerate(fnames):
        im.meta['FILE{0:02}'.format(i)] = f
    add_history(im.meta,
                'Combining NCOMBINE biases indicated in FILENN')

    # Interpolate over our ND filter
    #print(f'flat_combine_one_filt pre CorObsData: mem available: {mem.available/2**20}')
    emccd = CorDataNDparams(im, edge_mask=-nd_edge_expand)
    good_mask = np.ones(im.shape, bool)
    good_mask[emccd.ND_coords] = False
    points = np.nonzero(good_mask)
    values = im[points]
    xi = emccd.ND_coords

    log.debug(f'flat_combine_one_filt post CorObsData: mem available: {mem.available/2**20:.0}')

    # Linear behaved much better
    nd_replacement = interpolate.griddata(points,
                                          values,
                                          xi,
                                          method='linear')
                                          #method='cubic')
    log.debug(f'flat_combine_one_filt post interpolate.griddata mem available: {mem.available/2**20}')
    im.data[xi] = nd_replacement
    # Do one last smoothing and renormalization
    bkg_estimator = MedianBackground()
    b = Background2D(im, 20, mask=(im.data<flat_cut), filter_size=5,
                     bkg_estimator=bkg_estimator)
    max_flat = np.max(b.background)
    log.debug(f'flat_combine_one_filt post Background2D mem available: {mem.available/2**20}')
    im = im.divide(max_flat, handle_meta='first_found')
    im.mask = im.data < flat_cut
    im.meta['FLAT_CUT'] = (flat_cut, 'Value below which flat is masked')

    # Prepare to write
    tm = Time(np.mean(jds), format='jd')
    this_date = tm.fits
    this_dateb = this_date.split('T')[0]
    if this_dateb != this_dateb1:
        log.warning(f"first flat is on {this_dateb1} but average is {this_dateb}")

    outbase = '{}_{}'.format(this_dateb, this_filter)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outbase = os.path.join(outdir, outbase)
    out_fname = outbase + '_flat.fits'
    im.write(out_fname, overwrite=True)
    if show:
        impl = plt.imshow(im, origin='upper', cmap=plt.cm.gray)
        plt.show()
        plt.close()
    discard_intermediate(out_fnames, sdir,
                         calibration_scratch, keep_intermediate)
    
def flat_combine(directory=None,
                 collection=None,
                 subdirs=CALIBRATION_SUBDIRS,
                 glob_include=FLAT_GLOB,
                 num_processes=MAX_NUM_PROCESSES,
                 mem_frac=MAX_MEM_FRAC,
                 num_calibration_files=NUM_CALIBRATION_FILES,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=64, # uncertainty and mask not used in griddata
                 griddata_expand_factor=GRIDDATA_EXPAND_FACTOR,
                 **kwargs):
    if collection is not None:
        # Make sure 'directory' is a valid variable
        directory = collection.location
    log.info(f'flat_combine directory: {directory}')
    fdict_list = \
        fdict_list_collector(flat_fdict_creator,
                             directory=directory,
                             collection=collection,
                             subdirs=subdirs,
                             imagetyp='FLAT',
                             glob_include=glob_include)
    nfdicts = len(fdict_list)
    if nfdicts == 0:
        log.debug('No usable flats found in: ' + directory)
        return False
    
    one_filt_size = (num_calibration_files
                     * naxis1 * naxis2
                     * bitpix/8
                     * griddata_expand_factor)

    our_num_processes = num_can_process(nfdicts,
                                        num_processes=num_processes,
                                        mem_frac=mem_frac,
                                        process_size=one_filt_size,
                                        error_if_zero=False)
    our_num_processes = max(1, our_num_processes)

    # Combining files is the slow part, so we want the maximum of
    # processes doing that in parallel
    log.debug(f'flat_combine: {directory}, nfdicts = {nfdicts}, our_num_processes = {our_num_processes}')
    # Number of sub-processes in each process we will spawn
    num_subprocesses = int(num_processes / our_num_processes)
    # Similarly, the memory fraction for each process we will spawn
    subprocess_mem_frac = mem_frac / our_num_processes
    log.debug(f'flat_combine: {directory} num_processes = {num_processes}, mem_frac = {mem_frac:.0f}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')

    wwk = WorkerWithKwargs(flat_combine_one_fdict,
                           num_processes=num_subprocesses,
                           mem_frac=subprocess_mem_frac,
                           **kwargs)
    if nfdicts == 1:
        for fdict in fdict_list:
                wwk.worker(fdict)
    else:
        with NestablePool(processes=our_num_processes) as p:
            p.map(wwk.worker, fdict_list)

def flist_to_dict(flist):
    """Flat-specific dictionary creator"""
    dlist = []
    for f in flist:
        bname = os.path.basename(f)
        s = bname.split('_')
        d = {'fname': f, 'date': s[0], 'band': s[1], 'onoff': s[2]}
        dlist.append(d)
    return dlist
    
def flat_flux(fname_or_ccd):
    # getheader is much faster than CCDData.read or [Red]CordData.read
    # because these classmethods read the whole file
    if isinstance(fname_or_ccd, str):
        hdr = getheader(fname_or_ccd)
    else:
        hdr = ccd.meta
    maxval = hdr['FLATDIV']
    exptime = hdr['EXPTIME']
    flux = maxval/exptime
    return flux

######### Calibration object

def dir_has_calibration(directory, glob_include, subdirs=None):
    """Returns True if directory has calibration files matching pattern(s)
in glob_include.  Optionally checks subdirs"""
    if not os.path.isdir(directory):
        # This is the end of our recursive line
        return False
    if subdirs is None:
        subdirs = []
    for sd in subdirs:
        subdir = os.path.join(directory, sd)
        if dir_has_calibration(subdir, glob_include):
            return True
    # If we made it here, our subdirs had no calibration files or we
    # have been called recursively and are in one
    for gi in glob_include:
        flist = glob.glob(os.path.join(directory, gi))
        if len(flist) > 0:
            return True
    return False

class Calibration():
    """Class for conducting CCD calibrations"""
    def __init__(self,
                 reduce=False,
                 raw_data_root=RAW_DATA_ROOT,
                 calibration_root=CALIBRATION_ROOT,
                 subdirs=CALIBRATION_SUBDIRS,
                 fits_fixed_ignore=True,
                 keep_intermediate=False,
                 ccdt_tolerance=CCDT_TOLERANCE,
                 dark_exp_margin=DARK_EXP_MARGIN,
                 start_date=None,
                 stop_date=None,
                 gain_correct=True, # This is gain correcting the bias and dark
                 num_processes=MAX_NUM_PROCESSES,
                 mem_frac=MAX_MEM_FRAC,
                 num_ccdts=NUM_CCDTS,
                 num_dark_exptimes=NUM_DARK_EXPTIMES,
                 num_filts=NUM_FILTS,
                 num_calibration_files=NUM_CALIBRATION_FILES,
                 naxis1=sx694.naxis1,
                 naxis2=sx694.naxis2,
                 bitpix=MAX_CCDDATA_BITPIX,
                 process_expand_factor=COR_PROCESS_EXPAND_FACTOR,
                 griddata_expand_factor=GRIDDATA_EXPAND_FACTOR,
                 bias_glob=BIAS_GLOB, 
                 dark_glob=DARK_GLOB,
                 flat_glob=FLAT_GLOB,
                 flat_cut=FLAT_CUT,
                 nd_edge_expand=ND_EDGE_EXPAND,
                 stable_flat_date=STABLE_FLAT_DATE,
                 plot_flat_ratios=False,
                 lockfile=LOCKFILE):
        self._raw_data_root = raw_data_root
        self._calibration_root = calibration_root
        self._subdirs = subdirs
        self.keep_intermediate = keep_intermediate
        self._ccdt_tolerance = ccdt_tolerance
        self._dark_exp_margin=dark_exp_margin
        self._bias_table = None
        self._dark_table = None
        self._flat_table = None
        self._flat_ratio_list = None
        # gain_correct is set only in the biases and propagated
        # through the rest of the pipeline in cor_process
        self._gain_correct = gain_correct
        self._bias_glob = assure_list(bias_glob)
        self._dark_glob = assure_list(dark_glob)
        self._flat_glob = assure_list(flat_glob)
        self._lockfile = lockfile
        self.flat_cut = flat_cut
        self.nd_edge_expand = nd_edge_expand
        self.stable_flat_date = stable_flat_date
        self.plot_flat_ratios = plot_flat_ratios
        self.num_processes = num_processes
        self.mem_frac = mem_frac
        self.num_ccdts = num_ccdts
        self.num_dark_exptimes = num_dark_exptimes
        self.num_filts = num_filts
        self.num_calibration_files = num_calibration_files
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.bitpix = bitpix
        self.process_expand_factor = process_expand_factor
        self.griddata_expand_factor = griddata_expand_factor
        if start_date is None:
            self._start_date = datetime.datetime(1,1,1)
        else:
            self._start_date = datetime.datetime.strptime(calibration_start,
                                                          "%Y-%m-%d")
        if stop_date is None:
            # Make stop time tomorrow in case we are analyzing on the
            # UT boundary
            self._stop_date = (datetime.datetime.today()
                               + datetime.timedelta(days=1))
        else:
            self._stop_date = datetime.datetime.strptime(calibration_stop,
                                                         "%Y-%m-%d")
        assert self._start_date <= self._stop_date
        # These need to be on a per-instantiation basis, since they
        # depend on our particular start-stop range.  These are also
        # important, since we don't take calibrations every night.  The
        # cost of checking for new reductions is relatively low, since
        # it is mostly a directory listing exercise
        self._bias_dirs_dates_checked = None
        self._dark_dirs_dates_checked = None
        self._flat_dirs_dates_checked = None
        if reduce:
            self.reduce()

    @property
    def gain_correct(self):
        return self._gain_correct

    def dirs_dates_to_reduce(self, table_creator,
                             glob_include,
                             dirs_dates_checked=None,
                             subdirs=None):
        to_check = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        # See if we have reduced/checked any/everything in this
        # instantiation.  This is not as efficient as it could be
        # since we have sorted lists, but we don't have many elements,
        # so there is not much point in getting fancier
        if dirs_dates_checked is not None:
            to_check = [dt for dt in to_check
                        if not dt in dirs_dates_checked]
            if len(to_check) == 0:
                return []
        # Take any reductions on disk out of the list.  Note, we check
        # for date only, since we have lost the original directory
        # information once reduced
        tbl = table_creator(autoreduce=False, rescan=True)
        if tbl is not None:
            reduced_ts = [tm.to_datetime() for tm in tbl['dates']]
            # Remove duplicates
            reduced_ts = list(set(reduced_ts))
            to_check = [dt for dt in to_check
                        if not dt[1] in reduced_ts]
            if len(to_check) == 0:
                return []
        to_reduce = [dt for dt in to_check
                     if dir_has_calibration(dt[0],
                                            glob_include,
                                            subdirs=subdirs)]
        # Remove duplicates
        return sorted(list(set(to_reduce)))

    def reduce_bias(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.bias_table_create,
                                      self._bias_glob,
                                      self._bias_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_fdict_size = (self.num_calibration_files
                          * self.naxis1 * self.naxis2
                          * self.bitpix/8
                          * self.process_expand_factor)

        ncp = num_can_process(self.num_ccdts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_ccdts * one_fdict_size,
                              error_if_zero=False)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_bias: ndirs_dates = {ndirs_dates}')
        log.debug(f'Calibration.reduce_bias: self.num_processes = {self.num_processes}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')
        #return
        wwk = WorkerWithKwargs(bias_combine,
                               subdirs=self._subdirs,
                               glob_include=self._bias_glob,
                               outdir=self._calibration_root,
                               auto=True, # A little dangerous, but just one place for changes
                               gain_correct=self._gain_correct,
                               num_processes=self.num_processes,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               process_expand_factor=self.process_expand_factor,

                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NestablePool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.bias_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._bias_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce_dark(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.dark_table_create,
                                      self._dark_glob,
                                      self._dark_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_fdict_size = (self.num_calibration_files
                          * self.naxis1 * self.naxis2
                          * self.bitpix/8
                          * self.process_expand_factor)

        ncp = num_can_process(self.num_ccdts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_ccdts * one_fdict_size,
                              error_if_zero=False)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_dark: ndirs_dates = {ndirs_dates}')
        log.debug(f'Calibration.reduce_dark: self.num_processes = {self.num_processes}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')
        #return
        wwk = WorkerWithKwargs(dark_combine,
                               subdirs=self._subdirs,
                               glob_include=self._dark_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               process_expand_factor=self.process_expand_factor,

                               num_calibration_files=self.num_calibration_files,
                               mem_frac=self.mem_frac,
                               keep_intermediate=self.keep_intermediate)

        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NestablePool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.dark_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._dark_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce_flat(self):
        dirs_dates = \
            self.dirs_dates_to_reduce(self.flat_table_create,
                                      self._flat_glob,
                                      self._flat_dirs_dates_checked,
                                      self._subdirs)
        ndirs_dates = len(dirs_dates)
        if ndirs_dates == 0:
            return

        # If we made it here, we have some real work to do
        # Set a simple lockfile so we don't have multiple processes reducing
        lock = Lockfile(self._lockfile)
        lock.create()

        one_filt_size = (self.num_calibration_files
                         * self.naxis1 * self.naxis2
                         * self.bitpix/8
                         * self.griddata_expand_factor)

        # Our sub-process can divide and conquer if necessary
        ncp = num_can_process(self.num_filts,
                              num_processes=self.num_processes,
                              mem_frac=self.mem_frac,
                              process_size=self.num_filts * one_filt_size,
                              error_if_zero=False)
        our_num_processes = max(1, ncp)
        num_subprocesses = int(self.num_processes / our_num_processes)
        subprocess_mem_frac = self.mem_frac / our_num_processes
        log.debug(f'Calibration.reduce_flat: ndirs_dates = {ndirs_dates}')
        log.debug(f'Calibration.reduce_flat: self.num_processes = {self.num_processes}, our_num_processes = {our_num_processes}, num_subprocesses = {num_subprocesses}, subprocess_mem_frac = {subprocess_mem_frac:.2f}')
        wwk = WorkerWithKwargs(flat_combine,
                               subdirs=self._subdirs,
                               glob_include=self._flat_glob,
                               outdir=self._calibration_root,
                               calibration=self,
                               auto=True, # A little dangerous, but just one place for changes
                               num_processes=self.num_processes,
                               mem_frac=self.mem_frac,
                               num_calibration_files=self.num_calibration_files,
                               naxis1=self.naxis1,
                               naxis2=self.naxis2,
                               griddata_expand_factor=self.griddata_expand_factor,
                               keep_intermediate=self.keep_intermediate,
                               flat_cut=self.flat_cut,
                               nd_edge_expand=self.nd_edge_expand)
                               
        dirs = [dt[0] for dt in dirs_dates]
        if our_num_processes == 1:
            for d in dirs:
                wwk.worker(d)
        else:
            with NestablePool(processes=our_num_processes) as p:
                p.map(wwk.worker, dirs)

        self.flat_table_create(rescan=True, autoreduce=False)
        # This could potentially get set in dirs_dates_to_reduce, but
        # it seems better to set it after we have actually done the work
        all_dirs_dates = get_dirs_dates(self._raw_data_root,
                                  start=self._start_date,
                                  stop=self._stop_date)
        self._flat_dirs_dates_checked = all_dirs_dates
        lock.clear()

    def reduce(self):
        self.reduce_bias()
        self.reduce_dark()
        self.reduce_flat()

    def bias_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_bias()
            return self._bias_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._bias_table = None
        if self._bias_table is not None:
            return self._bias_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root,
                                        '*_bias_combined*'))
        fnames = [f for f in fnames if '.fits' in f]
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            if bad:
                ccdt = np.NAN
            else:
                ccdt = float(sfname[2])
            dates.append(date)
            ccdts.append(ccdt)
            bads.append(bad)
        self._bias_table = QTable([fnames, dates, ccdts, bads],
                                  names=('fnames', 'dates', 'ccdts', 'bad'),
                                  meta={'name': 'Bias information table'})
        return self._bias_table

    def dark_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_dark()
            return self._dark_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._dark_table = None
        if self._dark_table is not None:
            return self._dark_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root,
                                        '*_dark_combined*'))
        fnames = [f for f in fnames if '.fits' in f]
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        ccdts = []
        exptimes = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_')
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            if bad:
                ccdt = np.NAN
                exptime = np.NAN
            else:
                ccdt = float(sfname[2])
                exptime = sfname[4]
                exptime = float(exptime[:-1])
            dates.append(date)
            ccdts.append(ccdt)
            exptimes.append(exptime)
            bads.append(bad)
        self._dark_table = \
            QTable([fnames, dates, ccdts, exptimes, bads],
                   names=('fnames', 'dates', 'ccdts', 'exptimes', 'bad'),
                   meta={'name': 'Dark information table'})
        return self._dark_table

    def flat_table_create(self,
                          rescan=False, # Set to True after new biases have been added
                          autoreduce=True): # Set to False to break recursion
                                            # when first looking for 
        """Create table of bias info from calibration directory"""
        if autoreduce:
            # By default always do auto reduction to catch the latest downloads
            self.reduce_flat()
            return self._flat_table
        # If we made it here, autoreduce is guaranteed to be false
        if rescan:
            self._flat_table = None
        if self._flat_table is not None:
            return self._flat_table
        if not os.path.isdir(self._calibration_root):
            # We haven't reduced any calibration images yet and we
            # don't want to automatically do so (just yet)
            return None
        fnames = glob.glob(os.path.join(self._calibration_root,
                          '*_flat*'))
        fnames = [f for f in fnames if '.fits' in f]
        if len(fnames) == 0:
            # Catch the not autoreduce case when we still have no files
            return None
        # If we made it here, we have files to populate our table
        dates = []
        filts = []
        bads = []
        for fname in fnames:
            bfname = os.path.basename(fname)
            sfname = bfname.split('_', 1)
            date = Time(sfname[0], format='fits')
            bad = 'bad' in bfname
            filttail = sfname[1]
            filt_tail = filttail.split('_flat')
            filt = filt_tail[0]
            dates.append(date)
            filts.append(filt)
            bads.append(bad)
        self._flat_table = \
            QTable([fnames, dates, filts, bads],
                   names=('fnames', 'dates', 'filters', 'bad'),
                   meta={'name': 'Flat information table'})
        return self._flat_table

    @property
    def bias_table(self):
        return self.bias_table_create()

    @property
    def dark_table(self):
        return self.dark_table_create()

    @property
    def flat_table(self):
        return self.flat_table_create()

    def best_bias(self, fname_ccd_or_hdr, ccdt_tolerance=None):
        """Returns filename of best-matched bias for a file"""
        if ccdt_tolerance is None:
            ccdt_tolerance = self._ccdt_tolerance
        if isinstance(fname_ccd_or_hdr, Header):
            hdr = fname_ccd_or_hdr
        elif isinstance(fname_ccd_or_hdr, CCDData):
            hdr = fname_ccd_or_hdr.meta
        elif isinstance(fname_ccd_or_hdr, str):
            hdr = getheader(fname_ccd_or_hdr)
        tm = Time(hdr['DATE-OBS'], format='fits')
        ccdt = hdr['CCD-TEMP']
        # This is the entry point for reduction 
        bad = self.bias_table['bad']
        dccdts = ccdt - self.bias_table['ccdts']
        within_tol = np.abs(dccdts) < ccdt_tolerance
        good = np.logical_and(within_tol, ~bad)
        good_ccdt_idx = np.flatnonzero(good)
        if len(good_ccdt_idx) == 0:
            log.debug(f'No biases found within {ccdt_tolerance} C, broadening by factor of 2')
            return self.best_bias(hdr, ccdt_tolerance=ccdt_tolerance*2)
        ddates = tm - self.bias_table['dates']
        best_ccdt_date_idx = np.argmin(np.abs(ddates[good_ccdt_idx]))
        # unwrap
        best_ccdt_date_idx = good_ccdt_idx[best_ccdt_date_idx]
        return self._bias_table['fnames'][best_ccdt_date_idx]

    def best_dark(self,
                  fname_ccd_or_hdr,
                  ccdt_tolerance=None,
                  dark_exp_margin=None):
        """Returns filename of best-matched dark for a file"""
        if ccdt_tolerance is None:
            ccdt_tolerance = self._ccdt_tolerance
        if dark_exp_margin is None:
            dark_exp_margin = self._dark_exp_margin
        if isinstance(fname_ccd_or_hdr, Header):
            hdr = fname_ccd_or_hdr
        elif isinstance(fname_ccd_or_hdr, CCDData):
            hdr = fname_ccd_or_hdr.meta
        elif isinstance(fname_ccd_or_hdr, str):
            hdr = getheader(fname_ccd_or_hdr)
        tm = Time(hdr['DATE-OBS'], format='fits')
        ccdt = hdr['CCD-TEMP']
        exptime = hdr['EXPTIME']
        # This is the entry point for reduction 
        bad = self.dark_table['bad']
        dccdts = ccdt - self.dark_table['ccdts']
        within_tol = np.abs(dccdts) < ccdt_tolerance
        good = np.logical_and(within_tol, ~bad)
        good_ccdt_idx = np.flatnonzero(good)
        if len(good_ccdt_idx) == 0:
            log.debug(f'No darks found within {ccdt_tolerance} C, broadening by factor of 2')
            return self.best_dark(hdr, ccdt_tolerance=ccdt_tolerance*2)
        # Find the longest exposure time in our collection of darks
        # that matches our exposure.  Prefer longer exposure times by
        # dark_exp_margin
        dexptimes = exptime - self.dark_table['exptimes']
        good_exptime_idx = np.flatnonzero(
            abs(dexptimes[good_ccdt_idx]) <  dark_exp_margin)
        if len(good_exptime_idx) == 0:
            log.debug(f'No darks found with exptimes within {dark_exp_margin} s, broadening margin by factor of 2')
            return self.best_dark(hdr,
                                  ccdt_tolerance=ccdt_tolerance,
                                  dark_exp_margin=dark_exp_margin*2)
        # unwrap
        good_exptime_idx = good_ccdt_idx[good_exptime_idx]
        ddates = tm - self.dark_table['dates']
        best_exptime_date_idx = np.argmin(np.abs(ddates[good_exptime_idx]))
        # unwrap
        best_exptime_date_idx = good_exptime_idx[best_exptime_date_idx]
        return self._dark_table['fnames'][best_exptime_date_idx]
    # --> TODO: possibly put in the number of darks as a factor as
    # --> well, weighted by difference in time

    def best_flat(self, fname_ccd_or_hdr):
        """Returns filename of best-matched flat for a file"""
        if isinstance(fname_ccd_or_hdr, Header):
            hdr = fname_ccd_or_hdr
        elif isinstance(fname_ccd_or_hdr, CCDData):
            hdr = fname_ccd_or_hdr.meta
        elif isinstance(fname_ccd_or_hdr, str):
            hdr = getheader(fname_ccd_or_hdr)
        tm = Time(hdr['DATE-OBS'], format='fits')
        filt = hdr['FILTER']
        # This is the entry point for reduction 
        bad = self.flat_table['bad']
        this_filt = filt == self.flat_table['filters']
        good = np.logical_and(this_filt, ~bad)
        good_filt_idx = np.flatnonzero(good)
        if len(good_filt_idx) == 0:
            raise ValueError(f'No {filt} flats found')
        ddates = tm - self.flat_table['dates']
        best_filt_date_idx = np.argmin(np.abs(ddates[good_filt_idx]))
        # unwrap
        best_filt_date_idx = good_filt_idx[best_filt_date_idx]
        return self._flat_table['fnames'][best_filt_date_idx]

    @property
    def flat_ratio_list(self):
        # NOTE: making the flat ratio plots can only be triggered with
        # the plot_flat_ratios property

        if self._flat_ratio_list is not None:
            return self._flat_ratio_list
        
        on_list = glob.glob(os.path.join(self._calibration_root,
                                         '*on_flat.fits'))
        off_list = glob.glob(os.path.join(self._calibration_root,
                                          '*off_flat.fits'))

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
        tbl = QTable(rows=ratio_dlist)
            
        flat_ratio_list = []
        if self.plot_flat_ratios:
            f = plt.figure(figsize=[8.5, 11])
            plt.title('Sky flat ratios')
        #print('Narrow-band sky flat ratios fit to >2020-01-01 data only')
        #print('Day sky does have Na on-band emission, however, Greet 1988 PhD suggests')
        #print('it is so hard to measure with a good Fabry-Perot, that we are')
        #print('dominated by continuum')
        #print('COPY THESE INTO cormultipipe.py globals')
        for ib, band in enumerate(['Na', 'SII']):
            if self.plot_flat_ratios:
                plt.title(f'Sky flat ratios fit to > {self.stable_flat_date}')
            this_band = tbl[tbl['band'] == band]
            # Started taking flats in a more uniform way after 2020-01-01
            good_dates = this_band[this_band['date']
                                   > self.stable_flat_date]
            med_ratio = np.median(good_dates['ratio'])
            std_ratio = np.std(good_dates['ratio'])
            biweight_ratio = biweight_location(good_dates['ratio'])
            mad_std_ratio = mad_std(good_dates['ratio'])
            t_flat_ratio = {'band': band,
                            'biweight_ratio': biweight_ratio,
                            'mad_std_ratio': mad_std_ratio,
                            'med_ratio': med_ratio,
                            'std_ratio': std_ratio}
            #print(f'{band} med {med_ratio:.2f} +/- {std_ratio:.2f}')
            #print(f'{band} biweight {biweight_ratio:.2f} +/- {mad_std_ratio:.2f}')
            flat_ratio_list.append(t_flat_ratio)

            if self.plot_flat_ratios:
                ax = plt.subplot(2, 1, ib+1)
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
                #plt.plot(tbl['time'], tbl['ratio'], 'k.')
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
                plt.ylim([biweight_ratio-3*mad_std_ratio,
                          biweight_ratio+3*mad_std_ratio])
                plt.gcf().autofmt_xdate()
                outname = os.path.join(self._calibration_root,
                                       'flat__ratio_vs_time.png')
                savefig_overwrite(outname, transparent=True)
                plt.close()

        self._flat_ratio_list = flat_ratio_list
        return self._flat_ratio_list

    def flat_ratio(self, band):
        tflat_ratio = [d for d in self.flat_ratio_list
                       if d['band'] == band]
        if len(tflat_ratio) == 0:
            raise ValueError(f'Not a valid band: {band} ')
        tflat_ratio = tflat_ratio[0]
        cflat_ratio = [d for d in FLAT_RATIO_CHECK_LIST
                       if d['band'] == band]
        cflat_ratio = cflat_ratio[0]
        dflat_ratio = (np.abs(tflat_ratio['biweight_ratio']
                              - cflat_ratio['biweight_ratio']))
        if (dflat_ratio > cflat_ratio['mad_std_ratio']
            or dflat_ratio> tflat_ratio['mad_std_ratio']):
            raise ValueError(f'Derived {band} flat_ratio '
                             f'{tflat_ratio["biweight_ratio"]:.2f} '
                             f'out of range '
                             f'{cflat_ratio["biweight_ratio"]:.2f} +/-'
                             f'{cflat_ratio["mad_std_ratio"]:.2f}')
        
        return tflat_ratio['biweight_ratio'], tflat_ratio['mad_std_ratio']

class CalArgparseMixin:
    def add_calibration_start(self, 
                              default=None,
                              help=None,
                              **kwargs):
        option = 'calibration_start'
        if help is None:
            help = 'start directory/date (default: earliest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_calibration_stop(self, 
                             default=None,
                             help=None,
                             **kwargs):
        option = 'calibration_stop'
        if help is None:
            help = 'stop directory/date (default: latest)'
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

    def add_plot_flat_ratios(self, 
                             default=False,
                             help=None,
                             **kwargs):
        option = 'plot_flat_ratios'
        if help is None:
            help = 'plot flat ratios vs time'
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    
class CalArgparseHandler(CalArgparseMixin, CorArgparseHandler):
    def add_all(self, plot_flat_ratios=False):
        """Add options used in cmd"""
        self.add_plot_flat_ratios(default=plot_flat_ratios)
        self.add_raw_data_root()
        self.add_reduced_root(option='calibration_root',
                                default=CALIBRATION_ROOT)
        self.add_start(option='calibration_start')
        self.add_stop(option='calibration_stop')
        super().add_all()

    def cmd(self, args):
        super().cmd(args)
        c = Calibration(reduce=True,
                        raw_data_root=args.raw_data_root,
                        calibration_root=args.calibration_root,
                        start_date=args.calibration_start,
                        stop_date=args.calibration_stop,
                        fits_fixed_ignore=args.fits_fixed_ignore,
                        num_processes=args.num_processes,
                        mem_frac=args.mem_frac)
        return c
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run calibration to generate bias, dark, flat frames')
    aph = CalArgparseHandler(parser)
    aph.add_all(plot_flat_ratios=True)
    args = parser.parse_args()
    aph.cmd(args)



    #parser.add_argument(
    #    '--raw_data_root', help=f'raw data root (default: {RAW_DATA_ROOT})',
    #    default=RAW_DATA_ROOT)
    #parser.add_argument(
    #    '--calibration_root',
    #    help=f'calibration root (default: {CALIBRATION_ROOT})',
    #    default=CALIBRATION_ROOT)
    #parser.add_argument(
    #    '--calibration_start', help='start directory/date (default: earliest)')
    #parser.add_argument(
    #    '--calibration_stop', help='stop directory/date (default: latest)')
    #parser.add_argument(
    #    '--num_processes', type=float, default=0,
    #    help='number of subprocesses for parallelization; 0=all cores, <1 = fraction of total cores')
    #parser.set_defaults(func=calibrate_cmd)
    #
    ## Final set of commands that makes argparse work
    #args = parser.parse_args()
    ## This check for func is not needed if I make subparsers.required = True
    #if hasattr(args, 'func'):
    #    args.func(args)





    #c = Calibration(start_date='2019-09-01', stop_date='2021-12-31', reduce=True)



    #c = Calibration(start_date='2020-01-01', stop_date='2021-12-31', reduce=True)
    ##c = Calibration(start_date='2020-01-01', stop_date='2021-02-28', reduce=True)    
    ###t = c.dark_table_create(autoreduce=False, rescan=True)
    ##fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
    ##fname2 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na_off.fit'
    ##cmp = CorMultiPipe(auto=True, calibration=c,
    ##                   post_process_list=[detflux, nd_filter_mask])
    ##pout = cmp.pipeline([fname1, fname2], outdir='/tmp', overwrite=True)
    ##pout = cmp.pipeline([fname1], outdir='/tmp', overwrite=True)
    #
    ##ccd = RedCorData.read(fname1)
    ##ccd = cor_process(ccd, calibration=c, auto=True)
    ##ccd.write('/tmp/test.fits', overwrite=True)
    #
    #flat = '/data/IoIO/raw/2020-06-06/Sky_Flat-0002_B.fit'
    #cmp = CorMultiPipe(auto=True, calibration=c,
    #                   post_process_list=[flat_process])
    #pout = cmp.pipeline([flat], outdir='/tmp', overwrite=True)

    ##fname1 = '/data/IoIO/raw/20210310/HD 132052-S001-R001-C002-R.fts'
    #fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
    #pgd = RedCorData.read(fname1)
    #pgd.meta = sx694.metadata(pgd.meta)
    #pgd.meta = sx694.exp_correct(pgd.meta)
    #pgd.meta = sx694.date_obs(pgd.meta)
    #print(pgd.meta)
    ##
    #pgd = detflux(pgd)
    #print(pgd.meta)

    ##print(reduced_dir('/data/IoIO/raw/20210513'))
    ##print(reduced_dir('/data/IoIO/raw/20210513', create=True))
    ##print(reduced_dir('/data/IoIO/raw'))

    #c = Calibration(start_date='2019-02-18', stop_date='2021-12-31', reduce=True)
    #c = Calibration(start_date='2019-02-12', stop_date='2019-02-12', reduce=True)

    #c = Calibration(reduce=True)


    #f = fdict_list_collector(flat_fdict_creator, directory='/data/IoIO/raw/2019-08-25', imagetyp='flat', subdirs=CALIBRATION_SUBDIRS, glob_include=FLAT_GLOB)

    #print(f[0])

    #c = Calibration(start_date='2017-03-15', stop_date='2017-03-15', reduce=True)
    #c = Calibration(stop_date='2017-05-10', reduce=True)
    #c = Calibration(stop_date='2017-05-10')
    #c.reduce_bias()

    #c = Calibration(start_date='2020-07-11', stop_date='2020-07-11', reduce=True)
    #c = Calibration(reduce=True)

    #na_back_on = '/data/IoIO/raw/20210525/Jupiter-S007-R001-C001-Na_off.fts'
    #ccd = CorData.read(na_back_on)
    #nd_filter_mask(ccd)

    #    while rparent != ps:


    #print(reduced_dir('/data/IoIO/raw/20201111', '/tmp'))
    #print(reduced_dir2('/data/IoIO/raw/T20201111', '/tmp'))
    #print(get_dirs_dates('/data/IoIO/raw', start='2018-02-20', stop='2018-03-01'))
    #print(reduced_dir('/data/IoIO/raw', '/tmp'))

    #c = Calibration(reduce=True)
    #print(c.flat_ratio('Na'))
    ##print(c.flat_ratio('pdq'))
            
