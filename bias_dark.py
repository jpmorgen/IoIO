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

import ccdproc as ccdp

from west_aux.west_aux import add_history
from ReduceCorObs import get_dirs

# Measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx.  Value
# agrees well with Trius SX-694 advertised value (note, newer "PRO"
# model has a different gain value)
sx694_gain = 0.3
# Measured as per ioio.notebk Tue Jul 10 12:13:33 2018 MCT  jpmorgen@byted
# To be measured regularly as part of master bias creation
example_readnoise = 15.475665 * sx694_gain
# For ease of use, create some default ccdp.Keyword objects for gain
# and readnoise.  The value property in readnoise_keyword ends up
# being used like a local variable with the help of value_from=<FITS
# header> method.  The idea is the measured readnoise for a particular
# timeperiod/temperature is stored in the RDNOISE keyword of each
# master_bias and then, via the Keyword object, used in overscan
# subtraction, etc.
gain_keyword = ccdp.Keyword('GAIN', u.electron/u.adu, value=sx694_gain)
readnoise_keyword = ccdp.Keyword('RDNOISE', u.electron, example_readnoise)

#XXXdef calc_max_num_biases(max_mem_fraction=0.75,
#XXX                        num_processes=None,
#XXX                        logical=False):
#XXX    # Calculate how many bias images we can combine based on how many
#XXX    # CPUs and how much memory we want to use.  Default to using the
#XXX    # same fraction of the CPUs as the memory.  Default to using
#XXX    # "real" CPUs, not hyperthreads, since at least for reduction done
#XXX    # for Morgenthaler et al. 2019, I found going beyond the real CPU
#XXX    # limit didn't increase efficiency much.  Not psutil has better
#XXX    # facility with cpu_count than os.
#XXX    if num_processes is None:
#XXX        num_processes = max_mem_fraction * psutil.cpu_count(logical=logical)
#XXX    assert num_processes > 0
#XXX    # Let user do fraction to get at num_processes
#XXX    if num_processes == 0:
#XXX        psutil.cpu_count(logical=logical)
#XXX    if num_processes < 1:
#XXX        num_processes *= psutil.cpu_count(logical=logical)
#XXX    # The combiner doubles the size of a normal CCDData, which
#XXX    # consists of two double-precision images (primary plus error)
#XXX    # plus a single-byte mask image
#XXX    combiner_mem_per_image = 2 * NAXIS1 * NAXIS2 * (8*2 + 1)
#XXX    mem = psutil.virtual_memory()
#XXX    max_num_biases = (mem.available * max_mem_fraction /
#XXX                      combiner_mem_per_image / num_processes)
#XXX    return np.int(max_num_biases), num_processes

def ccddata_read(fname_or_ccd, *args, **kwargs):
    """Convenience function to read a FITS file into a CCDData object

    Catches the case where the raw FITS file does not have a BUNIT
    keyword, which otherwise causes CCDData.read to crash.  In this,
    ccddata_read assumes raw data units are in units of ADU.

    Also accepts ccd as a 

    Adds following FITS card if no BUNIT keyword present in metadata
        BUNIT = 'ADU' / physical units of the array values

    Parameters
    ----------
    fname_or_ccd : str or `~astropy.nddata.CCDData`

        If str, assumed to be a filename, which is read into a
        CCDData.  If ccddata, simply return the CCDData with BUNIT
        keyword possibly added
        
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
    return ccd

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
        ues = np.unique(exps)
        for ue in ues:
            exp_idx = np.flatnonzero(exps == ue)
            files = narrow_to_t['file'][exp_idx]
            full_files = [os.path.join(collection.location, f) for f in files]
            fdict_list.append({'T': mean_t,
                               'EXPTIME': ue,
                               'fnames': full_files})
    return fdict_list

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
    
def bias_combine(directory=None,
                 collection=None,
                 subdirs=['Calibration'],
                 glob_include=['Bias*', '*_bias.fit'],
                 temperature_tolerance=0.5,
                 outdir='/data/io/IoIO/reduced/bias_dark',
                 show=False,
                 gain_keyword=gain_keyword,
                 readnoise_keyword=readnoise_keyword,
                 additional_gain_comment='Measured value',
                 gain_correct=False):
    """Play with biases in a directory

    Parameters
    ----------
    gain_correct : Boolean
        Effects unit of stored images.  True: units of electron.
        False: unit of ADU.  Default: False
    """
    gain = gain_keyword.value.value
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
    log.debug('Biases found in ' + directory)
    # Loop through each member of our fdict_list, preparing summary
    # plot and combining biases
    for fdict in fdict_list:
        lccds = []
        stats = []
        jds = []
        for fname in fdict['fnames']:
            ccd = ccddata_read(fname)
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

        # Make sure outdir exists
        if not os.path.exists(outdir):
            os.mkdir(outdir)
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
        ### suggests that there is a separate ccdp.combine function
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
        # This is ultimately where we set the gain for all the reductions
        im.meta[gain_keyword.name] = (gain,
                                      repr(gain_keyword.unit) + ' '
                                      + additional_gain_comment)
        if gain_correct:
            im = ccdp.gain_correct(im, gain_keyword)
            gain = 1
        # Prepare to write
        fname = fbase + '_combined_bias.fits'
        # Collect metadata, preparing to write to FITS header and CSV
        # file.  FITS wants a tuple, CSV wants a dict
        # Note that std and mean behave differently on masked arrays,
        # returning a masked object themselves.  The output of
        # Combiner is a masked array.  The combine function does not
        # do that.  The later is appropriate for biases only
        #std =  np.asscalar(np.std(im).data   )
        std =  np.std(im)*gain
        med =  np.median(im)*gain
        #mean = np.asscalar(np.mean(im).data  )
        mean = np.mean(im)*gain
        tmin = np.min(im)*gain
        tmax = np.max(im)*gain
        av_rdnoise = np.mean(df['rdnoise'])
        print('std, mean, med, tmin, tmax (electron)')
        print(std, mean, med, tmin, tmax)
        im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
        im.meta['CCD-TEMP'] = (mean_t, 'average CCD temperature for combined biases')
        im.meta[readnoise_keyword.name] \
            = (av_rdnoise, 'readnoise measured during master_bias creation (electron)')
        im.meta['STD'] = (std, 'Standard deviation of image (electron)')
        im.meta['MEDIAN'] = (med, 'Median of image (electron)')
        im.meta['MEAN'] = (mean, 'Mean of image (electron)')
        im.meta['MIN'] = (tmin, 'Min of image (electron)')
        im.meta['MAX'] = (tmax, 'Max of image (electron)')
        im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (ADU)')
        im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
        im.meta['NCOMBINE'] = (len(lccds), 'Number of biases combined')
        # Record each bias filename
        for i, f in enumerate(fdict['fnames']):
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
        # Always display image in electrons
        impl = plt.imshow(im.multiply(gain), origin='lower', cmap=plt.cm.gray,
                          filternorm=0, interpolation='none',
                          vmin=med-std, vmax=med+std)
        plt.title('CCDT = {} C on {} (electrons)'.format(this_ccdt, this_dateb))
        plt.savefig((fbase + '_combined_bias.png'), transparent=True)
        if show:
            plt.show()
        plt.close()

#XXXdef old_bias_combine(directory=None,
#XXX                 collection=None,
#XXX                 outdir='/data/io/IoIO/reduced/bias_dark',
#XXX                 show=False,
#XXX                 temperature_tolerance=0.5,
#XXX                 gain_keyword=gain_keyword,
#XXX                 readnoise_keyword=readnoise_keyword,
#XXX                 additional_gain_comment='Measured value',
#XXX                 gain_correct=False):
#XXX    """Play with biases in a directory
#XXX
#XXX    Parameters
#XXX    ----------
#XXX    gain_correct : Boolean
#XXX        Effects unit of stored images.  True: units of electron.
#XXX        False: unit of ADU.  Default: False
#XXX    """
#XXX    gain = gain_keyword.value.value
#XXX    if collection is None:
#XXX        if not os.path.isdir(directory):
#XXX            log.debug('Not a directory, skipping: ' + directory)
#XXX            return False
#XXX        # Speed things up considerably by globbing the bias fnames, of
#XXX        # which I only have two types: those recorded by ACP (Bias*)
#XXX        # and those recorded by MaxIm (Bias* and *_bias.fit)
#XXX        collection = ccdp.ImageFileCollection(directory,
#XXX                                                 glob_include='Bias*')
#XXX        if collection.summary is None:
#XXX            collection = ccdp.ImageFileCollection(directory,
#XXX                                                     glob_include='*_bias.fit')
#XXX        if collection.summary is None:
#XXX            subdir = os.path.join(directory, 'Calibration')
#XXX            if not os.path.isdir(subdir):
#XXX                return False
#XXX            return bias_combine(subdir)            
#XXX    if collection.summary is None:
#XXX        return False
#XXX    log.debug('found biases in ' + directory)
#XXX    #if max_num_biases is None:
#XXX    #    # Make sure we don't overwhelm our memory
#XXX    #    max_num_biases, num_processes = calc_max_num_biases(num_processes=1)
#XXX    # Prepare to write output
#XXX    if not os.path.exists(outdir):
#XXX        os.mkdir(outdir)
#XXX    #bias_summary_fname = os.path.join(outdir, 'bias_summary.csv')
#XXX    
#XXX    # Find the distinct sets of temperatures within temperature_tolerance
#XXX    
#XXX    isbias = collection.summary['imagetyp'] == 'BIAS'
#XXX    ts = collection.summary['ccd-temp'][isbias]
#XXX    bias_fnames = collection.summary['file'][isbias]
#XXX    # ccd-temp is recorded as a string.  Convert it to a number so
#XXX    # we can sort +/- values properly
#XXX    ts = np.asarray(ts)
#XXX    # Get the sort indices so we can extract fnames in proper order
#XXX    tsort_idx = np.argsort(ts)
#XXX    ts = ts[tsort_idx]
#XXX    bias_fnames = bias_fnames[tsort_idx]
#XXX    # Spot jumps in t and translate them into slices into ts
#XXX    dts = ts[1:] - ts[0:-1]
#XXX    jump = np.flatnonzero(dts > temperature_tolerance)
#XXX    slices = np.append(0, jump+1)
#XXX    slices = np.append(slices, -1)
#XXX    for it in range(len(slices)-1):
#XXX        # Loop through each temperature set
#XXX        flist = [os.path.join(collection.location, f)
#XXX                 for f in bias_fnames[slices[it]:slices[it+1]]]
#XXX        lccds = [ccddata_read(f) for f in flist]
#XXX        dark_ccds = []
#XXX        stats = []
#XXX        jds = []
#XXX        this_ts = ts[slices[it]:slices[it+1]]
#XXX        for iccd, ccd in enumerate(lccds):
#XXX            im = ccd.data
#XXX            s = im.shape
#XXX            # Create uncertainty image
#XXX            diffs2 = (im[1:] - im[0:-1])**2
#XXX            rdnoise = np.sqrt(biweight_location(diffs2))
#XXX            uncertainty = np.multiply(rdnoise, np.ones(s))
#XXX            ccd.uncertainty = StdDevUncertainty(uncertainty)
#XXX            # Spot guider biases and binned main camera biases.  Note
#XXX            # Pythonic C index ordering
#XXX            if s != (NAXIS2, NAXIS1):
#XXX                log.debug('bias wrong shape: ' + str(s))
#XXX                continue
#XXX            # Spot biases recorded when too bright.
#XXX            m = np.asarray(s)/2 # Middle of CCD
#XXX            q = np.asarray(s)/4 # 1/4 point
#XXX            m = m.astype(int)
#XXX            q = q.astype(int)
#XXX            # --> check lowest y too, since large filters go all the
#XXX            # --> way to the edge See 20200428 dawn biases
#XXX            dark_patch = im[m[0]-50:m[0]+50, 0:100]
#XXX            light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
#XXX            mdp = np.median(dark_patch)
#XXX            mlp = np.median(light_patch)
#XXX            if (np.median(light_patch) - np.median(dark_patch) > 1):
#XXX                log.debug('bias recorded during light conditions: ' +
#XXX                          flist[iccd])
#XXX                log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
#XXX                continue
#XXX            dark_ccds.append(ccd)
#XXX            # Prepare to create a pandas data frame to track relevant
#XXX            # quantities
#XXX            tm = Time(ccd.meta['DATE-OBS'], format='fits')
#XXX            tt = tm.tt.datetime
#XXX            jds.append(tm.jd)
#XXX            stats.append({'time': tt,
#XXX                          'ccdt': this_ts[iccd],
#XXX                          'median': np.median(im),
#XXX                          'mean': np.mean(im),
#XXX                          'std': np.std(im)*gain,
#XXX                          'rdnoise': rdnoise*gain,
#XXX                          'min': np.min(im),  
#XXX                          'max': np.max(im)})
#XXX        lccds = dark_ccds
#XXX        mean_t = np.mean(this_ts)
#XXX        if len(stats) < 3:
#XXX            log.debug('Not enough good biases found at CCDT = {} C in {}'.format(mean_t, collection.location))
#XXX            continue
#XXX        df = pd.DataFrame(stats)
#XXX        tm = Time(np.mean(jds), format='jd')
#XXX        this_date = tm.fits
#XXX        this_dateb = this_date.split('T')[0]
#XXX        this_ccdt = '{:.2f}'.format(mean_t)
#XXX        f = plt.figure(figsize=[8.5, 11])
#XXX
#XXX        # In the absence of a formal overscan region, this is the best
#XXX        # I can do
#XXX        medians = df['median']
#XXX        overscan = np.mean(medians)
#XXX
#XXX        ax = plt.subplot(5, 1, 1)
#XXX        plt.title('CCDT = {} C on {}'.format(this_ccdt, this_dateb))
#XXX        ax.yaxis.set_minor_locator(AutoMinorLocator())
#XXX        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
#XXX        plt.plot(df['time'], df['max'], 'k.')
#XXX        plt.ylabel('max (ADU)')
#XXX
#XXX        ax = plt.subplot(5, 1, 2)
#XXX        ax.yaxis.set_minor_locator(AutoMinorLocator())
#XXX        ax.tick_params(which='both', bottom=True, top=True, left=True, right=False)
#XXX        plt.plot(df['time'], df['median'], 'k.')
#XXX        plt.plot(df['time'], df['mean'], 'r.')
#XXX        plt.ylabel('median & mean (ADU)')
#XXX        plt.legend(['median', 'mean'])
#XXX        secax = ax.secondary_yaxis \
#XXX            ('right',
#XXX             functions=(lambda adu: (adu - overscan)*gain,
#XXX                        lambda e: e/gain + overscan))
#XXX        secax.set_ylabel('Electrons')
#XXX
#XXX        ax=plt.subplot(5, 1, 3)
#XXX        ax.yaxis.set_minor_locator(AutoMinorLocator())
#XXX        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
#XXX        plt.plot(df['time'], df['min'], 'k.')
#XXX        plt.ylabel('min (ADU)')
#XXX
#XXX        ax=plt.subplot(5, 1, 4)
#XXX        ax.yaxis.set_minor_locator(AutoMinorLocator())
#XXX        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
#XXX        plt.plot(df['time'], df['std'], 'k.')
#XXX        plt.ylabel('std (electron)')
#XXX
#XXX        ax=plt.subplot(5, 1, 5)
#XXX        ax.yaxis.set_minor_locator(AutoMinorLocator())
#XXX        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
#XXX        plt.plot(df['time'], df['rdnoise'], 'k.')
#XXX        plt.ylabel('rdnoise (electron)')
#XXX
#XXX        plt.gcf().autofmt_xdate()
#XXX
#XXX
#XXX        fbase = os.path.join(outdir, this_dateb + '_ccdT_' + this_ccdt)
#XXX        plt.savefig((fbase + '_vs_time.png'), transparent=True)
#XXX        if show:
#XXX            plt.show()
#XXX        plt.close()
#XXX
#XXX        # At the 0.5 deg level, there seems to be no correlation between T and bias level
#XXX        #plt.plot(df['ccdt'], df['mean'], 'k.')
#XXX        #plt.xlabel('ccdt')
#XXX        #plt.ylabel('mean')
#XXX        #plt.show()
#XXX        
#XXX        ### Some nights were recorded with a bad ACP plan that
#XXX        ### recursively multiplied the number of biases
#XXX        ### -->
#XXX        ### https://mwcraig.github.io/ccd-as-book/02-04-Combine-bias-images-to-make-master.html
#XXX        ### suggests that there is a separate ccdp.combine function
#XXX        ### that takes care of this with mem_limit
#XXX        ##if len(stats) > max_num_biases:
#XXX        ##    log.debug('Limiting to {} the {} biases found at CCDT = {} C in {}'.format(max_num_biases, len(stats), mean_t, collection.location))
#XXX        ##    # Thanks to https://note.nkmk.me/en/python-pandas-list/ to
#XXX        ##    # get this into a list.  The Series object works OK
#XXX        ##    # internally to Pandas but not ouside
#XXX        ##    best_idx = df['mean'].argsort().values.tolist()
#XXX        ##    df = df.iloc[best_idx[:max_num_biases]]
#XXX        ##    lccds = [lccds[i] for i in best_idx[:max_num_biases]]
#XXX        ##    flist = [flist[i] for i in best_idx[:max_num_biases]]
#XXX            
#XXX        # Go through each image and subtract the median, since that
#XXX        # value wanders as a function of ambient (not CCD)
#XXX        # temperature.  To use ccd.subtract, type must match type of
#XXX        # array.  And they are integers anyway
#XXX
#XXX        # We need another list to keep track of the
#XXX        # overscan-subtracted ccds, since ccd.subtract returns a copy
#XXX        # which doesn't get put back into the original lccd.  And when
#XXX        # I tried to put it in the original lccd wierd things happened
#XXX        # anyway
#XXX        os_lccds = []
#XXX        for ccd, m in zip(lccds, medians):
#XXX            #ccd.data = ccd.data - m
#XXX            ccd = ccd.subtract(m*u.adu, handle_meta='first_found')
#XXX            os_lccds.append(ccd)
#XXX        lccds = os_lccds
#XXX
#XXX        ### Use ccdproc Combiner object iteratively as per example to
#XXX        ### mask out bad pixels
#XXX        ##combiner = ccdp.Combiner(lccds)
#XXX        ##old_n_masked = -1  # dummy value to make loop execute at least once
#XXX        ##new_n_masked = 0
#XXX        ##while (new_n_masked > old_n_masked):
#XXX        ##    #combiner.sigma_clipping(low_thresh=2, high_thresh=5,
#XXX        ##    #                        func=np.ma.median)
#XXX        ##    # Default to 1 std and np.ma.mean for func
#XXX        ##    combiner.sigma_clipping()
#XXX        ##    old_n_masked = new_n_masked
#XXX        ##    new_n_masked = combiner.data_arr.mask.sum()
#XXX        ##    print(old_n_masked, new_n_masked)
#XXX        ###Just one iteration for testing
#XXX        ###combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.median)
#XXX        ##combiner.sigma_clipping(low_thresh=1, high_thresh=1, func=np.ma.mean)
#XXX        ## I prefer average to get sub-ADU accuracy.  Median outputs in
#XXX        ## double-precision anyway, so it doesn't save space
#XXX        #combined_average = combiner.average_combine()
#XXX
#XXX        # Use ccdp.combine since it enables memory management by
#XXX        # breaking up images to smaller chunks (better than throwing
#XXX        # images away)
#XXX        mem = psutil.virtual_memory()
#XXX        im = \
#XXX            ccdp.combine(lccds,
#XXX                            method='average',
#XXX                            sigma_clip=True,
#XXX                            sigma_clip_low_thresh=5,
#XXX                            sigma_clip_high_thresh=5,
#XXX                            sigma_clip_func=np.ma.median,
#XXX                            sigma_clip_dev_func=mad_std,
#XXX                            mem_limit=mem.available*0.6)
#XXX        # This is ultimately where we set the gain for all the reductions
#XXX        im.meta[gain_keyword.name] = (gain,
#XXX                                      repr(gain_keyword.unit) + ' '
#XXX                                      + additional_gain_comment)
#XXX        if gain_correct:
#XXX            im = ccdp.gain_correct(im, gain_keyword)
#XXX            gain = 1
#XXX        # Prepare to write
#XXX        fname = fbase + '_combined_bias.fits'
#XXX        # Collect metadata, preparing to write to FITS header and CSV
#XXX        # file.  FITS wants a tuple, CSV wants a dict
#XXX        # Note that std and mean behave differently on masked arrays,
#XXX        # returning a masked object themselves.  The output of
#XXX        # Combiner is a masked array.  The combine function does not
#XXX        # do that.  The later is appropriate for biases only
#XXX        #std =  np.asscalar(np.std(im).data   )
#XXX        std =  np.std(im)*gain
#XXX        med =  np.median(im)*gain
#XXX        #mean = np.asscalar(np.mean(im).data  )
#XXX        mean = np.mean(im)*gain
#XXX        tmin = np.min(im)*gain
#XXX        tmax = np.max(im)*gain
#XXX        av_rdnoise = np.mean(df['rdnoise'])
#XXX        print('std, mean, med, tmin, tmax (electron)')
#XXX        print(std, mean, med, tmin, tmax)
#XXX        im.meta['DATE-OBS'] = (this_date, 'Average of DATE-OBS from set of biases')
#XXX        im.meta['CCD-TEMP'] = (mean_t, 'average CCD temperature for combined biases')
#XXX        im.meta[readnoise_keyword.name] \
#XXX            = (av_rdnoise, 'readnoise measured during master_bias creation (electron)')
#XXX        im.meta['STD'] = (std, 'Standard deviation of image (electron)')
#XXX        im.meta['MEDIAN'] = (med, 'Median of image (electron)')
#XXX        im.meta['MEAN'] = (mean, 'Mean of image (electron)')
#XXX        im.meta['MIN'] = (tmin, 'Min of image (electron)')
#XXX        im.meta['MAX'] = (tmax, 'Max of image (electron)')
#XXX        im.meta['HIERARCH OVERSCAN_VALUE'] = (overscan, 'Average of raw bias medians (ADU)')
#XXX        im.meta['HIERARCH SUBTRACT_OVERSCAN'] = (True, 'Overscan has been subtracted')
#XXX        #im.meta['NCOMBINE'] = (len(lccds), 'Number of biases combined')
#XXX        # Record each bias filename
#XXX        for i, f in enumerate(flist):
#XXX            im.meta['FILE{0:02}'.format(i)] = f
#XXX        # Prepare to write
#XXX        fname = fbase + '_combined_bias.fits'
#XXX        add_history(im.meta,
#XXX                    'Combining NCOMBINE biases indicated in FILENN')
#XXX        # Leave these large for fast calculations downstream and make
#XXX        # final results that primarily sit on disk in bulk small
#XXX        #im.data = im.data.astype('float32')
#XXX        #im.uncertainty.array = im.uncertainty.array.astype('float32')
#XXX        im.write(fname, overwrite=True)
#XXX        # Always display image in electrons
#XXX        impl = plt.imshow(im.multiply(gain), origin='lower', cmap=plt.cm.gray,
#XXX                          filternorm=0, interpolation='none',
#XXX                          vmin=med-std, vmax=med+std)
#XXX        plt.title('CCDT = {} C on {} (electrons)'.format(this_ccdt, this_dateb))
#XXX        plt.savefig((fbase + '_combined_bias.png'), transparent=True)
#XXX        if show:
#XXX            plt.show()
#XXX        plt.close()
#XXX        #return im
#XXX        
#XXX        #fieldnames = list(row.keys())
#XXX        #if not os.path.exists(bias_summary_fname):
#XXX        #    with open(bias_summary_fname, 'w', newline='') as csvfile:
#XXX        #        csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
#XXX        #                               quoting=csv.QUOTE_NONNUMERIC)
#XXX        #        csvdw.writeheader()
#XXX        #with open(bias_summary_fname, 'a', newline='') as csvfile:
#XXX        #    csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
#XXX        #                           quoting=csv.QUOTE_NONNUMERIC)
#XXX        #    csvdw.writerow(row)
#XXX
#XXXdef metadata(im, hdr):
#XXX    return hdr, row
#XXX
#XXXdef bias_closest_T():
#XXX    pass

def hist_of_im(im, binsize=1):
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
    #plt.plot(centers, hist)
    #plt.show()
    return (hist, centers)

def overscan_estimate(ccd_in, master_bias=None,
                      readnoise_key=readnoise_keyword,
                      gain_key=gain_keyword, binsize=None,
                      min_width=1, max_width=8, box_size=100,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    Uses the minimum of: (1) the first peak in the histogram of the
    image or (2) the minimum of the median of four boxes at the
    corners of the image.  Updates ccd metadata with OVERSCAN and
    OVERSCAN_METHOD keywords and, if needed, adds BUNIT keyword

    Works best if bias shape (particularly bias ramp) is subtracted
    first.  Will subtract bias if bias is supplied and has not been
    subtracted.

    Parameters
    ----------
    ccd_in : `~astropy.nddata.CCDData` or filename
        Image from which to extract overscan estimate

    master_bias : `~astropy.nddata.CCDData`, filename, or None
        Bias to subtract from ccd before estimate is calculated.
        Improves accruacy by removing bias ramp.  Bias can be in units
        of ADU or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

    readnoise_key : ccdproc.Keyword
        If bias supplied, used to extract readnoise from bias header.
        Otherwise value property used.
        Default = ``readnoise_keyword``.

    --> Note these next two are not as flexible as they can be because
    the value in the bias header will always override the value passed
    in the parameter.  Unlikely to need to do it that way, so I haven't
    bothered to write the code to make that convenient
    gain_key :  ccdproc.Keyword
        If bias supplied, used to extract gain from bias header.  
        Otherwise value property used.
        Default = ``gain_keyword``.

    binsize: float or None, optional
        The binsize to use for the histogram.  If None, binsize is 
        (readnoise in ADU)/4.  Default = None

    min_width : int, optional
        Minimum width peak to search for in histogram.  Keep in mind
        histogram bins are binsize ADU wide.  Default = 1

    max_width : int, optionsl
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
    ccd = ccddata_read(ccd_in)
    # For now don't get fancy with unit conversion
    assert ccd.unit is u.adu
    if master_bias is None:
        pass
    elif isinstance(master_bias, CCDData):
        # Make a copy because we are going to mess with it
        bias = master_bias.copy()
    else:
        bias = ccddata_read(master_bias)
    if isinstance(bias, CCDData):
        readnoise = readnoise_key.value_from(bias.meta)
        gain = gain_key.value_from(bias.meta)
    else:
        readnoise = readnoise_key.value
        gain = gain_key.value
    if isinstance(bias, CCDData):
        # Make sure bias hasn't been subtracted before
        if ccd.header.get('subtract_bias') is None:
            if bias.unit is u.electron:
                # Convert bias back to ADU for subtraction, if needed
                bias = bias.divide(gain)
            ccd = ccdp.subtract_bias(ccd, bias)
            log.info('overscan_estimate: subtracted bias to improve overscan estimate')
    # Corners method
    s = ccd.shape
    bs = box_size
    c00 = biweight_location(ccd[0:bs,0:bs])
    c10 = biweight_location(ccd[s[0]-bs:s[0],0:bs])
    c01 = biweight_location(ccd[0:bs,s[1]-bs:s[1]])
    c11 = biweight_location(ccd[s[0]-bs:s[0],s[1]-bs:s[1]])
    corners_method = min(c00, c10, c01, c11)
    # Histogram method.  The coronagraph creates a margin of
    # un-illuminated pixels on the CCD.  These are great for
    # estimating the bias and scattered light for spontanous
    # subtraction.  The ND filter provides a similar peak after bias
    # subutraction (or, rather, it is the second such peak).  Note
    # that the 1.25" filters do a better job at this than the 2"
    # filters but with carefully chosen parameters, the first small
    # peak can be spotted.  
    if binsize is None:
        # Calculate binsize based on readnoise in ADU, but oversample
        # by 4.  Note need to convert from Quantity to float
        binsize = readnoise/gain/4.
        binsize = binsize.value
    im_hist, im_hist_centers = hist_of_im(ccd, binsize)
    # The arguments to linspace are the critical parameters I played
    # with together with binsize to get the first small peak to be recognized
    im_peak_idx = signal.find_peaks_cwt(im_hist,
                                        np.linspace(min_width, max_width))
    hist_method = im_hist_centers[im_peak_idx[0]]
    overscan_methods = ['corners', 'histogram']
    overscan_values = np.asarray((corners_method, hist_method))
    o_idx = np.argmin(overscan_values)
    overscan = overscan_values[o_idx]
    meta = ('HIERARCH OVERSCAN_METHOD', overscan_methods[o_idx])
    if show:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9))
        ccds = ccd.subtract(1000*u.adu)
        range = 5*readnoise/gain
        range = range.value
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
    return overscan, meta

def subtract_overscan(ccd_in, *args, **kwargs):
    """Subtract overscan in absense of formal overscan region.  

    This is a wrapper around overscan_estimate in case I want to make
    overscan estimation more complicated by linking files within a
    directory.  Note: ccdproc's native subtract_overscan function can't be
    used because it assumes the overscan region is specified by a
    simple rectangle.

    """
    ccd = ccddata_read(ccd_in)
    overscan, meta = overscan_estimate(ccd, *args, **kwargs)
    ccd = ccd.subtract(overscan*u.adu, handle_meta='first_found')
    ccd.meta[meta[0]] = meta[1]
    ccd.meta['HIERARCH OVERSCAN_VALUE'] = overscan
    ccd.meta['HIERARCH SUBTRACT_OVERSCAN'] \
        = (True, 'Overscan has been subtracted')
    return ccd

#Xdef bias_subtract(fname, bias_fname, out_fname, show=False,
#X                  overscan=None, gain=None):
#X    # Think about making overscan possibly a directory or collection
#X    # to trigger calculation
#X    try:
#X        im = CCDData.read(fname)
#X    except Exception as e: 
#X        im = CCDData.read(fname, unit="adu")
#X        im.meta['BUNIT'] = ('ADU', 'Unit of pixel value')
#X    if show:
#X        im_orig = im.copy()
#X    bias = CCDData.read(bias_fname)
#X    if overscan is None:
#X        overscan = overscan_estimate(im, bias)
#X    im = im.subtract(overscan*u.adu, handle_meta='first_found')
#X    im = ccdp.subtract_bias(im, bias)
#X    if gain is None:
#X        gain = global_gain
#X    im.meta['GAIN'] = (gain, 'Measured electrons per ADU')
#X    g = ccdp.Keyword('GAIN', u.electron/u.adu)
#X    im = ccdp.gain_correct(im, g)
#X    im.meta['BUNIT'] = 'electron'
#X    im.meta['BIASNAME'] = bias_fname
#X    add_history(im.meta,
#X                'Subtracted OVERSCAN, BIASNAME, and gain corrected')
#X    #std =  np.asscalar(np.std(im).data   )
#X    std =  np.std(im)
#X    med =  np.median(im)
#X    #mean = np.asscalar(np.mean(im).data  )
#X    mean = np.mean(im)
#X    tmin = np.min(im)
#X    tmax = np.max(im)
#X    rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
#X    im.write(out_fname, overwrite=True)
#X    #print(im.header)
#X    print('std, rdnoise, mean, med, min, max')
#X    print(std, rdnoise, mean, med, tmin, tmax)
#X    if show:
#X        im_orig = im_orig - overscan
#X        med = np.median(im_orig[0:100, 0:100])
#X        std = np.std(im_orig[0:100, 0:100])
#X        #impl = plt.imshow(im, cmap=plt.cm.gray, filternorm=0,
#X        #                  interpolation='none', vmin=med-std, vmax=med+std)
#X        norm = ImageNormalize(stretch=SqrtStretch())
#X        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9))
#X        #ax1.imshow(im_orig, origin='lower', cmap='Greys_r', norm=norm)
#X        ax1.imshow(im_orig, origin='lower', cmap=plt.cm.gray, filternorm=0,
#X                   interpolation='none', vmin=med-std, vmax=med+std)
#X        ax1.set_title('Raw Data Minus Overscan')
#X        #ax2.imshow(im, origin='lower', cmap='Greys_r', norm=norm)
#X        med = np.median(im[0:100, 0:100])
#X        std = np.std(im[0:100, 0:100])
#X        ax2.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#X                   interpolation='none', vmin=med-std, vmax=med+std)
#X        ax2.set_title('Bias Subtracted')
#X        plt.show()
#X        plt.close()
#X    return im
    
# Copy and tweak ccdp.ccd_process
#def ccd_process(ccd, oscan=None, trim=None, error=False, master_bias=None,
#                dark_frame=None, master_flat=None, bad_pixel_mask=None,
#                gain=None, readnoise=None, oscan_median=True, oscan_model=None,
#                min_value=None, dark_exposure=None, data_exposure=None,
#                exposure_key=None, exposure_unit=None,
#                dark_scale=False, gain_corrected=True):
def ccd_process(ccd, oscan=None, error=False, master_bias=None,
                gain=None, readnoise=None, *args, **kwargs):
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

    # Bias subtract first to improve overscan

    # apply overscan correction unique to the IoIO SX694 CCD
    if oscan is True:
        nccd = subtract_overscan(nccd, master_bias=master_bias,
                                 *args, **kwargs)
    elif oscan is None or oscan is False:
        pass
    else:
        raise TypeError('oscan is not None, True or False')

    # Here is where we "infect" our processed data with gain and
    # readnoise metadata and, while we are at it, extract these values
    # for use in ccdp.ccd_process
    if master_bias is not None:
        # --> document these True options in help
        if gain is True:
            gn = gain_keyword.name
            gain = gain_keyword.value_from(master_bias.meta)
            gc = master_bias.meta.comments[gn]
            nccd.meta[gn] = (gain.value, gc)
            if master_bias.unit == u.electron/u.adu:
                gain_corrected = True
        if error is True:
            rn = readnoise_keyword.name
            readnoise = readnoise_keyword.value_from(master_bias.meta)
            rc = master_bias.meta.comments[rn]
            nccd.meta[rn] = (readnoise.value, rc)
        
    # create the error frame.  I can't trim my overscan, so there are
    # lots of pixels at the overscan level.  After overscan and bias
    # subtraction, many of them that are probably normal statitical
    # outliers are negative enough to overwhelm the readnoise in the
    # deviation calculation.  But I don't want the error estimate on
    # them to be NaN, since the error is really the readnoise.
    if error and gain is not None and readnoise is not None:
        nccd = ccdp.create_deviation(nccd, gain=gain, readnoise=readnoise,
                                disregard_nan=True)
    elif error and (gain is None or readnoise is None):
        raise ValueError(
            'gain and readnoise must be specified to create error frame.')
    

    return ccdp.ccd_process(nccd, master_bias=master_bias,
                            gain=gain, *args, **kwargs)

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
            ccd = ccddata_read(fname)
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
        combined_average = \
            ccdp.combine(lccds,
                         method='average',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=mem.available*0.6)
        im = combined_average
        # Create a mask that blanks out all our pixels that are just readnoise
        measured_readnoise = im.meta['RDNOISE']
        mask = im < measured_readnoise * mask_threshold
        im.mask = mask
        n_hot_pix = np.count_nonzero(mask == 0)
        #std =  np.std(im)
        std = np.asscalar(np.std(im).data)
        med =  np.asscalar(np.median(im).data)
        #mean = np.ma.mean(im)
        mean = np.asscalar(np.mean(im).data  )
        tmin = np.min(im)
        tmax = np.max(im)
        rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
        print('combined dark statistics for ' + fbase)
        print('std, rdnoise, mean, med, min, max, n_hot_pix')
        print(std, rdnoise, mean, med, tmin, tmax, n_hot_pix)
        im.meta['STD'] = (std, 'Standard deviation of image (electron)')
        im.meta['MEDIAN'] = (med, 'Median of image (electron)')
        im.meta['MEAN'] = (mean, 'Mean of image (electron)')
        im.meta['MIN'] = (tmin, 'Min of image (electron)')
        im.meta['MAX'] = (tmax, 'Max of image (electron)')
        im.meta['NCOMBINE'] = (len(lccds), 'Number of biases combined')
        add_history(im.meta,
                    'Combining NCOMBINE biases indicated in FILENN')
        ### Create a mask that blanks out all our pixels that are just readnoise
        ##measured_readnoise = im.meta['RDNOISE']
        ##mask = im < measured_readnoise * mask_threshold
        ##im.mask = mask
        ##n_hot_pix = np.count_nonzero(mask)
        im.meta['HIERARCH MASK_THRESHOLD'] \
            = (mask_threshold, 'Units of readnoise')
        im.meta['N_HOTPIX'] \
            = (n_hot_pix, 'number of pixels with dark current')
        add_history(im.meta,
                    'Masking pixels below MASK_THRESHOLD')        
        # Record each bias filename
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


#XX
#XXdef old_dark_combine(directory=None,
#XX                 collection=None,
#XX                 master_bias=None, # This is going to have to be True or something like that to trigger search for optimum
#XX                 outdir='/data/io/IoIO/reduced/bias_dark',
#XX                 show=False,
#XX                 temperature_tolerance=0.5):
#XX    if collection is None:
#XX        if not os.path.isdir(directory):
#XX            log.debug('Not a directory, skipping: ' + directory)
#XX            return False
#XX        # Speed things up considerably by globbing the dark fnames, of
#XX        # which I only have two types: those recorded by ACP (Dark*)
#XX        # and those recorded by MaxIm (Dark* and *_dark.fit)
#XX        collection = ccdp.ImageFileCollection(directory,
#XX                                                 glob_include='Dark*')
#XX        if collection.summary is None:
#XX            collection = ccdp.ImageFileCollection(directory,
#XX                                                     glob_include='*_dark.fit')
#XX        if collection.summary is None:
#XX            subdir = os.path.join(directory, 'Calibration')
#XX            if not os.path.isdir(subdir):
#XX                return False
#XX            return dark_combine(subdir)            
#XX    if collection.summary is None:
#XX        return False
#XX    log.debug('found darks in ' + directory)
#XX    if not os.path.exists(outdir):
#XX        os.mkdir(outdir)
#XX
#XX    
#XX    # Redo list logic to be more Pythonic and include exposure times
#XX    imagetyp = 'DARK'
#XX    our_imagetyp = collection.summary['imagetyp'] == imagetyp
#XX    narrow_to_imagetyp = collection.summary[our_imagetyp]
#XX    ts = narrow_to_imagetyp['ccd-temp']
#XX    # ccd-temp is recorded as a string.  Convert it to a number so
#XX    # we can sort +/- values properly
#XX    ts = np.asarray(ts)
#XX    # Get the sort indices so we can extract fnames in proper order
#XX    tsort_idx = np.argsort(ts)
#XX    ts = ts[tsort_idx]
#XX    # Spot jumps in t and translate them into slices into ts
#XX    dts = ts[1:] - ts[0:-1]
#XX    jump = np.flatnonzero(dts > temperature_tolerance)
#XX    tslices = np.append(0, jump+1)
#XX    tslices = np.append(tslices, -1)
#XX    fdict_list = []
#XX    for it in range(len(tslices)-1):
#XX        this_ts = ts[tslices[it]:tslices[it+1]]
#XX        mean_t = np.mean(this_ts)
#XX        print(mean_t)
#XX        narrow_to_t = narrow_to_imagetyp[tslices[it]:tslices[it+1]]
#XX        exps = narrow_to_t['exptime']
#XX        ues = np.unique(exps)
#XX        for ue in ues:
#XX            exp_idx = np.flatnonzero(exps == ue)
#XX            files = narrow_to_t['file'][exp_idx]
#XX            fdict_list.append({'T': mean_t,
#XX                               'EXPTIME': ue,
#XX                               'fnames': files})
#XX    print(fdict_list)
#XX    return
#XX    
#XX
#XX    # Find the distinct sets of temperatures within temperature_tolerance
#XX    # --> doing this for the second time.  Might want to make it some
#XX    # sort of function.  For darks I also need to collect times
#XX    
#XX    isdark = collection.summary['imagetyp'] == 'DARK'
#XX    ts = collection.summary['ccd-temp'][isdark]
#XX    dark_fnames = collection.summary['file'][isdark]
#XX    # ccd-temp is recorded as a string.  Convert it to a number so
#XX    # we can sort +/- values properly
#XX    ts = np.asarray(ts)
#XX    # Get the sort indices so we can extract fnames in proper order
#XX    tsort_idx = np.argsort(ts)
#XX    ts = ts[tsort_idx]
#XX    dark_fnames = dark_fnames[tsort_idx]
#XX    # Spot jumps in t and translate them into slices into ts
#XX    dts = ts[1:] - ts[0:-1]
#XX    jump = np.flatnonzero(dts > temperature_tolerance)
#XX    slice = np.append(0, jump+1)
#XX    slice = np.append(slice, -1)
#XX    for it in range(len(slice)-1):
#XX        # Loop through each temperature set
#XX        flist = [os.path.join(collection.location, f)
#XX                 for f in dark_fnames[slice[it]:slice[it+1]]]
#XX        lccds = [ccddata_read(f) for f in flist]
#XX        dark_ccds = []
#XX        stats = []
#XX        jds = []
#XX        this_ts = ts[slice[it]:slice[it+1]]
#XX        for iccd, ccd in enumerate(lccds):
#XX            s = ccd.shape
#XX            # Spot guider darks and binned main camera darks.  Note
#XX            # Pythonic C index ordering
#XX            if s != (NAXIS2, NAXIS1):
#XX                log.debug('dark wrong shape: ' + str(s))
#XX                continue
#XX            ccd = ccd_process(ccd, oscan=True, master_bias=master_bias,
#XX                              gain=True, error=True)
#XX            hdr = ccd.header
#XX            im = ccd.data
#XX            # Spot darks recorded when too bright.
#XX            m = np.asarray(s)/2 # Middle of CCD
#XX            q = np.asarray(s)/4 # 1/4 point
#XX            m = m.astype(int)
#XX            q = q.astype(int)
#XX            # --> check lowest y too, since large filters go all the
#XX            # --> way to the edge See 20200428 dawn biases
#XX            dark_patch = im[m[0]-50:m[0]+50, 0:100]
#XX            light_patch = im[m[0]-50:m[0]+50, q[1]:q[1]+100]
#XX            mdp = np.median(dark_patch)
#XX            mlp = np.median(light_patch)
#XX            # --> might need to adjust this tolerance
#XX            if (np.median(light_patch) - np.median(dark_patch) > 1):
#XX                log.debug('dark recorded during light conditions: ' +
#XX                          flist[iccd])
#XX                log.debug('light, dark patch medians ({:.4f}, {:.4f})'.format(mdp, mlp))
#XX                continue
#XX            dark_ccds.append(ccd)
#XX        lccds = dark_ccds
#XX        mem = psutil.virtual_memory()
#XX        combined_average = \
#XX            ccdp.combine(lccds,
#XX                         method='average',
#XX                         sigma_clip=True,
#XX                         sigma_clip_low_thresh=5,
#XX                         sigma_clip_high_thresh=5,
#XX                         sigma_clip_func=np.ma.median,
#XX                         sigma_clip_dev_func=mad_std,
#XX                         mem_limit=mem.available*0.6)
#XX        im = combined_average
#XX        std =  np.std(im)
#XX        med =  np.median(im)
#XX        mean = np.mean(im)
#XX        tmin = np.min(im)
#XX        tmax = np.max(im)
#XX        rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
#XX        print('combined dark statistics')
#XX        print('std, rdnoise, mean, med, min, max')
#XX        print(std, rdnoise, mean, med, tmin, tmax)
#XX        # Record each bias filename
#XX        for i, f in enumerate(flist):
#XX            im.meta['FILE{0:02}'.format(i)] = f
#XX        # Prepare to write
#XX        fname = fbase + '_combined_bias.fits'
#XX        add_history(im.meta,
#XX                    'Combining NCOMBINE biases indicated in FILENN')
#XX        # Leave these large for fast calculations downstream and make
#XX        # final results that primarily sit on disk in bulk small
#XX        #im.data = im.data.astype('float32')
#XX        #im.uncertainty.array = im.uncertainty.array.astype('float32')
#XX        im.write(fname, overwrite=True)
#XX        if show:
#XX            impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray,
#XX                              filternorm=0, interpolation='none',
#XX                              vmin=med-std, vmax=med+std)
#XX            plt.show()
#XX            plt.close()




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

# def dark_combine(dark_fnames, overscan_fname=None, show=False):
#     if overscan_fname is not None:
#         o = CCDData.read(overscan_fname, unit=u.adu)
#         overscan = np.median(o)
#     else:
#         overscan = 0
#     # --> note scope violation of directory
#     dark_fnames = [os.path.join(directory, f) for f in dark_fnames]
#     satlevel = (np.uint16(-1) - overscan) * global_gain
#     print('bias_level from bias immediately after dark sequence = ', overscan)
#     print('saturation level (electrons) ', satlevel)
#     lccds = [bias_subtract(d, bias_fname, '/tmp/test.fits', overscan=overscan)
#          for d in dark_fnames]
#     mem = psutil.virtual_memory()
#     combined_average = \
#         ccdp.combine(lccds,
#                         method='average',
#                         sigma_clip=True,
#                         sigma_clip_low_thresh=5,
#                         sigma_clip_high_thresh=5,
#                         sigma_clip_func=np.ma.median,
#                         sigma_clip_dev_func=mad_std,
#                         mem_limit=mem.available*0.6)
#     im = combined_average
#     std =  np.std(im)
#     med =  np.median(im)
#     mean = np.mean(im)
#     tmin = np.min(im)
#     tmax = np.max(im)
#     rdnoise = np.sqrt(np.median((im.data[1:] - im.data[0:-1])**2))
#     print('combined dark statistics')
#     print('std, rdnoise, mean, med, min, max')
#     print(std, rdnoise, mean, med, tmin, tmax)
#     if show:
#         impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#                           interpolation='none', vmin=med-std, vmax=med+std)
#         plt.show()
#         plt.close()
#     return im

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
    impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
                      interpolation='none', vmin=med-std, vmax=med+std)
    if isinstance(im, CCDData):
        plt.title(im.unit)
    else:
        plt.title('units not specified')    
    plt.show()
    plt.close()
    

log.setLevel('DEBUG')

bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
bias = ccddata_read(bias_fname)
dark_dir = '/data/io/IoIO/raw/20200711'
dark_combine(dark_dir, master_bias=bias, show=True)

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

#print(fname_by_imagetyp_t_exp('/data/io/IoIO/raw/20200711',
#                              subdirs=['Calibration'],
#                              glob_include=['Bias*', '*_bias.fit'],
#                              imagetyp='BIAS'))
#print('Now on to DARKS')
#print(fname_by_imagetyp_t_exp('/data/io/IoIO/raw/20200711', imagetyp='DARK'))

#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=True)
#bias_combine('/data/io/IoIO/raw/20200711', show=True, gain_correct=False)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
#master_bias = ccddata_read(bias_fname)
#ccd = ccddata_read(fname)
#b = ccd_process(ccd, oscan=True, master_bias=master_bias,
#                gain=True, error=True)
#quick_show(b)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#
#fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
#b = subtract_overscan(fname, bias_fname)
#quick_show(b)

## bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
## fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
## print(overscan_estimate(fname, bias_fname, show=True))
## #(1935.667312196714, ('OVERSCAN_METHOD', 'corners'))
## 
## fname = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0010_Na_on-band.fit'
## print(overscan_estimate(fname, bias_fname, show=True))
## #(1949.6699784443222, ('OVERSCAN_METHOD', 'corners'))
## 
## fname = '/data/io/IoIO/raw/2020-07-11/NEOWISE-0005_Na-on.fit'
## print(overscan_estimate(fname, bias_fname, show=True))
## #(1934.3136989482311, ('OVERSCAN_METHOD', 'histogram'))

### fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
### print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias, show=True))
### # 1932.5354651515347
### 
### fname = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0010_Na_on-band.fit'
### print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias, show=True))
### # 1966.1345777424763
### # a broad peak, so higher than it needs to be




#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#b = ccddata_read(bias_fname)
##print(b.meta)
#fname = '/data/io/IoIO/raw/2020-07-11/NEOWISE-0005_Na-on.fit'
#f = ccddata_read(fname)
##print(f.meta)
#g = ccddata_read(f)
#print(g.meta)

#bias_fname = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits'
#bias = CCDData.read(bias_fname)
#fname = '/data/io/IoIO/raw/2020-07-11/NEOWISE-0005_Na-on.fit'
#f = CCDData.read(fname, unit=u.adu)
#f.meta['BUNIT'] = ('ADU', 'Unit of pixel value')
#print(overscan_estimate(f, show=True))
##print(overscan_estimate(f, readnoise='RDNOISE', gain='GAIN', bias=bias, show=True))
##print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias, show=True))
##print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias_fname, show=True))
#print(f.meta)
#print(overscan_estimate(fname, binsize=100, bias=bias, show=True))
## 1930.2304235108968

# bias_adu = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias_ADU.fits'
# bias_electron = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias_electrons.fits'
# fname = '/data/io/IoIO/raw/20200708/Na_on-band_004.fits'
# 
# gain_obj = ccdp.Keyword('GAIN', u.electron/u.adu)
# readnoise_obj = ccdp.Keyword('RDNOISE', u.electron)
# 
# #im = ccdp.gain_correct(im, g)
# raw_ccd = ccddata_read(fname)
# bias_adu = ccddata_read(bias_adu)
# 
# bias_electron = ccddata_read(bias_electron)
# 
# ccd_bs = ccdp.subtract_bias(raw_ccd, bias_adu)
# ccd_bsos = subtract_overscan(ccd_bs, bias=bias_adu, readnoise='RDNOISE', gain='GAIN')
# #ccd_bsos = create_deviation(ccd_bsos, gain=global_gain*u.electron/u.adu,
# #                            readnoise=global_readnoise)
# 
# 
# ccd_os = subtract_overscan(raw_ccd, bias=bias_electron, readnoise='RDNOISE', gain='GAIN')
# ccd_osbs = ccdp.subtract_bias(ccd_os, bias_adu)
# 
# im = ccd_osbs.subtract(ccd_bsos)
# std =  np.std(im)
# med =  np.median(im)
# #mean = np.asscalar(np.mean(im).data  )
# mean = np.mean(im)
# tmin = np.min(im)
# tmax = np.max(im)
# impl = plt.imshow(im, origin='lower', cmap=plt.cm.gray, filternorm=0,
#                   interpolation='none', vmin=med-std, vmax=med+std)
# print('std, mean, med, min, max (ADU)')
# print(std, mean, med, tmin, tmax)
# 
# plt.show()
# plt.close()


#print(overscan_estimate(fname, readnoise='RDNOISE', gain='GAIN', bias=bias, show=True))
#bias = '/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias_electrons.fits'
#print(overscan_estimate(fname, readnoise='RDNOISE', gain='GAIN', bias=bias, show=True))
## 1923.0292722359077
### 
### fname = '/data/io/IoIO/raw/20200708/SII_on-band_004.fits'
### print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias, show=True))
### # 1932.5354651515347
### 
### fname = '/data/io/IoIO/raw/2020-07-07/Sky_Flat-0010_Na_on-band.fit'
### print(overscan_estimate(fname, readnoise='RDNOISE', bias=bias, show=True))
### # 1966.1345777424763
### # a broad peak, so higher than it needs to be

#bias_combine('/data/io/IoIO/raw/20200711', show=True)
#im = ccddata_read('/data/io/IoIO/reduced/bias_dark/2020-07-11_ccdT_-5.27_combined_bias.fits')
#
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


#collection = ccdp.ImageFileCollection(directory,
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


