"""Global variables and functions for processing SX694 data

Note: these describe a Starlight Xpress Trius SX694 CCD purchased in
2017 for the IoIO project.  The later "Pro" version, which has a
different gain but otherwise similar characteristics.

This module is intended to be imported like this:

import sx694

and the variables and functions referred to as sx694.naxis1, etc.

"""

import numpy as np
from scipy import signal

from astropy.io import fits
from astropy.stats import biweight_location

# Record in global variables Starlight Xpress Trius SX694 CCD
# characteristics.  Note that CCD was purchased in 2017 and is NOT the
# "Pro" version, which has a different gain but otherwise similar
# characteristics

camera_description = 'Starlight Xpress Trius SX694 mono, 2017 model version'

# naxis1 = fastest changing axis in FITS primary image = X in
# Cartesian thought
# naxis1 = next to fastest changing axis in FITS primary image = Y in
# Cartesian thought
naxis1 = 2750
naxis2 = 2200

# 16-bit A/D converter, stored in SATLEVEL keyword
satlevel = 2**16-1
satlevel_comment = 'Saturation level (ADU)'

# Gain measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx.
# Value agrees well with Trius SX-694 advertised value (note, newer
# "PRO" model has a different gain value).  Stored in GAIN keyword
gain = 0.3
gain_comment = 'Measured gain (electron/ADU)'

# Sample readnoise measured as per ioio.notebk
# Tue Jul 10 12:13:33 2018 MCT jpmorgen@byted 

# Readnoies is measured regularly as part of master bias creation and
# stored in the RDNOISE keyword.  This is used as a sanity check.
example_readnoise = 15.475665 * gain
example_readnoise_comment = '2018-07-10 readnoise (electron)'
readnoise_tolerance = 0.5 # Units of electrons

# Measurement in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx of
# when camera becomes non-linear.  Stored in NONLIN keyword.  Raw
# value of 42k was recorded with a typical overscan value.  Helps to
# remember ~40k is absolute max raw ADU to shoot for.  This is
# suspiciously close to the full-well depth in electrons of 17,000
# (web) - 18,000 (user's manual) provided by the manufacturer
# --> could do a better job of measuring the precise high end of this,
# since it could be as high as 50k
nonlin = 42000 - 1811
nonlin_comment = 'Measured nonlinearity point (ADU)'

# Exposure times at or below this value are counted on the camera and
# not in MaxIm.  There is a bug in the SX694 MaxIm driver seems to
# consistently add about exposure_correct seconds to the
# exposure time before asking the camera to read the CCD out.
# Measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx
# --> NEEDS TO BE VERIFIED WITH PHOTOMETRY FROM 2019 and 2020
# Corrected as part of local version of ccd_process
max_accurate_exposure = 0.7 # s
exposure_correct = 1.7 # s

# The SX694 and similar interline transfer CCDs have such low dark
# current that it is not really practical to map out the dark current
# in every pixel (a huge number of darks would be needed to beat down
# the readnoise).  Rather, cut everything below this threshold, which
# reads in units of readnoise (e.g., value in electrons is
# 3 * example_readnoise).  The remaining few pixels generate
# more dark current, so include only them in the dark images
dark_mask_threshold = 3

def read_hdu0(fname):
    """Read primary FITS header and image

    Parameters
    ----------
    fname : str or `~astropy.nddata.CCDData`
        FITS filename or already read `~astropy.nddata.CCDData`

    Returns
    -------
    (hdr, im) : tuple
        hdr = FITS header, im = primary FITS array
        
    """
    if isinstance(fname, str):
        with fits.open(fname) as HDUList:
            hdr = HDUList[0].header
            im = HDUList[0].data
            return (hdr, im)
    # Assume it is a CCDData, but don't check, otherwise we would
    # have to import ccdproc, which could slow data acquisition
    # processes unnecessarily
    ccd = fname
    hdr = ccd.meta
    im = ccd.data
    # Fix problem where BUNIT is not recorded until file is
    # written
    hdr['BUNIT'] = ccd.unit.to_string()
    return (hdr, im)

def metadata(hdr_in,
                 camera_description=camera_description,
                 gain=gain,
                 gain_comment=gain_comment,
                 satlevel=satlevel,
                 satlevel_comment=satlevel_comment,
                 nonlin=nonlin,
                 nonlin_comment=nonlin_comment,
                 readnoise=example_readnoise,
                 readnoise_comment=example_readnoise_comment,
                 *args, **kwargs):
    """Record SX694 CCD metadata in FITS header object"""
    if hdr_in.get('camera') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    hdr = hdr_in.copy()
    # Clean up double exposure time reference to avoid confusion
    if hdr.get('exposure') is not None:
        del hdr['EXPOSURE']
    hdr.insert('INSTRUME',
                    ('CAMERA', camera_description),
                    after=True)
    hdr['GAIN'] = (gain, gain_comment)
    # This gets used in ccdp.cosmicray_lacosmic
    hdr['SATLEVEL'] = (satlevel, satlevel_comment)
    # This is where the CCD starts to become non-linear and is
    # used for things like rejecting flats recorded when
    # conditions were too bright
    hdr['NONLIN'] = (nonlin, nonlin_comment)
    hdr['RDNOISE'] = (readnoise, readnoise_comment)
    return hdr

def exp_correct(hdr_in,
                    max_accurate_exposure=max_accurate_exposure,
                    exposure_correct=exposure_correct,
                    *args, **kwargs):
    """Correct exposure time for [SX694] CCD driver problem
     --> REFINE THIS ESTIMATE BASED ON MORE MEASUREMENTS
    """
    if hdr_in.get('OEXPTIME') is not None:
        # We have been here before, so exit quietly
        return hdr_in
    exptime = hdr_in['EXPTIME']
    if exptime <= max_accurate_exposure:
        # Exposure time should be accurate
        return hdr_in
    hdr = hdr_in.copy()
    hdr.insert('EXPTIME', 
               ('OEXPTIME', exptime,
                'original exposure time (seconds)'),
               after=True)
    exptime += exposure_correct
    hdr['EXPTIME'] = (exptime,
                      'corrected exposure time (seconds)')
    hdr.insert('OEXPTIME', 
               ('HIERARCH EXPTIME_CORRECTION',
                exposure_correct, '(seconds)'),
               after=True)
    #add_history(hdr,
    #            'Corrected exposure time for SX694 MaxIm driver bug')
    return hdr


# --> this is really more of a generic utility
def hist_of_im(im, binsize=1, show=False):
    """Returns a tuple of the histogram of image and index into *centers* of
bins."""
    # Code from west_aux.py, maskgen.
    # Histogram bin size should be related to readnoise
    hrange = (im.min(), im.max())
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

def overscan_estimate(im_in, hdr_in, hdr_out=None, master_bias=None,
                      binsize=None, min_width=1, max_width=8, box_size=100,
                      min_hist_val=10,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    For biases, returns in the median of the image.  For all others,
    uses the minimum of: (1) the first peak in the histogram of the
    image or (2) the minimum of the median of four boxes at the
    corners of the image.

    Works best if bias shape (particularly bias ramp) is subtracted
    first.  Will subtract bias if bias is supplied and has not been
    subtracted.

    Parameters
    ----------
    im_in : `~numpy.ndarray`
        Image from which to extract overscan estimate

    hdr_in : `astropy.io.fits.header` or None
        FITS header of input image

    hdr_out : `astropy.io.fits.header` or None
        Referece to metadata of ccd into which to write OVERSCAN_* cards.
        If None, no metadata will be returned

    master_bias : `~astropy.nddata.CCDData`, filename, or None
        Bias to subtract from ccd before estimate is calculated.
        Improves accruacy by removing bias ramp.  Bias can be in units
        of ADU or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

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
    # We mess with both the input image and hdr, so save local copies
    im = im_in.copy()
    hdr = metadata(hdr_in)
    if hdr_out is None:
        hdr_out = hdr
    bunit = hdr.get('bunit')
    if bunit is not None and bunit != 'adu':
        # For now don't get fancy with unit conversion
        raise ValueError(f'CCD units are {bunit} bur must be in ADU for overscan estimation')
    if hdr['IMAGETYP'] == "BIAS":
        overscan = np.median(ccd)
        hdr_out['HIERARCH OVERSCAN_MEDIAN'] = (overscan, 'ADU')
        hdr_out['HIERARCH OVERSCAN_METHOD'] \
            = ('median', 'Method used for overscan estimation')
        return overscan

    # Prepare for histogram method of overscan estimation.  These
    # keywords are guaranteed to be in meta because we put there there
    # in metadata
    readnoise = hdr['RDNOISE']
    gain = hdr['GAIN']
    if hdr.get('subtract_bias') is None and master_bias is not None:
        # Bias has not been subtracted and we have a bias around to be
        # able to do that subtraction
        bhdr, bias = read_hdu0(master_bias)
        # Improve our readnoise (measured) and gain (probably not
        # re-measured) values
        readnoise = bhdr['RDNOISE']
        gain = bhdr['GAIN']
        bbunit = bhdr['BUNIT']
        if bbunit == "electron":
            # Convert bias back to ADU for subtraction
            bias = bias / gain
        im = im - bias
        if isinstance(master_bias, str):
            hdr_out['HIERARCH OVERSCAN_MASTER_BIAS'] = 'OSBIAS'
            hdr_out['OSBIAS'] = master_bias
        else:
            hdr_out['HIERARCH OVERSCAN_MASTER_BIAS'] = 'CCDData object provided'
    #if hdr.get('subtract_bias') is None:
    #    log.warning('overscan_estimate: bias has not been subtracted, which can lead to inaccuracy of overscan estimate')
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.
    # Corners method
    s = im.shape
    bs = box_size
    c00 = biweight_location(im[0:bs,0:bs])
    c10 = biweight_location(im[s[0]-bs:s[0],0:bs])
    c01 = biweight_location(im[0:bs,s[1]-bs:s[1]])
    c11 = biweight_location(im[s[0]-bs:s[0],s[1]-bs:s[1]])
    #c00 = np.median(im[0:bs,0:bs])
    #c10 = np.median(im[s[0]-bs:s[0],0:bs])
    #c01 = np.median(im[0:bs,s[1]-bs:s[1]])
    #c11 = np.median(im[s[0]-bs:s[0],s[1]-bs:s[1]])
    corners_method = min(c00, c10, c01, c11)
    # Histogram method.  The first peak is the bias, the second is the
    # ND filter.  Note that the 1.25" filters do a better job at this
    # than the 2" filters but with carefully chosen parameters, the
    # first small peak can be spotted.
    if binsize is None:
        # Calculate binsize based on readnoise in ADU, but oversample
        # by 4.  Note need to convert from Quantity to float
        binsize = readnoise/gain/4.
    im_hist, im_hist_centers = hist_of_im(im, binsize)
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
    hdr_out['HIERARCH OVERSCAN_CORNERS'] = (corners_method, 'ADU')
    hdr_out['HIERARCH OVERSCAN_HISTOGRAM'] = (hist_method, 'ADU')
    o_idx = np.argmin(overscan_values)
    overscan = overscan_values[o_idx]
    hdr_out['HIERARCH OVERSCAN_METHOD'] = (overscan_methods[o_idx],
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
