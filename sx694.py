"""Global variables and functions for processing SX694 images

Issues unique to MaxIm control of SX products (e.g., exposure time
correction) and corecting FITS keywords written by ACP are also
handled here

These variables and functions describe a Starlight Xpress Trius SX694
CCD purchased in 2017 for the IoIO project.  The later "Pro" version,
which has a different gain but otherwise similar characteristics.

Note: this is where to put things unique to the *camera*.  Things
unique to an instrument (e.g. coronagraph) would go somewhere else.


This module is intended to be imported like this:

import sx694

and the variables and functions referred to as sx694.naxis1, etc.

"""

from astropy import log
from astropy import units as u
from astropy.time import Time

# --> These imports will all go away
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
satlevel_comment = 'Saturation level in BUNIT'

# Gain measured in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx.
# Value agrees well with Trius SX-694 advertised value (note, newer
# "PRO" model has a different gain value).  Stored in GAIN keyword
gain = 0.3
gain_comment = 'Measured gain (electron/adu)'

# Sample readnoise measured as per ioio.notebk
# Tue Jul 10 12:13:33 2018 MCT jpmorgen@byted 

# Readnoies is measured regularly as part of master bias creation and
# stored in the RDNOISE keyword.  This is used as a sanity check.
example_readnoise = 15.475665 * gain
example_readnoise_comment = '2018-07-10 readnoise (electron)'
readnoise_tolerance = 0.5 # Units of electrons

# Measurement in /data/io/IoIO/observing/Exposure_Time_Calcs.xlsx of
# when camera becomes non-linear (in adu).  Stored in NONLIN keyword.
# Raw value of 42k was recorded with a typical overscan value.  Helps
# to remember ~40k is absolute max raw adu to shoot for.  This is
# suspiciously close to the full-well depth in electrons of 17,000
# (web) - 18,000 (user's manual) provided by the manufacturer -->
# could do a better job of measuring the precise high end of this,
# since it could be as high as 50k.  Include bias, since that is how
# we will be working with it.
nonlin = 42000 #- 1811
nonlin_comment = 'Measured nonlinearity point in BUNIT'

# The SX694 and similar interline transfer CCDs have such low dark
# current that it is not really practical to map out the dark current
# in every pixel (a huge number of darks would be needed to beat down
# the readnoise).  Rather, cut everything below this threshold, which
# reads in units of readnoise (e.g., value in electrons is
# 3 * example_readnoise).  The remaining few pixels generate
# more dark current, so include only them in the dark images
dark_mask_threshold = 3

# See exp_correct_value.  According to email from Terry Platt Tue, 7
# Nov 2017 14:04:24 -0700
# This is unique to the SX Universal CCD drivers in MaxIm 
max_accurate_exposure = 0.7 # s

# Fri Jun 03 11:58:12 2022 EDT  jpmorgen@snipe
# Old version of Photometry gave different answers probably because of
# low threshold
#latency_change_dates = ['2020-06-01', '2020-11-01']
latency_change_dates = ['2020-11-01']

# Amplitude of typical wander of Meinburg loopstats thought the day.
# Primarily due to lack of temperature control of crystal oscillator
# on PC motherboard used for time keeping.  Convert to RMS, since that
# better reflects the true distribution of values asssume a standard
# deviation-like distribution
ntp_accuracy = 0.005 / np.sqrt(2) # s

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
    hdr.comments['exptime'] = 'Exposure time (second)'
    if hdr.get('instrume'):
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
    # MaxIm records [XY]PIXSZ if available from camera and FOCALLEN if
    # (correctly) provided by user.  Note [XY]PIXSZ are corrected for
    # binning, so we don't need to do that here
    xpixsz = hdr.get('XPIXSZ')
    ypixsz = hdr.get('YPIXSZ')
    focal_length = hdr.get('FOCALLEN')
    if xpixsz and ypixsz and focal_length:
        # Be completely general -- not all CCDs have square pixels
        pixel_size = np.average((xpixsz, xpixsz))
        # Proper FITS header units would make these obsolete
        pixel_size *= u.micron 
        focal_length *= u.mm
        plate_scale = pixel_size / focal_length
        plate_scale *= u.radian
        plate_scale = plate_scale.to(u.arcsec)
        # Proper FITS header units would make this easier
        hdr['PIXSCALE'] = (plate_scale.value, '[arcsec] approximate binned pixel scale')    
    return hdr

def approx_pix_solid_angle(ccd):
    """Returns pixel solid angle using basic telescope properties.
    Refinement requires photometric solution and proj_plane_pixel_scales

    """
    # After binning pixel size 
    pixsz = np.asarray((ccd.meta['XPIXSZ'], ccd.meta['YPIXSZ']))
    pix_area = np.prod(pixsz)
    pix_area *= u.micron**2
    focal_length = ccd.meta['FOCALLEN']*u.mm
    pix_solid_angle = pix_area / focal_length**2
    pix_solid_angle *= u.rad**2
    pix_solid_angle = pix_solid_angle.to(u.arcsec**2)
    return pix_solid_angle

def exp_correct_value(date_obs):
    """Provides the measured extra exposure correction time for
    exposures > max_accurate_exposure.  See detailed discussion in
    IoIO_reduction.notebk on 
    Fri Jun 03 11:56:20 2022 EDT  jpmorgen@snipe
    KEEP THE TABLE IN THIS CODE UP TO DATE
"""

    if date_obs < latency_change_dates[0]:
        exposure_correct = 2.19 # s
        exposure_correct_uncertainty = 0.31 # s 
    else:
        exposure_correct = 2.71 # s
        exposure_correct_uncertainty = 0.47 # s
    #Sat May 15 22:42:42 2021 EDT  jpmorgen@snipe
    # Old photometry code
    #if date_obs < latency_change_dates[0]:
    #    exposure_correct = 2.10 # s
    #    exposure_correct_uncertainty = 0.33 # s 
    #elif date_obs < latency_change_dates[1]:
    #    exposure_correct = 1.87 # s
    #    exposure_correct_uncertainty = 0.18 # s 
    #else:
    #    exposure_correct = 2.40 # s
    #    exposure_correct_uncertainty = 0.18 # s

    # Soften exposure_correct_uncertainty a bit, since the MAD ended
    # up giving the true minimum latency value because of all the
    # outliers.  We want more like a 1-sigma.  Distribution is
    # actually more lumped toward low end with a tail toward high values
    exposure_correct_uncertainty /= np.sqrt(2)
    return (exposure_correct, exposure_correct_uncertainty)

def exp_correct(hdr_in,
                *args, **kwargs):
    """Correct exposure time problems for Starlight Xpress drivers in
    MaxIm.  

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
                'commanded exposure time (s)'),
               after=True)
    exposure_correct, exposure_correct_uncertainty = \
        exp_correct_value(hdr['DATE-OBS'])
    exptime += exposure_correct
    hdr['EXPTIME'] = (exptime,
                      'corrected exposure time (s)')
    hdr.insert('OEXPTIME', 
               ('HIERARCH EXPTIME-UNCERTAINTY',
                exposure_correct_uncertainty, 'Measured RMS (s)'),
               after=True)
    hdr.insert('OEXPTIME', 
               ('HIERARCH EXPTIME-CORRECTION',
                exposure_correct, 'Measured (s)'),
               after=True)
    #add_history(hdr,
    #            'Corrected exposure time for SX694 MaxIm driver bug')
    return hdr

def date_obs(hdr_in,
             *args, **kargs):
    """Correct DATE-OBS keyword in FITS header for shutter latency
    See discussion in IoIO.notebk 
    Mon May 17 13:30:45 2021 EDT  jpmorgen@snipe
    and IoIO_reduction.notebk 
    Sun Mar 27 21:43:17 2022 EDT  jpmorgen@snipe"""
    hdr = hdr_in.copy()
    date_obs_str = hdr['DATE-OBS']
    date_obs = Time(date_obs_str, format='fits')
    exposure_correct, exposure_correct_uncertainty = \
        exp_correct_value(date_obs_str)
    # Best estimate of shutter latency is 1/2 the round-trip inferred
    # from exposure correction value
    shutter_latency = exposure_correct / 2 * u.s
    shutter_latency_uncertainty = exposure_correct_uncertainty / 2
    date_beg = date_obs + shutter_latency
    # Calculate date-obs precision based on how the string is written.
    # Technically, we should not record subsequent values to higher
    # precision than we receive, but doing that bookkeeping is a pain.
    # Instead, use a standard deviation-like approach to recording the
    # resulting uncertainty.  The reason that is not done in general
    # is that the precision really does give you some information
    # about the distribution of the true value being recorded: it has
    # equal likelyhood of being anywhere between +/- 0.5 of the last
    # digit recorded (e.g. value of 2 could be anywhere between 1.5
    # and 2.5).  This is a square probability distribution, rather
    # than a Gaussian, so stdev is not quite the right astropy
    # uncertainty type to use.  However, precision is usually small,
    # so it doesn't matter compared to other errors when added in
    # quadrature.  In this case, it can be as large as 0.5s, which
    # dominates.  But it is added in quadrature with the
    # shutter_laterncy_uncertainty (~0.1s), so there is some genuine
    # fuzz due to that.  The net result is something like a
    # broad-shouldered Gaussian of width 0.5.  Call that good enough.
    if '.' in date_obs_str:
        _, dec_str = date_obs_str.split('.')
        places = sum(c.isdigit() for c in dec_str)
        precision = 10**-places/2
    else:
        precision = 0.5
    date_beg_uncertainty = np.sqrt(precision**2
                                   + shutter_latency_uncertainty**2
                                   + ntp_accuracy**2)

    # DATE-AVG we need to have EXPTIME
    oexptime = hdr.get('OEXPTIME')
    exptime = hdr['EXPTIME']
    if exptime > max_accurate_exposure and oexptime is None:
        log.warning('EXPTIME not corrected for SX MaxIm driver issue.  This should normally have been done before this point')
        exptime += exposure_correct
    # A fancy CCD system can possibly have the shutter close and
    # reopen or even have multiple aperture settings at some f/stop so
    # that average in the time dimension is something other than the
    # simple midpoint.  Here all we have is open and close, so the
    # midpoint is the right thing.
    date_avg = date_beg + (exptime / 2) * u.s

    if exptime > max_accurate_exposure:
        # Note that this has shutter_latency_uncertainty show up twice
        # and then is combined in date_avg_uncertainty as if it is a
        # true independent quantity.  This is actually a reasonable
        # approach, since the shutter latency is one packet trip and
        # the exposure correction is cause by another two
        exptime_uncertainty = exposure_correct_uncertainty
    else:
        # Using the internal CCD timer, the precision with which
        # EXPTIME is recorded should be far in excess of any other
        # uncertainties
        exptime_uncertainty = 0
    date_avg_uncertainty = np.sqrt(date_beg_uncertainty**2
                                   + (exptime_uncertainty/2)**2)
    date_obs_str = hdr['DATE-OBS']
    hdr.insert('DATE-OBS',
               ('ODAT-OBS', date_obs_str,
               'Commanded shutter time (ISO 8601 UTC)'))
    # astropy.nddata.CCDData.read assumes DATE-OBS is the shutter open
    # time.  Calculate DATE-AVG here for reading with pgdata.tavg
    # method to avoid double-calculation on first read.  That method
    # can deal with its absence.
    hdr['DATE-OBS'] = (date_beg.fits,
                       'Best estimate shutter open time (UTC)')
    # These end up appearing in reverse order to this
    hdr.insert('DATE-OBS',
               ('HIERARCH NTP-ACCURACY', ntp_accuracy,
                'RMS typical (s)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('HIERARCH SHUTTER-LATENCY-UNCERTAINTY',
                shutter_latency_uncertainty,
                '(s)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('HIERARCH SHUTTER-LATENCY', shutter_latency.value,
                'Measured indirectly (s)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('HIERARCH DATE-AVG-UNCERTAINTY', date_avg_uncertainty,
                '(s)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('HIERARCH DATE-OBS-UNCERTAINTY', date_beg_uncertainty,
                '(s)'),
               after=True)
    # Avoid annoying WCS warnings
    hdr.insert('DATE-OBS',
               ('MJD-AVG', date_avg.mjd,
                'Best estimate midpoint of exposure (MJD)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('MJD-OBS', date_beg.mjd,
                'Best estimate shutter open time (MJD)'),
               after=True)
    hdr.insert('DATE-OBS',
               ('DATE-AVG', date_avg.fits,
                'Best estimate midpoint of exposure (UTC)'),
               after=True)
    # Remove now inaccurate and obsolete keywords
    if hdr.get('date'):
        del hdr['DATE']
    if hdr.get('TIME-OBS'):
        del hdr['TIME-OBS']
    if hdr.get('UT'):
        del hdr['UT']
    return hdr

############################################################################
# --> These will all go away when IoIO.corobsdata system is decommissioned #
############################################################################

# --> this is really more of a generic utility.  Also, this could be
# incorporated with ObsData if the ObsData image + header property
# were changed from an HDUList to a CCDData.data and .meta.  Or better
# yet, if there is not significant speed impact, PrecisionGuide could
# itself be a subclass of CCDData, simply adding property and methods
# for the precisionguide stuff.
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
    """Estimate overscan in adu in the absense of a formal overscan region

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
        of adu or electrons and is converted using the specified gain.
        If bias has already been subtracted, this step will be skipped
        but the bias header will be used to extract readnoise and gain
        using the *_key keywords.  Default is ``None``.

    binsize: float or None, optional
        The binsize to use for the histogram.  If None, binsize is 
        (readnoise in adu)/4.  Default = None

    min_width : int, optional
        Minimum width peak to search for in histogram.  Keep in mind
        histogram bins are binsize adu wide.  Default = 1

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
    if hdr_in.get('subtract_overscan') is not None:
        # We have been here before
        return 0
    # We mess with both the input image and hdr, so save local copies
    im = im_in.copy()
    hdr = metadata(hdr_in)
    if hdr_out is None:
        hdr_out = hdr
    bunit = hdr.get('bunit')
    if bunit is not None and bunit != 'adu':
        # For now don't get fancy with unit conversion
        raise ValueError(f'CCD units are {bunit} but must be in adu for overscan estimation')
    if hdr['IMAGETYP'] == "BIAS":
        overscan = np.median(im)
        hdr_out['HIERARCH OVERSCAN_MEDIAN'] = (overscan, 'adu')
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
            # Convert bias back to adu for subtraction
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
        # Calculate binsize based on readnoise in adu, but oversample
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
    hdr_out['HIERARCH OVERSCAN_CORNERS'] = (corners_method, 'adu')
    hdr_out['HIERARCH OVERSCAN_HISTOGRAM'] = (hist_method, 'adu')
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
        ax1.set_title('Image minus 1000 adu')
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
