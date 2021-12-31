"""Module supporting data and calculations for real-time coronagraph
observations.  More complicated data containers and calculations
appropriate for reduction go in a separate module

"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

#from ginga import toolkit
#from ginga.qtw.QtHelp import QtGui, QtCore
#from ginga.gw import Viewers
#from ginga.misc import log
#
#toolkit.use('qt5')
#logger = log.get_logger("viewer1", log_stderr=True, level=40)
#app = QtGui.QApplication([])
#w = QtGui.QMainWindow(logger)
#w.show()
#app.setActiveWindow(w)
#w.raise_()
#w.activateWindow()
#
##v = Viewers.CanvasView(logger=logger)


from astropy import log
from astropy import units as u
from astropy.stats import biweight_location

from astropy_fits_key import FitsKeyArithmeticMixin

from ccdmultipipe.utils import FilterWarningCCDData

from precisionguide import pgproperty, pgcoordproperty
from precisionguide import MaxImPGD, NoCenterPGD, CenterOfMassPGD
from precisionguide.utils import hist_of_im, iter_linfit

import sx694

# Ideally I make this a function that returns the proper
# default_ND_params as a function of date.  However, there is no point
# in doing this, since the only time this is used directly is when new
# observations are taken.  Old observations are always processed
# through the corprocess system, --> which uses appropriately timed
# flats (have to tweak to ensure that) to get the ND_params.

# 2018 end of run
#RUN_LEVEL_DEFAULT_ND_PARAMS \
#    = [[  3.63686271e-01,   3.68675375e-01],
#       [  1.28303305e+03,   1.39479846e+03]]

## Fri Feb 08 19:35:56 2019 EST  jpmorgen@snipe
#RUN_LEVEL_DEFAULT_ND_PARAMS \
#    = [[  4.65269008e-03,   8.76050569e-03],
#       [  1.27189987e+03,   1.37717911e+03]]

## Fri Apr 12 15:42:30 2019 EDT  jpmorgen@snipe
#RUN_LEVEL_DEFAULT_ND_PARAMS \
#    = [[1.40749551e-02, 2.36320869e-02],
#       [1.24240593e+03, 1.33789081e+03]]

# Sun Apr 25 00:48:25 2021 EDT  jpmorgen@snipe
RUN_LEVEL_DEFAULT_ND_PARAMS \
    = [[-3.32901729e-01, -3.21280155e-01],
       [ 1.26037690e+03,  1.38195602e+03]]


# Unbinned coords --> Note, what with the poor filter wheel centering
# after fall 2020, this might need to change into a function that
# floats around, though for now it is just used to speed finding hot
# pixels and so is OK
SMALL_FILT_CROP = np.asarray(((350, 550), (1900, 2100)))

# Experiments with medfilt show:
# /data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S7.fit and
# friends for how high the peak in a 1D profile gets above the ND median
# Jupiter = 1700
# Vega = 600
# Mercury = 70 on 90 off
# Galsat = 38
# Noisey sky = 18
PROFILE_PEAK_THRESHOLD = 20

# Definitive Jupiter good conditions 
BRIGHT_SAT_THRESHOLD = 1000
# 25 worked for a star, 250 should be conservative for Jupiter
# when it is attenuated by clouds.  As noted below, only works
# for one bright source
MIN_SOURCE_THRESHOLD = 250



def overscan_estimate(ccd_in, meta=None, master_bias=None,
                      binsize=None, min_width=1, max_width=8, box_size=100,
                      min_hist_val=10,
                      show=False, *args, **kwargs):
    """Estimate overscan in ADU in the absense of a formal overscan region

    For biases, returns in the median of the image.  For all others,
    uses the minimum of: (1) the first peak in the histogram of the
    image or (2) the minimum of the median of four boxes at the
    corners of the image (specific to the IoIO coronagraph)

    Works best if bias shape (particularly bias ramp) is subtracted
    first.  Will subtract bias if bias is supplied and has not been
    subtracted.

    Parameters
    ----------
    ccd_in : `~astropy.nddata.CCDData` or filename
        Image from which to extract overscan estimate

    meta : `astropy.io.fits.header` or None
        referece to metadata of ccd into which to write OVERSCAN_* cards.
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
    if ccd_in.meta.get('overscan_value') is not None:
        # Overscan has been subtracted in a previous reduction step,
        # so exit quietly
        return 0

    # Work with a copy since we mess with both the ccd.data and .meta
    ccd = ccd_in.copy()
    # Get CCD characteristics
    ccd.meta = sx694.metadata(ccd.meta)
    if meta is None:
        meta = ccd.meta
    if ccd.unit != u.adu:
        # For now don't get fancy with unit conversion
        raise ValueError('CCD units must be in ADU for overscan estimation')
    if ccd.meta['IMAGETYP'] == "BIAS":
        overscan = np.median(ccd)
        meta['HIERARCH OVERSCAN_MEDIAN'] = (overscan, 'ADU')
        meta['HIERARCH OVERSCAN_METHOD'] = \
            ('median', 'Method used for overscan estimation') 
        return overscan

    # Prepare for histogram method of overscan estimation.  These
    # keywords are guaranteed to be in meta because we put there there
    # in ccd_metadata
    readnoise = ccd.meta['RDNOISE']
    gain = ccd.meta['GAIN']
    if ccd.meta.get('subtract_bias') is None and master_bias is not None:
        # Bias has not been subtracted and we have a bias around to be
        # able to do that subtraction
        if isinstance(master_bias, str):
            bias = CorData.read(master_bias)
            meta['HIERARCH OVERSCAN_MASTER_BIAS'] = 'OSBIAS'
            meta['OSBIAS'] = master_bias
        else:
            # Work with a copy since we are going to muck with it
            bias = master_bias.copy()
            meta['HIERARCH OVERSCAN_MASTER_BIAS'] = 'CCDData object provided'
        # Improve our readnoise (measured) and gain (probably not
        # re-measured) values
        readnoise = bias.meta['RDNOISE']
        gain = bias.meta['GAIN']
        if bias.unit is u.electron:
            # Convert bias back to ADU for subtraction
            bias = bias.divide(gain*u.electron/u.adu)
        ccd = ccd.subtract(bias)
        ccd.meta['HIERARCH subtract_bias'] = True
    if type(ccd) != CorData and ccd.meta.get('subtract_bias') is None:
        # Don't gunk up logs when we are taking data, but subclasses
        # of CorObs (e.g. RedCorObs) will produce message
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

def keyword_arithmetic_image_handler(meta, operand1, operation, operand2,
                                     keylist=None, stdev_threshold=10):
    """Convert an image to a scalar for FITS keyword arithmetic"""

    # This is hard to do in general just looking at the data in
    # operand2 except the special case where stdev data = 0 (a value
    # turned into an array).  Start in the cases we know the answer
    if hasattr(operand2, 'meta'):
        imagetyp = operand2.meta.get('imagetyp')
        if imagetyp is None:
            # Trigger general code below
            o2 = None
        else:
            imagetyp = imagetyp.lower() 
            if imagetyp in ['bias', 'dark']:
                # For the coronagraph reduction scheme, biases and
                # darks are small perterbations around zero.  Other
                # projects may incorporate offset (overscan) into the
                # bias
                o2 = 0
            elif imagetyp == 'flat':
                # Check for unprocessed flat
                if operand2.meta.get('flatdiv') is None:
                    # med is not the best value, since the flats roll
                    # off significantly, but this is the wrong place
                    # to calculate it correctly.  Generally, flats are
                    # divided by a scalar first, their max val, and
                    # then the FLATDIV keyword is written, so this
                    # code should not be called
                    o2 = np.median(operand2)
                    log.warning('Arithmetic with unprocessed flat, keyword arithmetic will be off unless the median of the flat is close to the desired characteristic value')
                else:
                    # Processed flats are normalized to 1
                    o2 = 1
            else:
                # Trigger general code below
                o2 = None
    else:
        # No metadata to help us
        o2 = None
        
    if o2 is None:
        # If we made it here, we need to use a general approach to
        # recognize structure.  This fails for biases and darks, hence
        # the need for the code above
        if hasattr(operand2, 'data'):
            # --> np.std is seeing the data and stdev for some reason
            im = operand2.data
        else:
            im = operand2
        med = np.median(im)
        stdev = np.std(im)
        if med == 0:
            o2 = 0
        elif stdev_threshold * stdev < np.abs(med):
            # The most common case here is likely to be stdev = 0, or
            # numbers that have been made into arrays to make use of
            # the NDData uncertainty propagation mixin code
            o2 = med
        else:
            o2 = None

    return o2

class CorData(FitsKeyArithmeticMixin, CenterOfMassPGD, NoCenterPGD, MaxImPGD):
    def __init__(self, *args,
                 default_ND_params=None,
                 y_center_offset=0, # *Unbinned* Was 70 for a while See desired_center
                 n_y_steps=8, # was 15 (see adjustment in flat code)
                 x_filt_width=25,
                 edge_mask=(5, -15), # Absolute coords, ragged on right edge.  If one value, assumed equal from *each* edge (e.g., (5, -5)
                 cwt_width_arange_flat=None, # flat ND_params edge find
                 cwt_width_arange=None, # normal ND_params edge find
                 cwt_min_snr=1, # Their default seems to work well
                 search_margin=50, # on either side of nominal ND filter
                 max_fit_delta_pix=25, # Thowing out point in 1 line fit
                 max_parallel_delta_pix=50, # Find 2 lines inconsistent
                 max_ND_width_range=(80,400), # jump-starting flats & sanity check others
                 small_filt_crop=SMALL_FILT_CROP,
                 plot_prof=False,
                 plot_dprof=False,
                 plot_ND_edges=False,
                 no_obj_center=False, # set to True to save calc, BIAS, DARK, FLAT always set to True
                 profile_peak_threshold=PROFILE_PEAK_THRESHOLD,
                 bright_sat_threshold=BRIGHT_SAT_THRESHOLD,
                 min_source_threshold=MIN_SOURCE_THRESHOLD,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.
        self.default_ND_params = default_ND_params
        self.y_center_offset        = y_center_offset
        if cwt_width_arange_flat is None:
            cwt_width_arange_flat   = np.arange(2, 60)
            self.cwt_width_arange_flat  = cwt_width_arange_flat
        if cwt_width_arange is None:
            cwt_width_arange        = np.arange(8, 80)
            self.cwt_width_arange       = cwt_width_arange       
            self.n_y_steps              = n_y_steps              
            self.x_filt_width           = x_filt_width
            self.edge_mask              = edge_mask
            self.cwt_min_snr            = cwt_min_snr            
            self.search_margin          = search_margin           
            self.max_fit_delta_pix      = max_fit_delta_pix      
            self.max_parallel_delta_pix = max_parallel_delta_pix
            self.max_ND_width_range	    = max_ND_width_range
            self.small_filt_crop        = small_filt_crop
            self.plot_prof		    = plot_prof 
            self.plot_dprof             = plot_dprof
            self.plot_ND_edges	    = plot_ND_edges

        self.no_obj_center          = no_obj_center
        self.profile_peak_threshold = profile_peak_threshold
        self.bright_sat_threshold = bright_sat_threshold
        self.min_source_threshold = min_source_threshold

        self.arithmetic_keylist = ['satlevel', 'nonlin']
        self.handle_image = keyword_arithmetic_image_handler

    @pgproperty
    def default_ND_params(self):
        # Define our array and default value all in one (a little
        # risky if FNPAR are partially broken in the FITS header)
        ND_params = np.asarray(RUN_LEVEL_DEFAULT_ND_PARAMS)
        # Code from flat correction part of cor_process to use the
        # master flat ND params as default
        for i in range(2):
            for j in range(2):
                fndpar = self.meta.get(f'fndpar{i}{j}')
                if fndpar is None:
                    break
                ND_params[j][i] = fndpar
        return ND_params

    @pgproperty
    def imagetyp(self):
        imagetyp = self.meta.get('imagetyp')
        if imagetyp is None:
            return None
        return imagetyp.lower()

    @pgcoordproperty
    def edge_mask(self):
        pass

    @edge_mask.setter
    def edge_mask(self, edge_mask):
        edge_mask = np.asarray(edge_mask)
        if edge_mask.size == 1:
            edge_mask = np.append(edge_mask, -edge_mask)
        return edge_mask        

    @pgproperty
    def ND_params(self):
        """Returns parameters which characterize the coronagraph ND filter.
        Parameters are relative to *unbinned image.* It is important
        this be calculated on a per-image bases, since flexure in the
        instrument can shift it a bit side-to-side.  Unforunately,
        contrast is not sufficient to accurately spot the ND_filter in
        normal dark sky exposures without some sort of close starting
        point.  This is addressed by using the flats, which do have
        sufficient contrast, to provide RUN_LEVEL_DEFAULT_ND_PARAMS.

        """

        if self.imagetyp == 'bias' or self.imagetyp == 'dark':
            return self.default_ND_params


        # To simplify calculations, everything will be done with the
        # unbinned image.  
        ytop = self.small_filt_crop[0,0]
        ybot = self.small_filt_crop[1,0]

        # For CorObs, there will not be mask, uncertainty, etc., so we
        # just work with the data, which we will call "im."

        if self.imagetyp == 'flat':
            # Flats have high contrast and low sensitivity to hot
            # pixels, so we can work with the whole image.  It is OK
            # that this is not a copy since we are not going to muck
            # with it.  Since flats are high enough quality, we use
            # them to independently measure the ND_params, so there is
            # no need for the default (in fact that is how we derive
            # it!).  Finally, The flats produce very narrow peaks in
            # the ND_param algorithm when processed without a
            # default_ND_param and there is a significant filter
            # rotation.  Once things are morphed by the
            # default_ND_params (assuming they match the image), the
            # peaks are much broader.  So our cwt arange needs to be a
            # little different.
            im = self.self_unbinned.data
            default_ND_params = None
            cwt_width_arange = self.cwt_width_arange_flat
            self.n_y_steps = 25
        else:
            # Non-flat case
            default_ND_params = self.default_ND_params
            cwt_width_arange = self.cwt_width_arange
            # Do a quick filter to get rid of hot pixels in awkward
            # places.  Do this only for stuff inside small_filter_crop
            # since it is our most time-consuming step.  Also work in
            # our original, potentially binned image, so hot pixels
            # don't get blown up by unbinning.  This is also a natural
            # place to check that we have default_ND_params in the
            # non-flat case and warn accordingly.
            if default_ND_params is None:
                log.warning('No default_ND_params specified in '
                            'non-flat case.  This is likely to result '
                            'in a poor ND_coords calculation.')
                # For filtering hot pixels, this doesn't need to be
                # super precise
                tb_ND_params = RUN_LEVEL_DEFAULT_ND_PARAMS
            else:
                tb_ND_params = default_ND_params
                # Make a copy so we don't mess up the primary data array
            im = self.data.copy()
            xtop = self.binned(self.ND_edges(ytop, tb_ND_params))
            xbot = self.binned(self.ND_edges(ybot, tb_ND_params))
            # Get the far left and right coords, keeping in mind ND
            # filter might be oriented CW or CCW of vertical
            x0 = int(np.min((xbot, xtop))
                     - self.search_margin / self.binning[1])
            x1 = int(np.max((xbot, xtop))
                     + self.search_margin / self.binning[1])
            x0 = np.max((0, x0))
            x1 = np.min((x1, im.shape[1]))
            # This is the operation that messes with the array in place
            im[ytop:ybot, x0:x1] \
                = signal.medfilt(im[ytop:ybot, x0:x1], 
                                 kernel_size=3)
            # Unbin now that we have removed hot pixels from the
            # section we care about
            im = self.im_unbinned(im)
            
        # At this point, im may or may not be a copy of our primary
        # data.  But that is OK, we won't muck with it from now on
        # (promise)

        # The general method is to take the absolute value of the
        # gradient along each row to spot the edges of the ND filter.
        # Because contrast can be low in the Jupiter images, we need
        # to combine n_y_steps rows.  However, since the ND filter can
        # be tilted by ~20 degrees or so, combining rows washes out
        # the edge of the ND filter.  So shift each row to a common
        # center based on the default_ND_params.  Flats are high
        # contrast, so we can use a slightly different algorithm for
        # them and iterate to jump-start the process with them

        ND_edges = [] ; ypts = []

        # Create yrange at y_bin intervals starting at ytop (low
        # number in C fashion) and extending to ybot (high number),
        # chopping of the last one if it goes too far
        y_bin = int((ybot-ytop)/self.n_y_steps)
        #yrange = np.arange(0, im.shape[0], y_bin)
        yrange = np.arange(ytop, ybot, y_bin)
        if yrange[-1] + y_bin > ybot:
            yrange = yrange[0:-1]
            # picturing the image in C fashion, indexed from the top down,
            # ypt_top is the top point from which we bin y_bin rows together

        for ypt_top in yrange:
            # We will be referencing the measured points to the center
            # of the bin
            ycent = ypt_top+y_bin/2

            if default_ND_params is None:
                # We have already made sure we are a flat at this
                # point, so just run with it.  Flats are high
                # contrast, low noise.  When we run this the first
                # time around, features are rounded and shifted by the
                # ND angle, but still detectable.

                # We can chop off the edges of the smaller SII
                # filters to prevent problems with detection of
                # edges of those filters
                bounds = self.small_filt_crop[:,1]
                profile = np.sum(im[ypt_top:ypt_top+y_bin,
                                    bounds[0]:bounds[1]],
                                 0)
                #plt.plot(bounds[0]+np.arange(bounds[1]-bounds[0]), profile)
                #plt.show()
                # Just doing d2 gets two peaks, so multiply
                # by the original profile to kill the inner peaks
                smoothed_profile \
                    = signal.savgol_filter(profile, self.x_filt_width, 3)
                d = np.gradient(smoothed_profile, 10)
                d2 = np.gradient(d, 10)
                s = np.abs(d2) * profile
            else:
                # Non-flat case.  We want to morph the image by
                # shifting each row by by the amount predicted by the
                # default_ND_params.  This lines the edges of the ND
                # filter up for easy spotting.  We will morph the
                # image directly into a subim of just the right size
                default_ND_width = (default_ND_params[1,1]
                                    - default_ND_params[1,0])
                subim_hw = int(default_ND_width/2 + self.search_margin)
                subim = np.zeros((y_bin, 2*subim_hw))

                # rowpt is each row in the ypt_top y_bin, which we need to
                # shift to accumulate into a subim that is the morphed
                # image.
                for rowpt in np.arange(y_bin):
                    # determine how many columns we will shift each row by
                    # using the default_ND_params
                    this_ND_center \
                        = int(
                            np.round(
                                np.mean(
                                    self.ND_edges(
                                        rowpt+ypt_top,
                                        default_ND_params))))
                    dcenter = np.abs(this_ND_center
                                     - self.unbinned(self.shape)[1]/2)
                    if dcenter > self.unbinned(self.shape)[1]/4:
                        raise ValueError('this_ND_center too far off')
                    left = max((0, this_ND_center-subim_hw))
                    right = min((this_ND_center+subim_hw,
                                 this_ND_center+subim.shape[1]-1))
                    #print('(left, right): ', (left, right))
                    subim[rowpt, :] \
                        = im[ypt_top+rowpt, left:right]

                profile = np.sum(subim, 0)
                # This spots the sharp edge of the filter surprisingly
                # well, though the resulting peaks are a little fat
                # (see signal.find_peaks_cwt arguments, below)
                smoothed_profile \
                    = signal.savgol_filter(profile, self.x_filt_width, 0)
                d = np.gradient(smoothed_profile, 10)
                s = np.abs(d)
                # To match the logic in the flat case, calculate
                # bounds of the subim picturing that it is floating
                # inside of the full image
                bounds = im.shape[1]/2 + np.asarray((-subim_hw, subim_hw))
                bounds = bounds.astype(int)

            # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
            # points out same problem I had with with cwt.  It is too
            # sensitive to little peaks.  However, I can find the peaks
            # and just take the two largest ones
            #peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
            #peak_idx = signal.find_peaks_cwt(s, np.arange(2, 80), min_snr=2)
            peak_idx = signal.find_peaks_cwt(s,
                                             cwt_width_arange,
                                             min_snr=self.cwt_min_snr)
            # Need to change peak_idx into an array instead of a list for
            # indexing
            peak_idx = np.array(peak_idx)

            # Give up if we don't find two clear edges
            if peak_idx.size < 2:
                log.debug('No clear two peaks inside bounds ' + str(bounds))
                #plt.plot(s)
                #plt.show()
                continue

            if default_ND_params is None:
                # In the flat case where we are deriving ND_params for
                # the first time, assume we have a set of good peaks,
                # sort on peak size
                sorted_idx = np.argsort(s[peak_idx])
                # Unwrap
                peak_idx = peak_idx[sorted_idx]

                # Thow out if lower peak is too weak.  Use Carey Woodward's
                # trick of estimating the noise on the continuum To avoid
                # contamination, do this calc just over our desired interval
                #ss = s[bounds[0]:bounds[1]]

                #noise = np.std(ss[1:-1] - ss[0:-2])
                noise = np.std(s[1:-1] - s[0:-2])
                #print(noise)
                if s[peak_idx[-2]] < noise:
                    #print("Rejected -- not above noise threshold")
                    continue
                # Find top two and put back in index order
                edge_idx = np.sort(peak_idx[-2:])
                # Sanity check
                de = edge_idx[1] - edge_idx[0]
                if (de < self.max_ND_width_range[0]
                    or de > self.max_ND_width_range[1]):
                    continue

                # Accumulate in tuples
                ND_edges.append(edge_idx)
                ypts.append(ycent)

            else:
                # In lower S/N case.  Compute all the permutations and
                # combinations of peak differences so we can find the
                # pair that is closest to our expected value
                diff_arr = []
                for ip in np.arange(peak_idx.size-1):
                    for iop in np.arange(ip+1, peak_idx.size):
                        diff_arr.append((ip,
                                         iop, peak_idx[iop] - peak_idx[ip]))
                diff_arr = np.asarray(diff_arr)
                closest = np.abs(diff_arr[:,2] - default_ND_width)
                sorted_idx = np.argsort(closest)
                edge_idx = peak_idx[diff_arr[sorted_idx[0], 0:2]]
                # Sanity check
                de = edge_idx[1] - edge_idx[0]
                if (de < self.max_ND_width_range[0]
                    or de > self.max_ND_width_range[1]):
                    continue

                # Accumulate in tuples
                ND_edges.append(edge_idx)
                ypts.append(ycent)
                

            if self.plot_prof:
                plt.plot(profile)
                plt.show()
            if self.plot_dprof:
                plt.plot(s)
                plt.show()

        if len(ND_edges) < 2:
            if default_ND_params is None:
                raise ValueError('Not able to find ND filter position')
            log.warning('Unable to improve filter position over initial guess')
            return default_ND_params
        
        ND_edges = np.asarray(ND_edges) + bounds[0]
        ypts = np.asarray(ypts)
        
        # Put the ND_edges back into the original orientation before
        # we cshifted them with default_ND_params
        if default_ND_params is not None:
            es = []
            for iy in np.arange(ypts.size):
                this_default_ND_center\
                    = np.round(
                        np.mean(
                            self.ND_edges(
                                ypts[iy], default_ND_params)))
                cshift = int(this_default_ND_center - im.shape[1]/2.)
                es.append(ND_edges[iy,:] + cshift)

                #es.append(self.default_ND_params[1,:] - im.shape[1]/2. + self.default_ND_params[0,:]*(this_y - im.shape[0]/2))
            ND_edges =  np.asarray(es)

        if self.plot_ND_edges:
            plt.plot(ypts, ND_edges)
            plt.show()
            

        # Try an iterative approach to fitting lines to the ND_edges
        ND_edges = np.asarray(ND_edges)
        ND_params0 = iter_linfit(ypts-im.shape[0]/2, ND_edges[:,0],
                                 self.max_fit_delta_pix)
        ND_params1 = iter_linfit(ypts-im.shape[0]/2, ND_edges[:,1],
                                 self.max_fit_delta_pix)
        # Note when np.polyfit is given 2 vectors, the coefs
        # come out in columns, one per vector, as expected in C.
        ND_params = np.transpose(np.asarray((ND_params0, ND_params1)))
        
        # DEBUGGING
        #plt.plot(ypts, self.ND_edges(ypts, ND_params))
        #plt.show()

        dp = abs((ND_params[0,1] - ND_params[0,0]) * im.shape[0]/2)
        if dp > self.max_parallel_delta_pix:
            txt = f'ND filter edges are not parallel.  Edges are off by {dp:.0f} pixels.'
            #print(txt)
            #plt.plot(ypts, ND_edges)
            #plt.show()
            
            if default_ND_params is None:
                raise ValueError(txt + '  No initial try available, raising error.')
            log.warning(txt + ' Returning initial try.')
            ND_params = default_ND_params

        return ND_params

    def ND_edges(self, y, external_ND_params=None):
        """Returns unbinned x coords of ND filter edges at given unbinned y coordinate(s)"""
        if external_ND_params is not None:
            ND_params = external_ND_params
        else:
            # Avoid possible recursion error
            assert self.ND_params is not None
            ND_params = self.ND_params

        ND_params = np.asarray(ND_params)
        imshape = self.unbinned(self.shape)
        # --> I might be able to do this as a comprehension
        if np.asarray(y).size == 1:
            return ND_params[1,:] + ND_params[0,:]*(y - imshape[0]/2)
        es = []
        for this_y in y:
            es.append(ND_params[1,:] + ND_params[0,:]*(this_y - imshape[0]/2))
        return es
    
    @pgproperty
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter"""
        
        # ND is referenced
        xs = [] ; ys = []
        us = self.unbinned(self.shape)
        for iy in np.arange(0, us[0]):
            bounds = (self.ND_params[1,:]
                      + self.ND_params[0,:]*(iy - us[0]/2)
                      + self.edge_mask)
            bounds = bounds.astype(int)
            for ix in np.arange(bounds[0], bounds[1]):
                xs.append(ix)
                ys.append(iy)

        # Do a sanity check.  Note C order of indices
        badidx = np.where(np.asarray(ys) > us[0])
        if np.any(badidx[0]):
            raise ValueError('Y dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')
        badidx = np.where(np.asarray(xs) > us[1])
        if np.any(badidx[0]):
            raise ValueError('X dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')

        # NOTE C order and the fact that this is a tuple of tuples
        ND_coords = (ys, xs)
        return ND_coords

    # --> change this to a regular method
    def ND_coords_above(self, level):
        """Returns tuple of coordinates of pixels with decent signal in ND filter"""
        # Get the coordinates of the ND filter
        NDc = self.ND_coords
        abovec = np.where(self.self_unbinned.data[NDc] > level)
        above_NDc0 = np.asarray(NDc[0])[abovec]
        above_NDc1 = np.asarray(NDc[1])[abovec]
        return (above_NDc0, above_NDc1)

    # Turn ND_angle into a "getter"
    @pgproperty
    def ND_angle(self):
        """Calculate ND angle from vertical.  Note this assumes square pixels
        """
        ND_angle = np.degrees(np.arctan(np.average(self.ND_params[0,:])))
        return ND_angle

    @pgproperty
    def background(self):
        return overscan_estimate(self)

    def get_metadata(self):
        # Add our camera metadata.  Note, because there are many ways
        # this object is instantiated (e.g. during arithmetic), makes
        # sure we only do the FITS header manipulations when we are
        # reasonably sure we have our camera.
        instrument = self.meta.get('instrume')
        if (instrument == 'SX-H694'
            or instrument == 'IoIO Coronagraph'):
            self.meta = sx694.metadata(self.meta)
        # Other cameras would be added here...

    @property
    def readnoise(self):
        """Returns CCD readnoise as value in same unit as primary array"""
        self.get_metadata()
        # --> This gets better with FITS header units
        readnoise = self.meta.get('RDNOISE')
        if self.unit == u.adu:
            gain = self.meta.get('GAIN')
            readnoise /= gain
        return readnoise
        
    @property
    def nonlin(self):
        """Returns nonlinearity level in same unit as primary array"""
        self.get_metadata()
        return self.meta.get('NONLIN')# * self.unit
        
    

    @pgcoordproperty
    def obj_center(self):
        """Returns center pixel coords of Jupiter whether or not Jupiter is on ND filter.  Unbinned pixel coords are returned.  Use [Cor]ObsData.binned() to convert to binned pixels.
        """

        # Check to see if we really want to calculate the center
        imagetyp = self.meta.get('IMAGETYP')
        if imagetyp.lower() in ['bias', 'dark', 'flat']:
            self.no_obj_center = True
        if self.no_obj_center:
            return NoCenterPGD(self).obj_center           

        # Work with a copy of the unbinned data array, since we are
        # going to muck with it.  Make sure it is not raw int and flux
        # normalize it
        im = np.double(self.self_unbinned.data.copy()) / (np.prod(self.binning))
        back_level = self.background / (np.prod(self.binning))

        # Establish some metrics to see if Jupiter is on or off the ND
        # filter.  Easiest one is number of saturated pixels
        # /data/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit gives
        # 4090 of these.  Calculation below suggests 1000 should be a
        # good minimum number of saturated pixels (assuming no
        # additional scattered light).  A star off the ND filter
        # /data/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit
        # gives 124 num_sat
        log.debug(f'back_level = {self.background}, self.nonlin: {self.nonlin}, max im: {np.max(im)}')
        num_sat = (im >= self.nonlin).sum()

        # Make a 1D profile along the ND filter to search for a source there
        us = self.unbinned(self.shape)
        ND_profile = np.empty(us[0])
        for iy in np.arange(im.shape[0]):
            es = self.ND_edges(iy).astype(int)
            row = im[iy, es[0]:es[1]]
            # Get rid of cosmic rays
            row = signal.medfilt(row, 3)
            ND_profile[iy] = np.mean(row)

        diffs2 = (ND_profile[1:] - ND_profile[0:-1])**2
        profile_variance = np.sqrt(np.median(diffs2))

        ND_width = (self.ND_params[1,1]
                    - self.ND_params[1,0])
        prof_peak_idx = signal.find_peaks_cwt(ND_profile,
                                              np.linspace(4, ND_width))
        ymax_idx = np.argmax(ND_profile[prof_peak_idx])
        # unwrap
        ymax_idx = prof_peak_idx[ymax_idx]
        ymax = ND_profile[ymax_idx]
        med = np.median(ND_profile)
        std = np.std(ND_profile)
        peak_contrast = (ymax - med)/profile_variance
        log.debug(f'profile peak_contrast = {peak_contrast}, threshold = {self.profile_peak_threshold}, peak y = {ymax_idx}')

        #plt.plot(ND_profile)
        #plt.show()

        source_on_ND_filter = (peak_contrast > self.profile_peak_threshold)

        # Work another way to see if the ND filter has a low flux
        im  = im - back_level
        nonlin = self.nonlin - back_level

        # Come up with a metric for when Jupiter is off the ND filter.
        # Below is my scratch work        
        # Rj = np.asarray((50.1, 29.8))/2. # arcsec
        # plate = 1.59/2 # main "/pix
        # 
        # Rj/plate # Jupiter pixel radius
        # array([ 31.50943396,  18.74213836])
        # 
        # np.pi * (Rj/plate)**2 # Jupiter area in pix**2
        # array([ 3119.11276312,  1103.54018437])
        #
        # Jupiter is generally better than 1000
        # 
        # np.pi * (Rj/plate)**2 * 1000 
        # array([ 3119112.76311733,  1103540.18436529])

        log.debug(f'Number of saturated pixels in image = {num_sat}; bright source threshold = {self.bright_sat_threshold}, minimum source {self.min_source_threshold}')

        bright_source_off_ND = num_sat > self.bright_sat_threshold
        if bright_source_off_ND or not source_on_ND_filter:
            # Outside the ND filter, Jupiter should be saturating.  To
            # make the center of mass calc more accurate, just set
            # everything that is not getting toward saturation to 0
            # --> Might want to fine-tune or remove this so bright
            im[np.where(im < nonlin*0.7)] = 0
            
            # Catch Jupiter at it's minimum 
            # --> This logic doesn't work well to rule out case where
            # there are many bright stars, but to do that would
            # require a lot of extra work like segmentation, which is
            # not worth it for this object.  Looking at stars and
            # getting a good center is the job of PGAstromData or
            # whatever.  If I really wanted to rule out these cases
            # without this effort, I would set min_source_threshold to
            # 1000 or change the logic above
            sum_bright_pixels = np.sum(im)
            if sum_bright_pixels < nonlin * self.min_source_threshold:
                log.debug(f'No bright source found: number of bright pixels ~ {sum_bright_pixels/nonlin} < {self.min_source_threshold} self.min_source_threshold')
                return NoCenterPGD(self).obj_center           

            # If we made it here, Jupiter is outside the ND filter,
            # but shining bright enough to be found
            if bright_source_off_ND and source_on_ND_filter:
                # Both on and off - prefer off
                log.debug('Bright source near ND filter')
            elif not source_on_ND_filter:
                log.debug('No source on ND filter')
            else:
                log.debug('Bright source off of ND filter')
            
            # Use iterative approach
            ny, nx = im.shape
            y_x = np.asarray(center_of_mass(im))
            log.debug(f'First iteration COM (X, Y; binned) = {self.binned(y_x)[::-1]}')
            y = np.arange(ny) - y_x[0]
            x = np.arange(nx) - y_x[1]
            # input/output Cartesian direction by default
            xx, yy = np.meshgrid(x, y)
            rr = np.sqrt(xx**2 + yy**2)
            # --> Make this property of object
            im[np.where(rr > 200)] = 0
            y_x = np.asarray(center_of_mass(im))
            log.debug(f'Second iteration COM (X, Y; binned) = {self.binned(y_x)[::-1]}')
            self.quality = 6
            return y_x

        # If we made it here, we are reasonably sure Jupiter or a
        # suitably bright source is on the ND filter
        log.debug(f'Bright source on ND filter')

        # Zero out all pixels that (1) are not on the ND filter and
        # (2) do not have decent signal.  Not sure the fastest way to
        # do this given that we will work with a patch, below.  This
        # is certainly the quickest to program up
        NDmed = np.median(im[self.ND_coords])
        NDstd = np.std(im[self.ND_coords])
        log.debug(f'ND median, std {NDmed}, {NDstd}, 6*self.readnoise= {6*self.readnoise}')
        boost_factor = nonlin*1000
        boost_ND_coords = self.ND_coords_above(NDmed + 6*self.readnoise)
        im[boost_ND_coords[0], boost_ND_coords[1]] *= boost_factor
        im[np.where(im < boost_factor)] = 0

        # Use the peak on the ND filter to extract a patch over
        # which we calculate the COM
        patch_half_width = ND_width
        patch_half_width = patch_half_width.astype(int)
        es = self.ND_edges(ymax_idx)
        iy_x = np.asarray((ymax_idx, np.average(es))).astype(int)
        ll = iy_x - patch_half_width
        ur = iy_x + patch_half_width
        patch = im[ll[0]:ur[0], ll[1]:ur[1]]
        patch /= boost_factor

        #plt.imshow(patch)
        #plt.show()


        # Check for the case when Jupiter is near the edge of the ND
        # filter.  Optical effects result in bright pixels on the ND
        # filter that confuse the COM.
        bad_ND_coords = self.ND_coords_above(nonlin)
        nbad = len(bad_ND_coords[0])
        bright_on_ND_threshold = 50
        log.debug(f'Number of bright pixels on ND filter = {nbad}; threshold = {bright_on_ND_threshold}')
        if nbad > bright_on_ND_threshold: 
            log.debug(f'Setting bright pixels to ND median value')
            # As per calculations above, this is ~5% of Jupiter's area
            # Experiments with
            # /data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S7.fit
            # and friends show there is an edge of ~5 pixels around
            # the plateau.  Use gaussian filter to fuzz out our saturated pixels
            bad_patch = np.zeros_like(patch)
            bad_patch[np.where(patch > nonlin)] = nonlin
            #plt.imshow(bad_patch)
            #plt.show()
            bad_patch = gaussian_filter(bad_patch, sigma=5)
            #plt.imshow(bad_patch)
            #plt.show()
            patch[np.where(bad_patch > 0.1*np.max(bad_patch))] = NDmed
            #plt.imshow(patch)
            #plt.show()
            pcenter = np.asarray(center_of_mass(patch))
            y_x = pcenter + ll
            log.debug(f'Scattered light cleaned COM (X, Y; binned) = {self.binned(y_x)[::-1]}')        
            
            self.quality = 6
            return y_x

        # If we made it here, Jupiter should be clean on the ND
        # filter.
        pcenter = np.asarray(center_of_mass(patch))
        y_x = pcenter + ll
        log.debug(f'Object COM from clean patch (X, Y; binned) = {self.binned(y_x)[::-1]}')
        self.quality = 8
        return y_x

        ## First iteration COM
        #pcenter = np.asarray(center_of_mass(patch))
        #y_x = pcenter + ll
        #log.debug(f'First iteration COM (X, Y; binned) = {self.binned(y_x)[::-1]}')        
        #
        ##y_x = np.asarray(center_of_mass(im))
        ##log.debug(f'First iteration COM (X, Y; binned) = {self.binned(y_x)[::-1]}')        
        ### Work with a patch of our (background-subtracted) image
        ### centered on the first iteration COM.  
        ##ND_width = self.ND_params[1,1] - self.ND_params[1,0]
        ##patch_half_width = ND_width / 2
        ##patch_half_width = patch_half_width.astype(int)
        ##iy_x = y_x.astype(int)
        ##ll = iy_x - patch_half_width
        ##ur = iy_x + patch_half_width
        ##patch = im[ll[0]:ur[0], ll[1]:ur[1]]
        ##patch /= boost_factor
        #
        #log.debug(f'First iteration cpatch (X, Y) = {cpatch[::-1]}')        
        #plt.imshow(patch)
        #plt.show()
        #
        #
        #
        ### --> Experiment with a really big hammer.  Wasn't any
        ### slower, but didn't give an different answer
        ##pccd = CorData(patch, meta=self.meta)
        ##photometry = Photometry(ccd=pccd)
        ##sc = photometry.source_catalog
        ##tbl = sc.to_table()
        ##tbl.sort('segment_flux', reverse=True)
        ##tbl.show_in_browser()


    @pgproperty
    def obj_to_ND(self):
        """Returns perpendicular distance of obj center to center of ND filter
        """
        if self.quality == 0:
            # This quantity doesn't make sense if there is an invalid center
            return None
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # has a better factor
        imshape = self.unbinned(self.shape)
        m = np.average(self.ND_params[0,:])
        b = np.average(self.ND_params[1,:])
        x1 = 1100; x2 = 1200
        # The line is actually going vertically, so X in is the C
        # convention of along a column.  Also remember our X coordinate
        # is relative to the center of the image
        y1 = m * (x1 - imshape[0]/2)  + b
        y2 = m * (x2 - imshape[0]/2)  + b
        x0 = self.obj_center[0]
        y0 = self.obj_center[1]
        d = (np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
             / ((x2 - x1)**2 + (y2 - y1)**2)**0.5)
        return d

    @pgcoordproperty
    def desired_center(self):
        """Returns Y, X center of ND filter at Y position determined by
        self.y_center_offset.  

        """
        # in 2019, moved this down a little to get around a piece of
        # dust and what looks like a fold in the ND filter unbinned,
        # 70 pixels down in MaxIm was about the right amount (yes,
        # MaxIm 0,0 is at the top left and Python goes Y, X.  And yes,
        # we need to work in unbinned coords, since hot pixels and the
        # ND filter are referenced to the actual CCD
        offset = np.asarray((self.y_center_offset, 0))
        unbinned_desired_center = (self.unbinned(super().desired_center)
                                   + offset)
        y_center = unbinned_desired_center[0]
        x_center = np.average(self.ND_edges(y_center))
        desired_center = np.asarray((y_center, x_center))
        # Check to make sure desired center is close to the center of the image
        ims = np.asarray(self.shape)
        bdc = self.binned(desired_center)
        low = bdc < ims*0.25
        high = bdc > ims*0.75
        if np.any(np.asarray((low, high))):
            raise ValueError('Desired center is too far from center of image.  In original image coordinates:' + str(self.binned(desired_center)))
        return desired_center

    def _card_write(self):
        """Write FITS card unique to CorData"""
        # Priorities ND_params as first-written, since they, together
        # with the boundaries set up by the flats (which initialize
        # the ND_params) provide critical context for all other views
        # of the data
        self.meta['NDPAR00'] = (self.ND_params[0,0],
                                'ND filt left side slope')
        self.meta['NDPAR01'] = (self.ND_params[1,0],
                                'ND filt left side offset at Y cent. of im')
        self.meta['NDPAR10'] = (self.ND_params[0,1],
                                'ND filt right side slope')
        self.meta['NDPAR11'] = (self.ND_params[1,1],
                                'ND filt right side offset at Y cent. of im')
        super()._card_write()
        if self.obj_to_ND is not None:
            self.meta['HIERARCH OBJ_TO_ND_CENTER'] \
                = (self.obj_to_ND,
                   'Obj perp dist to ND filt (pix)')

## This ended up being no better
#class RedCorData(CorData):
#    """This might end up going into the regular CorData if it is not too slow"""
#
#    @pgcoordproperty
#    def obj_center(self):
#        """Refines determination of Jupiter's center on ND filter (if appropriate)"""
#        com_center = super().obj_center
#        # Use ND width as measuring stick
#        # --> This section of code could be a separate procedure.  See
#        # comet_figure for motivation
#        # --> There might already be something that does this in
#        # astropy or elsewhere
#        ND_width = self.ND_params[1,1] - self.ND_params[1,0]
#        print('ND_width:', ND_width)
#        print('obj_to_ND:', self.obj_to_ND)
#        if (self.quality <= 6
#            or (self.obj_to_ND
#                > ND_width / 2)):
#            # Poor quality or just too far out from ND center
#            return com_center
#        patch_half_width = ND_width / 2
#        # --> Check to see what type patch_half_width really is at this point
#        patch_half_width = patch_half_width.astype(int)
#        icom_center = com_center.astype(int)
#        print('icom_center:', icom_center)
#        ll = icom_center - patch_half_width
#        ur = icom_center + patch_half_width
#
#        # Boost what we hope is Jupiter
#        im = np.double(self.self_unbinned.data.copy())
#        # When Jupiter is near the edge of the ND filter, optical
#        # effects result in bright pixels on the ND filter that
#        # confuse the COM.  Try knocking down those bright pixels
#        NDmed = np.median(self.self_unbinned.data[self.ND_coords])
#        bad_ND_coords = self.ND_coords_above(self.nonlin)
#        im[bad_ND_coords[0], bad_ND_coords[1]] = NDmed
#        boost_ND_coords = self.ND_coords_above(NDmed + 6*self.readnoise)
#        im[boost_ND_coords[0], boost_ND_coords[1]] *= 1000
#        patch = im[ll[0]:ur[0], ll[1]:ur[1]]
#        patch -= np.median(patch)
#
#        ## --> experimenting with jupiter-like convolution
#        ##kernel = Gaussian2DKernel(50)
#        #kernel = Gaussian2DKernel(4)
#        #kernel.normalize()
#        ##patch = convolve(patch, kernel)
#        #cpatch = signal.correlate2d(patch, kernel)
#
#        print('patch.shape:', patch.shape)
#        rpatch = np.rot90(patch)
#        plt.imshow(patch)
#        plt.show()
#        plt.imshow(rpatch)
#        plt.show()
#
#        cpatch = signal.correlate2d(patch, rpatch)
#        ccenter = np.unravel_index(np.argmax(cpatch), cpatch.shape)
#        ccenter = np.asarray(ccenter)
#        ccenter = np.asarray(center_of_mass(cpatch))
#        print(f'ccenter = {ccenter}')
#        # Return array is twice the size of patch
#        center = ccenter/2 + ll
#        # --> consider check with com_center
#        dcenter = center - com_center
#        dcenter_threshold = 10
#        if np.linalg.norm(dcenter) > dcenter_threshold:
#            log.warning(f'correlated center more than '
#                        f'{dcenter_threshold} pixels from COM threashold')
#        print(dcenter)
#       
#        plt.imshow(cpatch)
#        plt.show()
#        from ccdmultipipe.utils.ccddata import FbuCCDData
#        out = FbuCCDData(cpatch, unit='adu')
#        out.write('/tmp/cpatch.fits', overwrite=True)
#
#        return center
#
    

        #
        #print('ccenter:', ccenter)
        #print('ccenter/2 + ll:', ccenter/2 + ll)
        ##print(com_center)
        #
        #return com_center        
        
#class TestObs(CorData):
#    pass
#
#bias_fname = '/data/IoIO/reduced/Calibration/2020-07-11_ccdT_-5.3_bias_combined.fits'
#master_bias = CorData.read(bias_fname)
#dfname = '/data/IoIO/raw/20200711/Dark-S005-R003-C010-B1.fts'
#dark = CorData.read(dfname)
##o = overscan_estimate(dark, meta=dark.meta, master_bias=bias_fname)
#o = overscan_estimate(dark, meta=dark.meta, master_bias=master_bias)
#print(o)
#print(dark.meta)
#
#dark = TestObs.read(dfname)
#o = overscan_estimate(dark, meta=dark.meta)
#print(dark.meta)

##from IoIO_working_corobsdata.IoIO import CorObsData
#log.setLevel('DEBUG')
#old_default_ND_params \
#    = [[  3.63686271e-01,   3.68675375e-01],
#       [  1.28303305e+03,   1.39479846e+03]]

##fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_ND_centered.fit'
##flat_fname = '/data/IoIO/raw/2021-04-25/Sky_Flat-0001_R.fit'
##f = CorData.read(flat_fname)
##print(f.ND_params)
##c = CorData.read(fname)#, plot_ND_edges=True)
##print(c.ND_params)
##print(c.ND_angle)
##print("print(c.obj_center, c.desired_center)")
##print(c.obj_center, c.desired_center)


#c = CorData.read(flat_fname, plot_ND_edges=True)
#print(c.ND_params[1,1] - c.ND_params[1,0])
#
#
#
#print(c.ND_params)
##[[-3.32707525e-01, -3.22880506e-01],
## [ 1.23915169e+03,  1.40603931e+03]]
#
##cwt_width_arange_flat = np.arange(2, 60)
##cwt_width_arange        = np.arange(8, 80)
#
#c.cwt_width_arange_flat = np.arange(2, 80)
#c.ND_params = None
#print(c.ND_params)
##[[-3.32707525e-01 -3.14641399e-01]
## [ 1.23915169e+03  1.40151592e+03]]
#c.cwt_width_arange_flat = np.arange(8, 80)
#c.ND_params = None
#print(c.ND_params)
##[[-3.32707525e-01 -3.17529953e-01]
## [ 1.23915169e+03  1.40014752e+03]]
#

#oc = CorObsData(fname, default_ND_params=RUN_LEVEL_DEFAULT_ND_PARAMS)
#print(c.ND_params - oc.ND_params)
#print(c.ND_angle - oc.ND_angle)
#print(c.desired_center - oc.desired_center)
#print(c.obj_center - oc.obj_center)
#print(c.obj_center, c.desired_center)
#c.write('/tmp/test.fits', overwrite=True)

if __name__ == '__main__':
    log.setLevel('DEBUG')
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_ND_centered.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S1.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S2.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S3.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S4.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S5.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S6.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S7.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S8.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S9.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge1.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S10.fit'
    fname = '/data/IoIO/raw/2021-04_Astrometry/Gal_sat_on_ND.fit'
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Main_Astrometry_East_of_Pier.fit'
    #fname = '/data/IoIO/raw/20210616/CK20R040-S001-R001-C001-R_dupe-6.fts'
    #fname = '/data/IoIO/raw/2021-05-18/Mercury-0007_R.fit'
    #fname = '/data/IoIO/raw/2021-05-18/Mercury-0006_Na-on.fit'
    #fname = '/data/IoIO/raw/2021-05-18/Mercury-0006_Na_off.fit'
    #fname = '/data/IoIO/raw/2021-05-18/Mercury-0003_R.fit'

    #fname = '/data/IoIO/raw/2021-04_Astrometry/VegaOnND.fit'
    
    #from IoIO import CorObsData
    #ccd = CorObsData(fname)
    #print(ccd.obj_center)
    #print(ccd.quality)

    ccd = CorData.read(fname)
    print(ccd.obj_center)
    print(ccd.quality)

    #ccd = RedCorData.read(fname)
    ##print(ccd.obj_center)
    ##ccd = CorData.read(fname)
    #print(ccd.obj_center)
    #print(ccd.quality)

    #from IoIO_working_corobsdata.IoIO import CorObsData
    #log.setLevel('DEBUG')
    #old_default_ND_params \
    #    = [[  3.63686271e-01,   3.68675375e-01],
    #       [  1.28303305e+03,   1.39479846e+03]]
    #fname = '/data/IoIO/raw/2021-04_Astrometry/Jupiter_ND_centered.fit'
    ##fname = '/data/IoIO/raw/2021-04-25/Sky_Flat-0001_R.fit'
    #c = CorData.read(fname)#, plot_ND_edges=True)
    #oc = CorObsData(fname, default_ND_params=RUN_LEVEL_DEFAULT_ND_PARAMS)
    #print(c.ND_params - oc.ND_params)
    #print(c.ND_angle - oc.ND_angle)
    #print(c.desired_center - oc.desired_center)
    #print(c.obj_center - oc.obj_center)
    #print(c.obj_center, c.desired_center)
    #c.write('/tmp/test.fits', overwrite=True)
    

    #log.setLevel('DEBUG')
    #default_ND_params \
    #    = [[  3.63686271e-01,   3.68675375e-01],
    #       [  1.28303305e+03,   1.39479846e+03]]
    #
    #default_ND_params = True
    #from IoIO import CorObsData
    #fname = '/data/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit'
    #c = CorData.read(fname, default_ND_params=default_ND_params)#,
    #                 #plot_ND_edges=True)
    #oc = CorObsData(fname, default_ND_params=RUN_LEVEL_DEFAULT_ND_PARAMS)#,
    #                #plot_ND_edges=True)
    #print(c.ND_params - oc.ND_params)
    #print(c.ND_angle - oc.ND_angle)
    #print(c.desired_center - oc.desired_center)
    #print(c.obj_center - oc.obj_center)
    #print(c.obj_center, c.desired_center)
    #c.write('/tmp/test.fits', overwrite=True)
    #
    ##t = c.self_unbinned
    ##print(t)
    ##s = t.divide(t, handle_meta='first_found')
    ##print(s)
    ##print(s.meta)
    ##s = t.divide(3, handle_meta='first_found')
    ##print(s)
    ##print(s.meta)
    ##print(s.arithmetic_keylist)
    #
    #off_filt_fname = '/data/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit'
    #bias_fname = '/data/IoIO/reduced/Calibration/2020-03-26_ccdT_-20.2_bias_combined.fits'
    #dark_fname = '/data/IoIO/reduced/Calibration/2020-03-25_ccdT_-0.3_exptime_3.0s_dark_combined.fits'
    #flat_fname = '/data/IoIO/reduced/Calibration/2020-03-30_Na_off_flat.fits'
    #off = CorData.read(off_filt_fname)
    #bias = CorData.read(bias_fname)
    #dark = CorData.read(dark_fname)
    #flat = CorData.read(flat_fname)
    #
    #lumpy = off.copy()
    #lumpy.unit = u.electron
    #
    #t = lumpy.subtract(bias, handle_meta='first_found')
    #print(f'substract SATLEVEL = {t.meta["satlevel"]} NONLIN = {t.meta["nonlin"]}, med = {np.median(t)}')
    #
    #t = lumpy.divide(flat, handle_meta='first_found')
    #print(f'flat SATLEVEL = {t.meta["satlevel"]} NONLIN = {t.meta["nonlin"]}, med = {np.median(t)}')
    #
    #
    #t = lumpy.subtract(t, handle_meta='first_found')
    ##print(f'self-substract SATLEVEL = {t.meta["satlevel"]} NONLIN = {t.meta["nonlin"]}, med = {np.median(t)}')
    #
    #
    #fname = '/data/IoIO/raw/20200522/SII_on-band_005.fits'
    #c = CorData.read(fname)
    #
    #oc = CorObsData(fname)#, default_ND_params=RUN_LEVEL_DEFAULT_ND_PARAMS)#,
    #                #plot_ND_edges=True)
    #print(c.ND_params - oc.ND_params)
    #print(c.ND_angle - oc.ND_angle)
    #print(c.desired_center - oc.desired_center)
    #print(c.obj_center - oc.obj_center)
    #print(c.obj_center, c.desired_center)
    #c.write('/tmp/test.fits', overwrite=True)
