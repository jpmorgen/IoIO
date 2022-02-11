"""Base class and utilities for the CorData system.

"""

from copy import deepcopy

import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from astropy import log
from astropy import units as u
from astropy.stats import biweight_location

from astropy_fits_key import FitsKeyArithmeticMixin

from precisionguide import pgproperty, pgcoordproperty
from precisionguide import MaxImPGD, NoCenterPGD
from precisionguide.utils import hist_of_im

import sx694

# All global values are referenced to the unbinned, full-frame
# CCD.  Calculations (should be) done such that binned & subframed
# images are kept in their native formats


# ND_params[0, :] represent the slope relative to the *Y-axis* of the
# left and right sides of the ND filter.  ND_params[1,:] represents
# the *unbinned* X-positions of the left and right sides of the ND
# filter at the Y center of the full-frame image.  Transformations to
# and from binned and subframed coordinate systems images are done
# assuming square pixels, in which ND_params[0, :] is invariant, but
# ND_params[1,:] changes.
ND_REF_Y = sx694.naxis2 / 2

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
        Edge size of square box in *unbinned* coordinates used to
        extract biweight median location from the corners of the image
        for this method of overscan estimation.  Default = 100

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

    # Change to base class makes this difficult to implement, since
    # CorData is the old RedCorData
    #if type(ccd) != CorData and ccd.meta.get('subtract_bias') is None:
    #    # Don't gunk up logs when we are taking data, but subclasses
    #    # of CorObs (e.g. RedCorObs) will produce message
    #    log.warning('overscan_estimate: bias has not been subtracted, which can lead to inaccuracy of overscan estimate')

    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.
    # Corners method
    s = ccd.shape
    bs = int(ccd.x_binned(box_size)) # yes, we have square pixels
    c00 = biweight_location(ccd.data[0:bs,0:bs])
    c10 = biweight_location(ccd.data[s[0]-bs:s[0],0:bs])
    c01 = biweight_location(ccd.data[0:bs,s[1]-bs:s[1]])
    c11 = biweight_location(ccd.data[s[0]-bs:s[0],s[1]-bs:s[1]])
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

class CorDataBase(FitsKeyArithmeticMixin, NoCenterPGD, MaxImPGD):
    def __init__(self, data,
                 default_ND_params=None,
                 ND_params=None, # for _slice
                 ND_ref_y=ND_REF_Y,
                 copy=False,
                 **kwargs):
        # Pattern after NDData init but skip all the tests
        if isinstance(data, CorDataBase):
            # Sigh.  We have to undo the convenience of our pgproperty
            # lest we trigger the calculation of property, which leads
            # to recursion problems
            obj_dict = data.__dict__
            ND_params = obj_dict.get('ND_params')
            ND_ref_y = data.ND_ref_y
        if copy:
            ND_params = deepcopy(ND_params)
            ND_ref_y = deepcopy(ND_ref_y)

        super().__init__(data, copy=copy, **kwargs)
        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.
        if ND_params is not None:
            default_ND_params 		= ND_params
        self.default_ND_params 		= default_ND_params
        self.ND_params = ND_params
        self.ND_ref_y        		= ND_ref_y

        self.arithmetic_keylist = ['satlevel', 'nonlin']
        self.handle_image = keyword_arithmetic_image_handler

    def _init_args_copy(self, kwargs):
        kwargs = super()._init_args_copy(kwargs)
        obj_dict = self.__dict__
        kwargs['ND_params'] = obj_dict.get('ND_params')
        kwargs['default_ND_params'] = self.default_ND_params
        kwargs['ND_ref_y'] = self.ND_ref_y
        return kwargs
        
    @pgproperty
    def default_ND_params(self):
        """Returns default ND_params and set Y reference point, self.ND_ref_y

        Values are queried from FITS header, with globals used as fall-back
        """
        
        run_level_default_ND_params = np.asarray(RUN_LEVEL_DEFAULT_ND_PARAMS)
        # Get flat ND params from FITS header, if available
        FND_params = np.full_like(run_level_default_ND_params, np.NAN)
        for i in range(2):
            for j in range(2):
                fndpar = self.meta.get(f'fndpar{i}{j}')
                if fndpar is None:
                    continue
                FND_params[j][i] = fndpar
        if np.all(np.isnan(FND_params)):
            # Nothing in FITS header.  Use our record of
            # default_run_level_ND_params.  Note ifs don't quite
            # bracket ND_params in a nice way, so pay attention when
            # adding subsequent adjustments
            date_obs = self.meta['DATE-OBS']
            if date_obs > '2021-04-25T00:00:00':
                ND_params = run_level_default_ND_params
            elif date_obs > '2019-04-12T00:00:00':
                ND_params = np.asarray([[1.40749551e-02, 2.36320869e-02],
                                        [1.24240593e+03, 1.33789081e+03]])
            elif date_obs > '2019-02-08T00:00:00':
                ND_params = np.asarray([[  4.65269008e-03,   8.76050569e-03],
                                        [  1.27189987e+03,   1.37717911e+03]])
            else:
                ND_params = np.asarray([[  3.63686271e-01,   3.68675375e-01],
                                        [  1.28303305e+03,   1.39479846e+03]])
        if np.any(np.isnan(ND_params)):
            log.warning('malformed FITS header, some FNDPAR values missing!')
            
        ND_ref_y = self.meta.get('ND_REF_Y')
        self.ND_ref_y = ND_ref_y or self.ND_ref_y
        return ND_params

    @pgproperty
    def imagetyp(self):
        imagetyp = self.meta.get('imagetyp')
        if imagetyp is None:
            return None
        return imagetyp.lower()

    @pgcoordproperty
    def edge_mask(self):
        """Pixels on *inside* edge of ND filter to remove when calculating
        ND_coords.  Use a negative value to return coordinates
        extending beyond the ND filter, e.g. for masking.  Stored as a
        tuple (left, right), thus if an asymmetric value is desired,
        right should the negative of left.  If just one value is
        provided, the setter automatically negates it for the right
        hand value"""

    @edge_mask.setter
    def edge_mask(self, edge_mask):
        edge_mask = np.asarray(edge_mask)
        if edge_mask.size == 1:
            edge_mask = np.append(edge_mask, -edge_mask)
        return edge_mask        

    def ND_params_unbinned(self, ND_params_in):
        ND_params = ND_params_in.copy()
        ND_params[1, :] = self.x_unbinned(ND_params[1, :])
        return ND_params

    def ND_params_binned(self, ND_params_in):
        ND_params = ND_params_in.copy()
        ND_params[1, :] = self.x_binned(ND_params[1, :])
        return ND_params

    def ND_edges(self, y, ND_params, ND_ref_y):
        """Returns x coords of ND filter edges and center at given y(s)

        Parameters
        ---------
        y : int or numpy.ndarray
            Input y values referenced to the current self.data view
            (e.g. may be binned and subframed)

        ND_params : numpy.ndarray
            Proper ND_params for self.data view

        ND_ref_y : int
            Proper Y reference point for self.data view

        Returns
        -------
        edges, midpoint : tuple
            Edges is a 2-element numpy.ndarray float, 
            midpoint is a numpy scalar float

        """
        #print(f'in ND_edges, ND_params = {ND_params[1,:]}')
        if np.isscalar(y):
            es = ND_params[1,:] + ND_params[0,:]*(y - ND_ref_y)
            mid = np.mean(es)
            return es, mid
        es = []
        mid = []
        for ty in y:
            tes = ND_params[1,:] + ND_params[0,:]*(ty - ND_ref_y)
            mid.append(np.mean(tes))
            es.append(tes)            
        return np.asarray(es), np.asarray(mid)
                         
    @pgproperty
    def ND_params(self):
        """Returns default parameters which characterize the coronagraph ND
        filter for the cases where precise correction for instrument
        flexure is not needed.

        """
        return self.default_ND_params

    @pgproperty
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter referenced to the potentially binned and subframed image and including the edge mask.  Change the edge-maks property, set this to None and it will recalculate the next time you ask for it"""
        xs = np.asarray((), dtype=int) ; ys = np.asarray((), dtype=int)
        ND_params = self.ND_params_binned(self.ND_params)
        ND_ref_y = self.y_binned(self.ND_ref_y)
        edge_mask = self.edge_mask/self.binning[1]
        for iy in np.arange(self.shape[0]):
            ND_es, _ = self.ND_edges(iy, ND_params, ND_ref_y)
            bounds = ND_es + edge_mask
            if (np.all(bounds < 0)
                or np.all(bounds > self.shape[1])):
                continue
            bounds = bounds.astype(int)
            bounds[0] = np.max((0, bounds[0]))
            bounds[1] = np.min((bounds[1], self.shape[1]))
            more_xs = np.arange(bounds[0], bounds[1])
            xs = np.append(xs, more_xs)
            ys = np.append(ys, np.full_like(more_xs, iy))
        ND_coords = np.asarray((ys, xs))
        array_of_tuples = map(tuple, ND_coords)
        tuple_of_tuples = tuple(array_of_tuples)
        return tuple_of_tuples

    def ND_coords_above(self, level):
        """Returns tuple of coordinates of pixels with decent signal in ND filter"""
        # Get the coordinates of the ND filter
        NDc = self.ND_coords
        im = self.data
        abovec = np.where(im[NDc] > level)
        if abovec is None:
            return None
        above_NDc0 = np.asarray(NDc[0])[abovec]
        above_NDc1 = np.asarray(NDc[1])[abovec]
        return (above_NDc0, above_NDc1)
    
        #print(f'abovec = {abovec}')
        ##return(abovec)
        ## Unwrap
        #abovec = NDc[abovec]
        #print(f'abovec = {abovec}')
        #return abovec

    # Turn ND_angle into a "getter"
    @pgproperty
    def ND_angle(self):
        """Calculate ND angle from vertical.  Note this assumes square pixels
        """
        ND_angle = np.degrees(np.arctan(np.average(self.ND_params[0,:])))
        return ND_angle

    @pgproperty
    def background(self):
        """This might eventually get better, but for now just returns overscan"""
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
        
    def _card_write(self):
        """Write FITS card unique to CorData"""
        # Priorities ND_params as first-written, since they, together
        # with the boundaries set up by the flats (which initialize
        # the ND_params) provide critical context for all other views
        # of the data
        self.meta['NDPAR00'] = (self.ND_params[0,0],
                                'ND filt left side slope')
        self.meta['NDPAR01'] = (self.ND_params[1,0],
                                'Full frame X dist of ND filt left side at ND_REF_Y')
        self.meta['NDPAR10'] = (self.ND_params[0,1],
                                'ND filt right side slope')
        self.meta['NDPAR11'] = (self.ND_params[1,1],
                                'Full frame X dist of ND filt right side at ND_REF_Y')
        self.meta['ND_REF_Y'] = (self.ND_ref_y,
                                'Full-frame Y reference point of ND_params')
        super()._card_write()
        if self.quality > 5:
            self.meta['HIERARCH OBJ_TO_ND_CENTER'] \
                = (self.obj_to_ND,
                   'Obj perp dist to ND filt (pix)')

if __name__ == "__main__":
    log.setLevel('DEBUG')
    ccd = CorDataBase.read('/data/IoIO/raw/2017-03-14/Bias-0001_1x1.fit')
    print(f'background = {ccd.background}')
    ccd = CorDataBase.read('/data/IoIO/raw/2017-03-14/Bias-0001_2x2.fit')
    print(f'background = {ccd.background}')
    ccd = CorDataBase.read('/data/IoIO/raw/2017-03-14/Bias-0001_4x4.fit')
    print(f'background = {ccd.background}')

    print(f'ND_params = {ccd.ND_params}')

    ccd = CorDataBase.read('/data/IoIO/raw/2017-03-14/IPT-0001_off-band.fit')
    print(f'background = {ccd.background}')
    print(f'ND_params = {ccd.ND_params}')

    print(overscan_estimate(ccd, show=True))

    # Binned 2x2
    fname = '/data/IoIO/raw/2021-04_Astrometry/Main_Astrometry_East_of_Pier.fit'
    ccd = CorDataBase.read(fname)

    print(overscan_estimate(ccd, show=True))
    print(f'ND_params = {ccd.ND_params}')

