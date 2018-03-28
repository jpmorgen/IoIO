#!/usr/bin/python3
#--> Getting command line to work in Windows.  You need to edit
# registry to make 
# Computer\HKEY_CLASSES_ROOT\Applications\python.exe\shell\open\command
# "C:\ProgramData\Anaconda3\python.exe" "%1" %*
# Thanks to https://stackoverflow.com/questions/29540541/executable-python-script-not-take-sys-argv-in-windows

#from jpm_fns import display
# for testing
import define as D
import importlib

import argparse
import sys
import os
import socket
import time
import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from scipy import signal, ndimage
# For GuideBoxCommander/GuideBoxMover system --> may improve
import json
import subprocess

if socket.gethostname() == "snipe":
    raw_data_root = '/data/io/IoIO/raw'
elif socket.gethostname() == "puppy":
    # --> This doesn't work.  I need Unc?
    #raw_data_root = '//snipe/data/io/IoIO/raw'
    raw_data_root = r'\\snipe\data\io\IoIO\raw'
    default_telescope = 'ScopeSim.Telescope'
elif socket.gethostname() == "IoIO1U1":
    raw_data_root = r'C:\Users\PLANETARY SCIENCE\Desktop\IoIO\data'
    # --> Eventually, it would be nice to have this in a chooser
    default_telescope = 'AstroPhysicsV2.Telescope'

# --> I may improve this location or the technique of message passing
default_guide_box_command_file = os.path.join(raw_data_root, 'GuideBoxCommand.txt')
default_guide_box_log_file = os.path.join(raw_data_root, 'GuideBoxLog.txt')

run_level_default_ND_params \
    = [[  3.63686271e-01,   3.68675375e-01],
       [  1.28303305e+03,   1.39479846e+03]]


# These are shared definitions between Windows and Linux
class ObsData():
    """Base class for observations, enabling object centering, etc.

    This is intended to work in an active obsering setting, so
    generally an image array will be received, the desired properties
    will be calculated from it and those properties will be read by
    the calling code.

    """

    def __init__(self, HDUList_im_or_fname=None,
                 desired_center=None):
        if HDUList_im_or_fname is None:
            raise ValueError('No HDUList_im_or_fname provided')
        # Set up our basic FITS image info
        self.fname = None
        self.header = None
        self._binning = None
        self._subframe_origin = None
        self._we_opened_file = None
        # This is the most basic observation data to be tracked.
        # These are in pixels
        self._obj_center = None
        self._desired_center = desired_center
        if not self._desired_center is None:
            self._desired_center = np.asarray(self._desired_center)
        # --> Work with these
        self.obj_center_err = np.asarray((1.,1.))
        self.desired_center_tolerance = np.asarray((5.,5.))
        # 0 -- 10 scale indicating quality of obj_center and
        # desired_center calculations
        self.quality = 0
        # astropy time object for calc_flex_pix_rate
        self.TRateChange = None
        self.Tmidpoint = None
        # Amount of guide box motion since first observation
        # units=main camera pixels
        self.total_flex_dpix = None
        
        # one-time motion, just before exposure
        self.delta_pix = None
        # Make the guts of __init__ methods that can be overridden
        # Read our image
        self.read_im(HDUList_im_or_fname)
        # Populate our object
        self.populate_obj()
        self.cleanup()
        
    def populate_obj(self):
        """Calculate quantities that will be stored long-term in object"""
        # Note that if MaxIm is not configured to write IRAF-complient
        # keywords, IMAGETYP gets a little longer and is capitalized
        # http://diffractionlimited.com/wp-content/uploads/2016/11/sbfitsext_1r0.pdf
        kwd = self.header['IMAGETYP'].upper()
        if 'DARK' in kwd or 'BIAS' in kwd or 'FLAT' in kwd:
            raise ValueError('Not able to process IMAGETYP = ' + self.header['IMAGETYP'])
        # Do our work & leave the results in the property
        self.obj_center
        self.desired_center
        # --> CHANGE ME BACK
        self._desired_center = np.asarray((1100, 1150))


    def cleanup(self):
        """Close open file, deference large array"""
        if self._we_opened_file:
            self.close_fits()
        del self.HDUList


    def read_im(self, HDUList_im_or_fname=None):
        """Returns an astropy.fits.HDUList given a filename, image or
        HDUList.  If you have a set of HDUs, you'll need to put them
        together into an HDUList yourself, since this can't guess how
        to do that"""
        if HDUList_im_or_fname is None:
            log.info('No error, just saying that you have no image.')
            HDUList = None
        elif isinstance(HDUList_im_or_fname, fits.HDUList):
            HDUList = HDUList_im_or_fname
            self.fname = HDUList.filename()
        elif isinstance(HDUList_im_or_fname, str):
            HDUList = fits.open(HDUList_im_or_fname)
            self.fname = HDUList.filename()
            self._we_opened_file = True
        elif isinstance(HDUList_im_or_fname, np.ndarray):
            hdu = fits.PrimaryHDU(HDUList_im_or_fname)
            HDUList = fits.HDUList(hdu)
        else:
            raise ValueError('Not a valid input, HDUList_im_or_fname')
        if HDUList is not None:
            # Store the header in our object.  This is just a
            # reference at first, but after HDUList is deleted, this
            # becomes the only copy
            # https://stackoverflow.com/questions/22069727/python-garbage-collector-behavior-on-compound-objects
            self.header = HDUList[0].header
            # Calculate an astropy Time object for the midpoint of the
            # observation for ease of time delta calculations.
            # Account for darktime, if available
            exptime = self.header.get('DARKTIME') 
            if exptime is None:
                exptime = self.header['EXPTIME']
            # Use units to help with astropy.time calculations
            exptime *= u.s
            self.Tmidpoint = (Time(self.header['DATE-OBS'], format='fits')
                              + exptime/2)
            try:
                # Note Astropy Pythonic transpose Y, X order
                self._binning = (self.header['YBINNING'],
                                 self.header['XBINNING'])
                self._binning = np.asarray(self._binning)
                # This is in binned coordinates
                self._subframe_origin = (self.header['YORGSUBF'],
                                         self.header['XORGSUBF'])
                self._subframe_origin = np.asarray(self._subframe_origin)
            except:
                log.warning('Could not read binning or subframe origin from image header.  Did you pass a valid MaxIm-recorded image and header?  Assuming binning = 1, subframe_origin = 0,0')
                self._binning = np.asarray((1,1))
                self._subframe_origin = (0,0)
        self.HDUList = HDUList
        return self.HDUList
    
    def unbinned(self, coords):
        """Returns coords referenced to full CCD given internally stored binning/subim info"""
        coords = np.asarray(coords)
        return np.asarray(self._binning * coords + self._subframe_origin)

    def binned(self, coords):
        """Assuming coords are referenced to full CCD, return location in binned coordinates relative to the subframe origin"""
        coords = np.asarray(coords)
        return np.asarray((coords - self._subframe_origin) / self._binning)
        
    def HDU_unbinned(self):
        """Unbin primary HDU image
        """
        a = self.HDUList[0].data
        # Don't bother if we are already unbinned
        if np.sum(self._binning) == 2:
            return a
        newshape = self._binning * a.shape
        # From http://scipy-cookbook.readthedocs.io/items/Rebinning.html
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new)
                   for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        unbinned = a[tuple(indices)]
        # Check to see if we need to make a larger array into which to
        # plop unbinned array
        if np.sum(self._subframe_origin) > 0:
            # Note subframe origin reads in binned pixels
            origin = self.unbinned(self._subframe_origin)
            full_unbinned = np.zeros(origin + unbinned.shape)
            full_unbinned[origin[0]:, origin[1]:] = unbinned
            unbinned = full_unbinned
        return unbinned

    def close_fits(self):
        if self.HDUList.fileinfo is not None:
            self.HDUList.close()
            self._we_opened_file = None

    def iter_linfit(self, x, y, max_resid=None):
        """Performs least squares linear fit iteratively to discard bad points

        If you actually know the statistical weights on the points,
        just use polyfit directly.

        """
        # Let polyfit report errors in x and y
        coefs = np.polyfit(x, y, 1)
        # We are done if we have just two points
        if len(x) == 2:
            return coefs
            
        # Our first fit may be significantly pulled off by bad
        # point(s), particularly if the number of points is small.
        # Construct a repeat until loop the Python way with
        # while... break to iterate to squeeze bad points out with
        # low weights
        last_redchi2 = None
        iterations = 1
        while True:
            # Calculate weights roughly based on chi**2, but not going
            # to infinity
            yfit = x * coefs[0] + coefs[1]
            resid = (y - yfit)
            if resid.all == 0:
                break
            # Add 1 to avoid divide by zero error
            resid2 = resid**2 + 1
            # Use the residual as the variance + do the algebra
            redchi2 = np.sum(1/(resid2))
            coefs = np.polyfit(x, y, 1, w=1/resid2)
            # Converge to a reasonable epsilon
            if last_redchi2 and last_redchi2 - redchi2 < np.finfo(float).eps*10:
                break
            last_redchi2 = redchi2
            iterations += 1

        # The next level of cleanliness is to exclude any points above
        # max_resid from the fit (if specified)
        if max_resid is not None:
            goodc = np.where(np.abs(resid) < max_resid)
            # Where returns a tuple of arrays!
            if len(goodc[0]) >= 2:
                coefs = self.iter_linfit(x[goodc], y[goodc])
        return coefs
    
        
    def hist_of_im(self, im, readnoise):
        """Returns histogram of image and index into centers of bins.  
Uses readnoise (default = 5 e- RMS) to define bin widths
        """
        if not readnoise:
            readnoise = 5
        # Code from west_aux.py, maskgen.

        # Histogram bin size should be related to readnoise
        hrange = (im.min(), im.max())
        nbins = int((hrange[1] - hrange[0]) / readnoise)
        hist, edges = np.histogram(im, bins=nbins,
                                   range=hrange, density=True)
        # Convert edges of histogram bins to centers
        centers = (edges[0:-1] + edges[1:])/2
        #plt.plot(centers, hist)
        #plt.show()
        return (hist, centers)

    def back_level(self, im, **kwargs):
        # Use the histogram technique to spot the bias level of the image.
        # The coronagraph creates a margin of un-illuminated pixels on the
        # CCD.  These are great for estimating the bias and scattered
        # light for spontanous subtraction.  The ND filter provides a
        # similar peak after bias subutraction (or, rather, it is the
        # second such peak)
        # --> This is very specific to the coronagraph.  Consider porting first peak find from IDL
        # Pass on readnoise, if supplied
        im_hist, im_hist_centers = self.hist_of_im(im, kwargs)
        im_peak_idx = signal.find_peaks_cwt(im_hist, np.arange(10, 50))
        return im_hist_centers[im_peak_idx[0]]
        #im -= im_hist_centers[im_peak_idx[0]]

    def imshow(self, im=None):
        if im is None:
            im = self.HDUList[0].data
        plt.imshow(im)
        plt.show()

    @property
    def obj_center(self):
        """Returns pixel coordinate of the brightests object in the image in
        UNBINNED Y, X coordinates.  Does basic median filtering to get
        rid of cosmic rays.  It is assumed this will be overridden
        with better object finders, such as one that uses PinPoint
        astrometry.

        """
    
        if self._obj_center is not None:
            return self._obj_center
        # Take the median to get rid of cosmic rays
        im = self.HDUList[0].data
        im = signal.medfilt(im, kernel_size=3)
        im_center = np.unravel_index(np.argmax(im), im.shape)
        # Pretty-print our object center before we unbin
        log.debug('Object center (X, Y; binned) = ' + str(im_center[::-1]))
        self._obj_center = self.unbinned(im_center)
        # Set quality just above the border, since we haven't done
        # much work on this
        self.quality = 6
        return self._obj_center

    @property
    def desired_center(self):
        """If desired_center hasn't been explicitly set, this returns the
        geometric center of image.  NOTE: The return order of indices
        is astropy FITS Pythonic: Y, X

        """
        if self._desired_center is not None:
            return self._desired_center
        im = self.HDUList[0].data
        im_center = np.asarray(im.shape)/2
        self._desired_center = self.unbinned(im_center)
        return self._desired_center

    # Allow user to move desired center around
    @desired_center.setter
    def desired_center(self, value):
        self._desired_center = value
    

    # --> I don't think I need these
    ## World coordinates may be calculated by of some subclasses.
    ## Worst case scenario, we calculate them with MaxImData.scope_wcs
    ## when we need them
    #@property
    #def w_obj_center(self):
    #    """World coordinates of object center"""
    #    return self._w_obj_center
    #    
    #@w_obj_center.setter
    #def w_obj_center(self, value):
    #    self._w_obj_center = value
    #
    #@property
    #def w_desired_center(self):
    #    """World coordinates of object center"""
    #    return self._w_desired_center
    #    
    #@w_desired_center.setter
    #def w_desired_center(self, value):
    #    self._w_desired_center = value
    #
    #@property
    #def dra_ddec(self):
    #    if self._dra_ddec is not None:
    #        return self._dra_ddec
    #    # This will raise its own error if the world coordinates have
    #    # not been calculated
    #    self._dra_ddec = self.w_obj_center - self.w_desired_center
    #    return self._dra_ddec

class CorObsData(ObsData):
    """Object for containing coronagraph image data used for centering Jupiter
    """
    # This is for jump-starting the ND_params calculation with flats.
    # Be generous in case the angle is very large.  If it is still
    # hard to get a good ND solution, use more n_y_steps
    def __init__(self,
                 HDUList_im_or_fname=None,
                 default_ND_params=None,
                 y_center=None,
                 n_y_steps=8, # was 15
                 x_filt_width=25,
                 edge_mask=5,
                 cwt_width_arange=None, # Default set by image type in populate_obj
                 cwt_min_snr=1, # Their default seems to work well
                 search_margin=50, # on either side of nominal ND filter
                 max_fit_delta_pix=25, # Thowing out point in 1 line fit
                 max_parallel_delta_pix=50, # Find 2 lines inconsistent
                 max_ND_width_range=[80,400], # jump-starting flats & sanity check others
                 biasnoise=20, # std of a typical bias image
                 plot_prof=False,
                 plot_dprof=False,
                 plot_ND_edges=False):

        # These should be invariants of the instrument:
        self.SII_filt_crop = np.asarray(((350, 550), (1900, 2100)))

        self.y_center = y_center

        # Define defaults for ND mask finding algorithm.  It is easy
        # to find the ND mask in flats, but not Jupiter images.  We
        # can use the answer from the flats to get the answer for
        # Jupiter.  This was from:
        # '/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit'
        #self.default_ND_params = ((-7.35537190e-02,  -6.71900826e-02), 
        #                          (1.24290909e+03,   1.34830909e+03))
        #
        ## And we can refine it further for a good Jupiter example
        ##print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit',
        ##                  initial_try=((-7.35537190e-02,  -6.71900826e-02), 
        ##                               (1.24290909e+03,   1.34830909e+03))))
        #self.default_ND_params = ((-6.57640346e-02,  -5.77888855e-02),
        #                          (1.23532221e+03,   1.34183584e+03))
        #
        ## flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
        #self.default_ND_params = ((3.78040775e-01,  3.84787113e-01),
        #                          (1.24664929e+03,   1.35807856e+03))
        #
        ## flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
        #self.default_ND_params = ((3.75447820e-01,  3.87551301e-01),
        #                          (1.18163633e+03,   1.42002571e+03))
        
        #self.default_ND_params = ((3.73276728e-01,   3.89055377e-01),
        #                          (1.12473263e+03,   1.47580210e+03))

        #--> temporary
        #self.default_ND_params = np.asarray(self.default_ND_params)

        self.default_ND_params = None
        if default_ND_params is not None:
            self.default_ND_params = np.asarray(default_ND_params)

        self.n_y_steps =              n_y_steps              
        self.x_filt_width =           x_filt_width           
        self.edge_mask =              edge_mask              
        self.cwt_width_arange =       cwt_width_arange       
        self.cwt_min_snr =            cwt_min_snr            
        self.search_margin =           search_margin           
        self.max_fit_delta_pix =      max_fit_delta_pix      
        self.max_parallel_delta_pix = max_parallel_delta_pix
        self.max_ND_width_range	    = max_ND_width_range
        self.biasnoise		    = biasnoise
        self.plot_prof		    = plot_prof 
        self.plot_dprof             = plot_dprof
        self.plot_ND_edges	    = plot_ND_edges

        # Flats use a different algorithm
        self.isflat = None
        
        # The ND_params are the primary property we work hard to
        # generate.  These will be the slopes and intercepts of the
        # two lines defining the edges of the ND filter.  The origin
        # of the lines is the Y center of the unbinned, full-frame
        # chip
        self._ND_params = None
        # These are the coordinates into the ND filter
        self._ND_coords = None
        # Angle is the (average) angle of the lines, useful for cases
        # where the filter is rotated significantly off of 90 degrees
        # (which is where I will run it frequently)
        self._ND_angle = None
        # Distance from center of object to center of ND filter
        self._obj_to_ND = None
        # Inherit init from base class, which does basic FITS reading,
        # calls populate_obj, and cleanup methods
        super().__init__(HDUList_im_or_fname)

    def populate_obj(self):
        """Calculate quantities that will be stored long-term in object"""
        # Note that if MaxIm is not configured to write IRAF-complient
        # keywords, IMAGETYP gets a little longer and is capitalized
        # http://diffractionlimited.com/wp-content/uploads/2016/11/sbfitsext_1r0.pdf
        kwd = self.header['IMAGETYP'].upper()
        if 'DARK' in kwd or 'BIAS' in kwd:
            raise ValueError('Not able to process IMAGETYP = ' + self.header['IMAGETYP'])
        # We can go as far as the N_params for flats.  In fact, we
        # have to to get a good default_ND_params for LIGHT frames
        if 'FLAT' in kwd:
            self.isflat = True

        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.
        if self.y_center is None:
            self.y_center = self.HDUList[0].data.shape[0]*self._binning[0]/2

        # See if our image has already been through the system.  This
        # saves us the work of using self.get_ND_params
        if self.header.get('NDPAR00') is not None:
            ND_params = np.zeros((2,2))
            # Note transpose, since we are working in C!
            ND_params[0,0] = self.header['NDPAR00']
            ND_params[1,0] = self.header['NDPAR01']
            ND_params[0,1] = self.header['NDPAR10']
            ND_params[1,1] = self.header['NDPAR11']
            self._ND_params = ND_params
        else:
            if not self.isflat and self.default_ND_params is None:
                self.default_ND_params = np.asarray(run_level_default_ND_params)
                #log.info('Setting default_ND_params from run_level_default_ND_params' + str(self.default_ND_params))
                
        # Get ready to generate the ND_params, which is our hardest work
        
        # The flats produce very narrow peaks in the ND_param
        # algorithm when processed without a default_ND_param and
        # there is a significant filter rotation.  Once things are
        # morphed by the default_ND_params (assuming they match the
        # image), the peaks are much broader
        if self.cwt_width_arange is None:
            if self.default_ND_params is None:
                self.cwt_width_arange = np.arange(2, 60)
            else:
                self.cwt_width_arange = np.arange(8, 80)

        # Do our work & leave the results in the property
        self.ND_params
        if self.isflat:
            return
        self.obj_center
        self.desired_center
        self.obj_to_ND


    @property
    def obj_center(self):
        """Returns center pixel coords of Jupiter whether or not Jupiter is on ND filter.  Unbinned pixel coords are returned.  Use [Cor]Obs_Data.binned() to convert to binned pixels.
        """
        # Returns stored center for object, None for flats
        if self._obj_center is not None or self.isflat:
            return self._obj_center
        
        # Work with unbinned image
        im = self.HDU_unbinned()

        # Establish some metrics to see if Jupiter is on or off the ND
        # filter.  Easiest one is number of saturated pixels
        # /data/io/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit gives
        # 4090 of these.  Calculation below suggests 1000 should be a
        # good minimum number of saturated pixels (assuming no
        # additional scattered light).  A star off the ND filter
        # /data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit
        # gives 124 num_sat
        satc = np.where(im > 60000)
        num_sat = len(satc[0])
        #log.debug('Number of saturated pixels in image: ' + str(num_sat))

        # Work another way to see if the ND filter has a low flux
        im = im - self.back_level(im)
        
        # Get the coordinates of the ND filter
        NDc = self.ND_coords

        # Filter ND coords for ones that are at least 5 std of the
        # bias noise above the median.  Calculate a fresh median for
        # the ND filter just in case it is different than the median
        # of the image as a whole (which is now 0 -- see above).  We
        # can't use the std of the ND filter, since it is too biased
        # by Jupiter when it is there.
        NDmed = np.median(im[NDc])
        boostc = np.where(im[NDc] > (NDmed + 5*self.biasnoise))
        boost_NDc0 = np.asarray(NDc[0])[boostc]
        boost_NDc1 = np.asarray(NDc[1])[boostc]

        # Come up with a metric for when Jupiter is in the ND filter.
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
        
        sum_on_ND_filter = np.sum(im[boost_NDc0, boost_NDc1])
        #log.debug('sum of significant pixels on ND filter = ' + str(sum_on_ND_filter))
        if num_sat > 1000 or sum_on_ND_filter < 1E6:
            log.warning('Jupiter outside of ND filter?')
            # Outside the ND filter, Jupiter should be saturating.  To
            # make the center of mass calc more accurate, just set
            # everything that is not getting toward saturation to 0
            # --> Might want to fine-tune or remove this so bright
            im[np.where(im < 40000)] = 0
            
            #log.debug('Approx number of saturating pixels ' + str(np.sum(im)/65000))

            # 25 worked for a star, 250 should be conservative for
            # Jupiter (see above calcs)
            # if np.sum(im) < 65000 * 25:
            if np.sum(im) < 65000 * 250:
                self.quality = 4
                log.warning('Jupiter (or suitably bright object) not found in image.  This object is unlikely to show up on the ND filter.  Seeting quality to ' + str(self.quality))
            else:
                self.quality = 6
            # If we made it here, Jupiter is outside the ND filter,
            # but shining bright enough to be found
            # --> only look for very bright pixels
            y_x = np.asarray(ndimage.measurements.center_of_mass(im))
            self._obj_center = y_x
            return self._obj_center

        # Here is where we boost what is sure to be Jupiter, if Jupiter is
        # in the ND filter
        im[boost_NDc0, boost_NDc1] *= 1000
        # Clean up any signal from clouds off the ND filter, which can
        # mess up the center of mass calculation
        im[np.where(im < 65000)] = 0
        y_x = ndimage.measurements.center_of_mass(im)

        #print(y_x[::-1])
        #plt.imshow(im)
        #plt.show()
        #return (y_x[::-1], ND_center)

        # Stay in Pythonic y, x coords
        self._obj_center = np.asarray(y_x)
        log.debug('Object center (X, Y; binned) = ' + str(self.binned(self._obj_center)[::-1]))
        self.quality = 6
        return self._obj_center

    @property
    def desired_center(self):
        """Returns Y, X center of ND filter at Y position self.y_center in unbinned coordinates

        Default self.y_center is set to ny/2 at instantiation of
        object but can be modified & this routine will calculate
        proper X value

        """
        # Returns stored center for object, None for flats
        if self._desired_center is not None or self.isflat:
            return self._desired_center
        desired_center = np.asarray((self.y_center, np.average(self.ND_edges(self.y_center))))
        # Check to make sure desired center is close to the center of the image
        ims = np.asarray(self.HDUList[0].data.shape)
        bdc = self.binned(desired_center)
        low = bdc < ims*0.25
        high = bdc > ims*0.75
        if np.any(np.asarray((low, high))):
            raise ValueError('Desired center is too far from center of image.  In original image coordinates:' + repr(self.binned(desired_center)))
        self._desired_center = desired_center
        return self._desired_center
        
    #@desired_center.setter
    # --> Consider making this able to put the object anywhere on the
    # image (currently inherited) _and_ move it around relative to the
    # ND filter

    @property
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter"""
        if self._ND_coords is not None:
            return self._ND_coords

        # --> consider making this faster with coordinate math

        # Work with unbinned image
        im = self.HDU_unbinned()
        
        xs = [] ; ys = []
        for iy in np.arange(0, im.shape[0]):
            bounds = (self.ND_params[1,:]
                      + self.ND_params[0,:]*(iy - im.shape[0]/2)
                      + np.asarray((self.edge_mask, -self.edge_mask)))
            bounds = bounds.astype(int)
            for ix in np.arange(bounds[0], bounds[1]):
                xs.append(ix)
                ys.append(iy)

        # Do a sanity check.  Note C order of indices
        badidx = np.where(np.asarray(ys) > im.shape[0])
        if np.any(badidx[0]):
            raise ValueError('Y dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')
        badidx = np.where(np.asarray(xs) > im.shape[1])
        if np.any(badidx[0]):
            raise ValueError('X dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')

        self._ND_coords = (ys, xs)
        # NOTE C order and the fact that this is a tuple of tuples
        return self._ND_coords

    def ND_edges(self, y, external_ND_params=None):
        """Returns unbinned x coords of ND filter edges at given unbinned y coordinate(s)"""
        if external_ND_params is not None:
            ND_params = external_ND_params
        else:
            # Avoid recursion error
            assert self._ND_params is not None
            ND_params = self.ND_params

        ND_params = np.asarray(ND_params)
        imshape = self.unbinned(self.HDUList[0].data.shape)
        # --> I might be able to do this as a comprehension
        if np.asarray(y).size == 1:
            return ND_params[1,:] + ND_params[0,:]*(y - imshape[0]/2)
        es = []
        for this_y in y:
            es.append(ND_params[1,:] + ND_params[0,:]*(this_y - imshape[0]/2))
        return es
    
    # Turn ND_angle into a "getter"
    @property
    def ND_angle(self):
        """Calculate ND angle from vertical.  Note this assumes square pixels
        """
        if self._ND_angle is not None or self.isflat:
            return self._ND_angle
    
        # If we made it here, we need to calculate the angle
        # get_ND_params should have caught pathological cases, so we can
        # just use the average of the slopes

        self._ND_angle = np.degrees(np.arctan(np.average(self.ND_params[0,:])))
        return self._ND_angle

    @property
    def obj_to_ND(self):
        """Returns perpendicular distance of obj center to center of ND filter
        """
        if self._obj_to_ND is not None or self.isflat:
            return self._obj_to_ND
        
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # has a better factor
        imshape = self.unbinned(self.HDUList[0].data.shape)
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
        self._obj_to_ND = d
        return self._obj_to_ND

    @property
    def ND_params(self):
        """Returns parameters which characterize the coronagraph ND filter, calculating if necessary"""
        if self._ND_params is not None:
            return self._ND_params

        # If we made it here, we need to do the heavy lifting of
        # finding the ND filter in the image
        assert isinstance(self.HDUList, fits.HDUList)

        # These are unbinned coordinates
        ytop = self.SII_filt_crop[0,0]
        ybot = self.SII_filt_crop[1,0]

        # Trying a filter to get rid of cosmic ray hits in awkward
        # places.  Do this only for section of CCD we will be working
        # with, since it is our most time-consuming step
        if not self.isflat:
            im = self.HDUList[0].data
            xtop = self.binned(self.ND_edges(ytop, self.default_ND_params))
            xbot = self.binned(self.ND_edges(ybot, self.default_ND_params))
            x0 = int(np.min(xtop) - self.search_margin / self._binning[1])
            x1 = int(np.max(xtop) + self.search_margin / self._binning[1])
            x0 = np.max((0, x0))
            x1 = np.min((x1, im.shape[1]))
            im[ytop:ybot, x0:x1] \
                = signal.medfilt(im[ytop:ybot, x0:x1], 
                                 kernel_size=3)
        im = self.HDU_unbinned()

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

            if self.default_ND_params is None:
                # We have already made sure we are a flat at this
                # point, so just run with it.  Flats are high
                # contrast, low noise.  When we run this the first
                # time around, features are rounded and shifted by the
                # ND angle, but still detectable.

                # We can chop off the edges of the smaller SII
                # filters to prevent problems with detection of
                # edges of those filters
                bounds = self.SII_filt_crop[:,1]
                profile = np.sum(im[ypt_top:ypt_top+y_bin,
                                    bounds[0]:bounds[1]],
                                 0)
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
                default_ND_width = (self.default_ND_params[1,1]
                                    - self.default_ND_params[1,0])
                subim_hw = int(default_ND_width/2 + self.search_margin)
                subim = np.zeros((y_bin, 2*subim_hw))

                # rowpt is each row in the ypt_top y_bin, which we need to
                # shift to accumulate into a subim that is the morphed
                # image.
                for rowpt in np.arange(y_bin):
                    # determine how many columns we will shift each row by
                    # using the default_ND_params
                    this_ND_center \
                        = np.int(
                            np.round(
                                np.mean(
                                    self.ND_edges(
                                        rowpt+ypt_top,
                                        self.default_ND_params))))
                    subim[rowpt, :] \
                        = im[ypt_top+rowpt, 
                             this_ND_center-subim_hw:this_ND_center+subim_hw]

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
                                             self.cwt_width_arange,
                                             min_snr=self.cwt_min_snr)
            # Need to change peak_idx into an array instead of a list for
            # indexing
            peak_idx = np.array(peak_idx)

            # Give up if we don't find two clear edges
            if peak_idx.size < 2:
                log.info('No clear two peaks inside bounds ' + str(bounds))
                #plt.plot(s)
                #plt.show()
                continue

            if self.default_ND_params is None:
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
            if self.default_ND_params is None:
                raise ValueError('Not able to find ND filter position')
            log.warning('Unable to improve filter position over initial guess')
            self._ND_params = self.default_ND_params
            return self._ND_params
            
        ND_edges = np.asarray(ND_edges) + bounds[0]
        ypts = np.asarray(ypts)
        
        # Put the ND_edges back into the original orientation before
        # we cshifted them with default_ND_params
        if self.default_ND_params is not None:
            es = []
            for iy in np.arange(ypts.size):
                this_default_ND_center\
                    = np.round(
                        np.mean(
                            self.ND_edges(
                                ypts[iy], self.default_ND_params)))
                cshift = int(this_default_ND_center - im.shape[1]/2.)
                es.append(ND_edges[iy,:] + cshift)

                #es.append(self.default_ND_params[1,:] - im.shape[1]/2. + self.default_ND_params[0,:]*(this_y - im.shape[0]/2))
            ND_edges =  np.asarray(es)

        if self.plot_ND_edges:
            plt.plot(ypts, ND_edges)
            plt.show()
        

        # Try an iterative approach to fitting lines to the ND_edges
        ND_edges = np.asarray(ND_edges)
        ND_params0 = self.iter_linfit(ypts-im.shape[0]/2, ND_edges[:,0],
                                      self.max_fit_delta_pix)
        ND_params1 = self.iter_linfit(ypts-im.shape[0]/2, ND_edges[:,1],
                                      self.max_fit_delta_pix)
        # Note when np.polyfit is given 2 vectors, the coefs
        # come out in columns, one per vector, as expected in C.
        ND_params = np.transpose(np.asarray((ND_params0, ND_params1)))
                
        # DEBUGGING
        #plt.plot(ypts, self.ND_edges(ypts, ND_params))
        #plt.show()

        dp = abs((ND_params[0,1] - ND_params[0,0]) * im.shape[0]/2)
        if dp > self.max_parallel_delta_pix:
            txt = 'ND filter edges are not parallel.  Edges are off by ' + str(dp) + ' pixels.'
            #print(txt)
            #plt.plot(ypts, ND_edges)
            #plt.show()
            
            if self.default_ND_params is None:
                raise ValueError(txt + '  No initial try available, raising error.')
            log.warning(txt + ' Returning initial try.')
            ND_params = self.default_ND_params

        self._ND_params = ND_params
        # The HDUList headers are objects, so we can do this
        # assignment and the original object property gets modified
        h = self.HDUList[0].header
        # Note transpose, since we are working in C!
        self.header['NDPAR00'] = (ND_params[0,0], 'ND filt left side slope at Y center of im')
        self.header['NDPAR01'] = (ND_params[1,0], 'ND filt left side offset at Y center of im')
        self.header['NDPAR10'] = (ND_params[0,1], 'ND filt right side slope at Y center of im')
        self.header['NDPAR11'] = (ND_params[1,1], 'ND filt right side offset at Y center of im')

        return self._ND_params

    #def pos(self, default_ND_params=self.default_ND_params,):
        
    # Code provided by Daniel R. Morgenthaler, May 2017
    #
    #
        #This is a picture string. ^
    #
    #
    #    if nd_pos is None:
    #        print('These are 4 numbers and nd_pos is none.')
    def area(self, width_nd_pos, length_nd_pos, variable=True):
        if variable is False or variable is None:
            print('The area of the netral density filter is ' +
                  str(width_nd_pos * length_nd_pos) +  '.')
        elif variable is True:
            return str(width_nd_pos * length_nd_pos)
        else:
            raiseValueError('Use True False or None in variable')
        
    def perimeter(self, width_nd_pos,  length_nd_pos, variable=True):
        if variable is False or variable is None:
            print('The perimeter of the netral density filter is ' +
                  str(width_nd_pos * 2 + 2 *  length_nd_pos) +  '.')
        elif variable is True:
            return str(width_nd_pos * 2 + 2 *  length_nd_pos) +  '.'
        else:
            raiseValueError('Use True False or None in variable')
            
    def VS(self,v1,value1,v2,value2,v3,value3):
        v1=value1 ; v2=value2 ; v3=value3
        return (v1, v2, v3)

    # Code above was provided by Daniel R. Morgenthaler, May 2017

def get_default_ND_params(dir='.', maxcount=None):
    """Derive default_ND_params from up to maxcount flats in dir
    """
    if not os.path.isdir(dir):
        raise ValueError("Specify directory to search for flats and derive default_ND_params")
    if maxcount is None:
        maxcount = 10

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    flats = []
    for f in sorted(files):
        if 'flat' in f.lower():
            flats.append(os.path.join(dir, f))

    # Create default_ND_params out of the flats in this dir.
    ND_params_list = []
    # Just do 10 flats
    for fcount, f in enumerate(flats):
        try:
            # Iterate to get independent default_ND_params for
            # each flat
            default_ND_params = None
            for i in np.arange(3):
                F = CorObsData(f, default_ND_params=default_ND_params)
                default_ND_params = F.ND_params
        except ValueError as e:
            log.error('Skipping: ' + f + '. ' + str(e))
            continue
        ND_params_list.append(default_ND_params)
        if fcount >= maxcount:
            break
    if len(ND_params_list) == 0:
        raise ValueError('No good flats found in ' + dir)

    # If we made it here, we have a decent list of ND_params.  Now
    # take the median to create a really nice default_ND_params
    ND_params_array = np.asarray(ND_params_list)
    default_ND_params \
        = ((np.median(ND_params_array[:, 0, 0]),
            np.median(ND_params_array[:, 0, 1])),
           (np.median(ND_params_array[:, 1, 0]),
            np.median(ND_params_array[:, 1, 1])))
    return np.asarray(default_ND_params)

def ND_params_tree(rawdir='/data/io/IoIO/raw'):
    """Calculate ND_params for all observations in a directory tree
    """
    start = time.time()
    # We have some files recorded before there were flats, so get ready to
    # loop back for them
    skipped_dirs = []
    
    # list date directory one level deep
    totalcount = 0
    totaltime = 0
    dirs = [os.path.join(rawdir, d)
            for d in os.listdir(rawdir) if os.path.isdir(os.path.join(rawdir, d))]
    persistent_default_ND_params = None
    for d in sorted(dirs):
        D.say(d)
        try:
            default_ND_params = get_default_ND_params(d)
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            log.error('Problem with flats: ' + str(e))
            if persistent_default_ND_params is not None:
                default_ND_params = persistent_default_ND_params
        
        result = ND_params_dir(d, default_ND_params=default_ND_params)
        print(result)
        totalcount += result[0]
        totaltime += result[1]
        
    end = time.time()
    D.say('Total elapsed time: ' + str(end - start) + 's')
    D.say(str(totalcount) + ' obj files took ' + str(totaltime) + 's')
    D.say('Average time per file: ' + str(totalcount / totaltime) + 's')


def ND_params_dir(dir=None, default_ND_params=None):
    """Calculate ND_params for all observations in a directory
    """
    if default_ND_params is None:
        try:
            default_ND_params = get_default_ND_params(dir)
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            raise ValueError('Problem with flats in ' + dir + ': '  + str(e))
            
            if persistent_default_ND_params is not None:
                default_ND_params = persistent_default_ND_params

    # Collect file names
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    objs = []
    for f in sorted(files):
        if 'flat' in f.lower():
            pass
        elif 'bias' in f.lower():
            pass 
        elif 'dark' in f.lower():
            pass
        else:
            objs.append(os.path.join(dir, f))

        start = time.time()

    count = 0
    for count, f in enumerate(objs):
        D.say(f)
        try:
            O = CorObsData(f, default_ND_params=default_ND_params)
            D.say(O.obj_center)
            if O.obj_to_ND > 30:
                log.warning('Large dist: ' + str(int(O.obj_to_ND)))
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            log.error('Skipping: ' + str(e))

    elapsed = time.time() - start
        
    return((count, elapsed, count/elapsed))

class test_independent():
    def __init__(self):
        self.loop_state = True
    def loop(self, interval=2):
        while self.loop_state:
            D.say('Before sleep')
            time.sleep(interval)
            D.say('After sleep')
            yield

def guide_calc(x1, y1, fits_t1=None, x2=None, y2=None, fits_t2=None, guide_dt=10, guide_dx=0, guide_dy=0, last_guide=None, aggressiveness=0.5, target_c = np.asarray((1297, 1100))):
    """ Calculate offset guider values given pixel and times"""

    # Pixel scales in arcsec/pix
    main_scale = 1.59/2
    guide_scale = 4.42
    typical_expo = 385 * u.s
    
    if last_guide is None:
        guide_dt = guide_dt * u.s
        previous_dc_dt = np.asarray((guide_dx, guide_dy)) / guide_dt
    else:
        guide_dt = last_guide[0]
        previous_dc_dt = np.asarray((last_guide[1], last_guide[2])) / guide_dt

    # Convert input time interval to proper units
    
    # time period to use in offset guide file
    new_guide_dt = 10 * u.s

    if fits_t1 is None:
        t1 = Time('2017-01-01T00:00:00', format='fits')
    else:
        t1 = Time(fits_t1, format='fits')
    if fits_t2 is None:
        # Take our typical exposure time to settle toward the center
        t2 = t1 + typical_expo
    else:
        if fits_t1 is None:
            raise ValueError('fits_t1 given, but fits_t1 not supplied')
        t2 = Time(fits_t2, format='fits')
    dt = (t2 - t1) * 24*3600 * u.s / u.day

    c1 = np.asarray((x1, y1))
    c2 = np.asarray((x2, y2))
    
    if x2 is None and y2 is None:
        latest_c = c1
        measured_dc_dt = 0
    else:
        latest_c = c2
        measured_dc_dt = (c2 - c1) / dt * main_scale / guide_scale

    # Despite the motion of the previous_dc_dt, we are still getting
    # some motion form the measured_dc_dt.  We want to reverse that
    # motion and add in a little more to get to the target center.  Do
    # this gently over the time scale of our previous dt, moderated by
    # the aggressiveness
    
    target_c_dc_dt = (latest_c - target_c) / dt * aggressiveness
    print(target_c_dc_dt * dt)

    r = new_guide_dt * (previous_dc_dt - measured_dc_dt - target_c_dc_dt)
    
    # Print out new_rates
    print('{0} {1:+.3f} {2:+.3f}'.format(new_guide_dt/u.s, r[0], r[1]))

    return (new_guide_dt, r[0], r[1])

# Keep our Windows code conveniently in the same module
if sys.platform == 'win32':
    # These are needed for MaxImData
    import win32com.client
    import ASCOM_namespace as ASCOM

    # --> these are things that eventually I would want to store in a
    # --> configuration file
    # --> CHANGE ME BACK TO 1s(or 7s) and filter 0 (0.7s or 0.3 on
    # --> Vega filter 1 works for day) 
    default_exptime = 1
    default_filt = 0
    default_cent_tol = 5   # Pixels
    default_guider_exptime = 1 # chage back to 1 for night, 0.2 for day
    run_level_main_astrometry = os.path.join(
        raw_data_root, '2018-01-18/PinPointSolutionEastofPier.fit')
    run_level_guider_astrometry = os.path.join(
        raw_data_root, '2018-01-24/GuiderPinPointSolutionWestofPier.fit')
    #raw_data_root, '2018-01-24/GuiderPinPointSolutionEastofPier.fit')

    #Daniel
    if True:
        class MakeList():
            def __init__(self, list_, item, makelist=True):
                if makelist is True:
                    list_=[]
            def append(self, item):
                list_.append(item)
    #Daniel
    
    
    class MaxImData():
        """Stores data related to controlling MaxIm DL via ActiveX/COM events.
    
        Notes: 
    
        MaxIm camera, guide camera, and telescope must be set up properly
        first (e.g. you have used the setup for interactive observations).
        Even so, the first time this is run, keep an eye out for MaxIm
        dialogs, as this program will hang until they are answered.  To
        fix this, a wathdog timer would need to be used.
    
        Technical note for downstreeam object use: we don't have access to
        the MaxIm CCDCamera.ImageArray, but we do have access to similar
        information (and FITS keys) in the Document object.  The CCDCamera
        object is linked to the actual last image read, where the Document
        object is linked to the currently active window.  This means the
        calling routine could potentially expect the last image read in
        but instead get the image currently under focus by the user.  The
        solution to this is to (carefully) use notify events to interrupt
        MaxIm precisely when the event you expect happens (e.g. exposure
        or guide image acuired).  Then you are sure the Document object
        has the info you expect.  Beware that while you have control,
        MaxIm is stuck and bad things may happen, like the guider might
        get lost, etc.  If your program is going to take a long time to
        work with the information it just got, figure out a way to do so
        asynchronously
    
        """
    
        def __init__(self,
                     main_astrometry=None,
                     guider_astrometry=None,
                     default_filt=default_filt):
            self.main_astrometry = main_astrometry
            self.guider_astrometry = guider_astrometry
            # --> Eventually this might be some first-time init to
            # record the images and run PinPoint on them
            if self.main_astrometry is None:
                self.main_astrometry = run_level_main_astrometry
            if self.guider_astrometry is None:
                self.guider_astrometry = run_level_guider_astrometry
            self.default_filt = default_filt

            # Mount & guider information --> some of this might
            # eventually be tracked by this application in cases where
            # it is not available from the mount
            self.alignment_mode = None
            self.guider_cal_pier_side = None
            self.guide_rates = None # degrees/s
            self.guider_exptime = None
            self.guider_commanded_running = None
            # --> Eventually make this some sort of configurable
            # --> thing, since not all filter wheels need it
            self.main_filt_change_time = 10 # seconds it takes to guarantee filter change 
            
            # Don't move the guide box too fast
            self.guide_box_steps_per_pix = 2
            self.guider_settle_cycle = 5
            self.guider_settle_tolerance = 0.5
            self.loop_sleep_time = 0.2
            self.guider_max_settle_time = 120 # seconds
            
            # Create containers for all of the objects that can be
            # returned by MaxIm.  We'll only populate them when we need
            # them.  Some of these we may never use or write code for
            self.Application = None
            self.CCDCamera = None
            self.Document = None
            self.Telescope = None
            self.telescope_connectable = None
            # ASCOM enumerations -- not working the way I want, so
            # importing ASCOM_namespace
            self.DeviceInterface = None
            
            # There is no convenient way to get the FITS header from MaxIm
            # unless we write the file and read it in.  Instead allow for
            # getting a selection of FITS keys to pass around in a
            # standard astropy fits HDUList
            self.FITS_keys = None
            self.HDUList = None
            self.required_FITS_keys = ('DATE-OBS', 'EXPTIME', 'EXPOSURE', 'XBINNING', 'YBINNING', 'XORGSUBF', 'YORGSUBF', 'FILTER', 'IMAGETYP', 'OBJECT')
    
            # We can use the CCDCamera.GuiderMaxMove[XY] property for an
            # indication of how long it is safe to press the guider
            # movement buttons
            self.guider_max_move_multiplier = 20
            self.connect()
            self.populate_obj()

        #def __del__(self):
        #    # Trying to keep camera from getting disconnected on exit
        #    # --> What seems to work is having FocusMax recycled after
        #    # this firsts connected (though not positive of that)
        #    self.CCDCamera.LinkEnabled == True

        # --> I think gencache might let me do what I want
        def getDeviceInterface(self):
            """THIS DOES NOT WORK.  I guess name space is handled another way, poassibly at the source code level.  I am importing ASCOM_namespace to handle this"""
            if self.DeviceInterface is not None:
                return True
            try:
                self.DeviceInterface = win32com.client.Dispatch('ASCOM.DeviceInterface')
            except:
                raise EnvironmentError('Error instantiating ASCOM DeviceInterface.  Is ASCOM installed? -- No, the real problem is the DeviceInterface thing is some sort of namespace include/import which I have not figured out how to get access to the way I plug into COM')
            # Catch any other weird errors
            assert isinstance(self.DeviceInterface, win32com.client.CDispatch)
            
        def getTelescope(self):
            if self.Telescope is not None:
                return
            try:
                # --> This will keep trying as we do things, in case
                # people have turned on the telescope
                self.Telescope = win32com.client.Dispatch(default_telescope)
                self.telescope_connectable = True
            except:
                log.warning('Not able to connect to telescope.  Some features like auto pier flip for German equatorial mounts (GEMs) and automatic declination compensation for RA motions will not be available.')
                self.telescope_connectable = False

                #raise EnvironmentError('Error instantiating telescope control object ' + default_telescope + '.  Is the telescope on and installed?')
            # Catch any other weird errors
            #assert isinstance(self.Telescope, win32com.client.CDispatch)
            
        def getApplication(self):
            if self.Application is not None:
                return True
            try:
                self.Application = win32com.client.Dispatch("MaxIm.Application")
            except:
                raise EnvironmentError('Error creating MaxIM application object.  Is MaxIM installed?')
            # Catch any other weird errors
            assert isinstance(self.Application, win32com.client.CDispatch)

        def getCCDCamera(self):
            if self.CCDCamera is not None:
                return True
            try:
                self.CCDCamera = win32com.client.Dispatch("MaxIm.CCDCamera")
                #win32com.client.WithEvents(self.CCDCamera,
                #                           self.CCDCameraEventHandler)
            except:
                raise EnvironmentError('Error creating CCDCamera object.  Is there a CCD Camera set up in MaxIm?')
            # Catch any other weird errors
            assert isinstance(self.CCDCamera, win32com.client.CDispatch)

        # --> This is an event handler that doesn't work
        class CCDCameraEventHandler():
            #"""This hopefully magically receives the names of events from the client""" 
            # https://vlasenkov.blogspot.ru/2017/03/python-win32com-multithreading.html
            
            def CCDCamera_Notify(self, event_code):
                log.debug('Received event_code = ' + str(event_code))
                

        def getDocument(self):
            """Gets the document object of the last CCD camera exposure"""
            #"""Gets the document object of the current window"""
            # The CurrentDocument object gets refreshed when new images
            # are taken, so all we need is to make sure we are connected
            # to begin with

            # NOTE: the Application.CurrentDocument is not what we
            # want, since that depends on which windows has focus.
            if self.Document is not None:
                return True
            self.getCCDCamera()
            try:
                self.Document = self.CCDCamera.Document
            except:
                raise EnvironmentError('Error retrieving document object')
            #self.getApplication()
            #try:
            #    self.Document = self.Application.CurrentDocument
            #except:
            #    raise EnvironmentError('Error retrieving document object')
            # Catch any other weird errors
            assert isinstance(self.Document, win32com.client.CDispatch)

        def connect(self):
            """Link to telescope, CCD camera(s), filter wheels, etc."""

            # MaxIm can connect to the telescope and use things like
            # pier side to automatically adjust guiding calculations,
            # but it doesn't make the telescope pier side available to
            # the user.  That means we need to connect separately for
            # our calculations.  Furthermore, ACP doesn't like to have
            # MaxIm connected to the telescope while guiding (except
            # through the ASCOM guide ports or relays), so we need to
            # do everything out-of-band
            self.getTelescope()
            if self.telescope_connectable:
                self.Telescope.Connected = True
                if self.Telescope.Connected == False:
                    raise EnvironmentError('Link to telescope failed.  Is the power on to the mount?')
            self.getApplication()
            ## --> ACP doesn't like MaxIm being connected to the
            ## --> telescope.  We will have to use the property of
            ## --> telecsope and copy over to appropriate places in
            ## --> MaxIm, as if we were operating by hand
            #self.Application.TelescopeConnected = True
            #if self.Application.TelescopeConnected == False:
            #    raise EnvironmentError('MaxIm link to telescope failed.  Is the power on to the mount?')
            self.getCCDCamera()
            self.CCDCamera.LinkEnabled = True
            if self.CCDCamera.LinkEnabled == False:
                raise EnvironmentError('Link to camera hardware failed.  Is the power on to the CCD (including any connection hardware such as USB hubs)?')

        def populate_obj(self, guider_astrometry=None):
            """Called by connect() to fill our object property with useful info"""

            # Fill our object with things we know
            if self.telescope_connectable:
                self.alignment_mode = self.Telescope.AlignmentMode
            else:
                # --> Eventually this warning might go away
                log.error("Mount is not connected -- did you specify one in setup [currently the software source code]?  If you have a German equatorial mount (GEM), this software will likely not work properly upon pier flips [because code has not yet been written to let you specify the mode of your telescope on the fly].  Other mount types will work OK, but you should keep track of the Scope Dec. box in MaxIm's Guide tab.")

            # Calculate guider rates based on astrometry and
            # guider calibration.  Possibly we could be called
            # again with a different astrometry
            if guider_astrometry is None:
                guider_astrometry = self.guider_astrometry
            # We want to get the center of the CCD, which means we
            # need to dig into the FITS header
            if isinstance(guider_astrometry, str):
                HDUList = fits.open(guider_astrometry)
                astrometry = HDUList[0].header
                HDUList.close()
            # Create a vector that is as long as we are willing to
            # move in each axis.  The origin of the vector is
            # reference point of the CCD (typically the center)
            # Work in unbinned pixels
            x0 = astrometry['XBINNING'] * astrometry['CRPIX1'] + astrometry['XORGSUBF']
            y0 = astrometry['YBINNING'] * astrometry['CRPIX2'] + astrometry['YORGSUBF']
            dt = (self.guider_max_move_multiplier
                  * self.CCDCamera.GuiderMaxMoveX)
            dx = (self.CCDCamera.GuiderXSpeed * dt)
                  #/ np.cos(np.radians(astrometry['CRVAL2'])))
            dy = self.CCDCamera.GuiderYSpeed * dt
            # GuiderAngle is measured CCW from N according to
            # http://acp.dc3.com/RotatedGuiding.pdf I think this
            # means I need to rotate my vector CW to have a simple
            # translation between RA and DEC after my astrometric
            # transformation
            ang_ccw = self.CCDCamera.GuiderAngle
            vec = self.rot((dx, dy), -ang_ccw)
            #--> try CW & it is a litle worse -- camera angle is
            #-178, so hard to tell
            #vec = self.rot((dx, dy), ang_ccw)
            x1 = x0 + vec[0]
            y1 = y0 + vec[1]
            # Transpose, since we are in pix
            w_coords = self.scope_wcs(((y0, x0), (y1, x1)),
                                      to_world=True,
                                      astrometry=guider_astrometry,
                                      absolute=True)
            dra_ddec = w_coords[1, :] - w_coords[0, :]
            self.calculated_guide_rates = np.abs(dra_ddec/dt)
            if not self.telescope_connectable:
                self.guide_rates = self.calculated_guide_rates
            else:
                # Always assume telescope reported guide rates are
                # correct, but warn if guider rates are off by 10%
                self.guide_rates \
                    = np.asarray((self.Telescope.GuideRateRightAscension,
                                  self.Telescope.GuideRateDeclination))
                if (np.abs(self.calculated_guide_rates[0]
                          - self.Telescope.GuideRateRightAscension)
                    > 0.1 * self.Telescope.GuideRateRightAscension):
                    log.warning('Calculated RA guide rate is off by more than 10% (scope reported, calculated): ' + str((self.Telescope.GuideRateRightAscension, self.calculated_guide_rates[0])) + '.  Have you specified the correct guider astrometery image?  Have you changed the guide rates changed since calibrating the guider?  Assuming reported telescope guide rates are correct.')
                if (np.abs(self.calculated_guide_rates[1]
                          - self.Telescope.GuideRateDeclination)
                    > 0.1 * self.Telescope.GuideRateDeclination):
                    log.warning('Calculated DEC guide rate is off by more than 10% (scope reported, calculated): ' + str((self.Telescope.GuideRateDeclination, self.calculated_guide_rates[1])) + '.  Have you specified the correct guider astrometery image?  Have you changed the guide rates changed since calibrating the guider?  Assuming reported telescope guide rates are correct.')
                
                    

            # Now run the calculation the other way, not taking
            # out the guider angle, to determine the pier side
            # when calibration took place
            ra0 = astrometry['CRVAL1']
            dec0 = astrometry['CRVAL2']
            dra_ddec = self.guide_rates * dt
            ra1 = ra0 + dra_ddec[0]
            dec1 = dec0 + dra_ddec[1]
            p_coords = self.scope_wcs((ra1, dec1),
                                      to_pix=True,
                                      astrometry=guider_astrometry,
                                      absolute=True)
            # remember transpose
            dp = p_coords[::-1] - np.asarray((x0, y0))

            # Calculate our axis flip RELATIVE to guider astrometry.
            # Note that only one axis is needed, since both flip in
            # the astrometric sense (only RA flips in the motion
            # sense, assuming DEC motor commanded consistently in the
            # same direction)
            if np.sign(dp[0] / self.CCDCamera.GuiderXSpeed) == 1:
                if self.alignment_mode == ASCOM.algGermanPolar:
                    if astrometry['PIERSIDE'] == 'EAST':
                        self.guider_cal_pier_side = ASCOM.pierEast
                    elif astrometry['PIERSIDE'] == 'WEST':
                        self.guider_cal_pier_side = ASCOM.pierWest
            else:
                if self.alignment_mode != ASCOM.algGermanPolar:
                    log.error('German equatorial mount (GEM) pier flip detected between guider astrometry data and guider calibration but mount is currently not reporting alignment mode ' + str(self.alignment_mode) + '.  Did you change your equipment?')
                    # Set our alignment mode, just in case we find
                    # it useful, but this is a fortuitous catch,
                    # since clibration and astrometry could have
                    # been recorded on the same side of the pier.
                    # --> Ultimately some interface would be
                    # needed to track mount type and flip state if
                    # not reported
                    #self.alignment_mode = ASCOM.algGermanPolar
                if astrometry['PIERSIDE'] == 'EAST':
                    self.guider_cal_pier_side = ASCOM.pierWest
                elif astrometry['PIERSIDE'] == 'WEST':
                    self.guider_cal_pier_side = ASCOM.pierEast
                else:
                    # --> interface would want to possibly record this
                    log.error('German equatorial mount (GEM) pier flip detected between guider astrometry data and guider calibration but mount not reporting PIERSIDE in guider astrometry file.  Was this file recorded with MaxIm?  Was the mount properly configured through an ASCOM driver when the calibration took place?')
                        
        def guider_cycle(self, n=1):
            """Returns average and RMS guider error magnitude after n guider cycles

            Parameters
            ----------
            n : int like
                Number of guider cycles.  Default = 1

            norm : boolean
                Return norm of guider error.  Default False


            """
            # --> This would be better with events
            if not self.CCDCamera.GuiderRunning:
                log.warning('Guider not running')
                return None
            this_norm = 0
            last_norm = 0
            running_total = 0
            running_sq = 0
            for i in range(n):
                while True:
                    # Wait until MaxIm gets the first measurement.
                    # eps is epsilon, a very small number.  Don't
                    # forget to force the logic to get a measurement!
                    this_norm = 0
                    while this_norm < 1000 * np.finfo(np.float).eps:
                        # --> this needs a timeout
                        time.sleep(self.loop_sleep_time)
                        this_norm = np.linalg.norm(
                            (self.CCDCamera.GuiderYError,
                             self.CCDCamera.GuiderXError))
                    #log.debug('this_norm: ' + str(this_norm))
                    #log.debug('last_norm: ' + str(last_norm)) 
                    if (np.abs(last_norm - this_norm)
                        > 1000 * np.finfo(np.float).eps):
                        # We have a new reading.
                        break
                    # Keep looking for the new reading
                    last_norm = this_norm
                    #log.debug('last_norm after reassignment: ' + str(last_norm))
                running_total += this_norm
                running_sq += this_norm**2
                last_norm = this_norm
            return (running_total/n, (running_sq/n)**0.5)

        def guider_settle(self):
            """Wait for guider to settle"""
            if not self.CCDCamera.GuiderRunning:
                log.warning('Guider not running')
                return False
            start = time.time()
            now = start
            av = self.guider_settle_tolerance + 1
            rms = av
            while (rms > self.guider_settle_tolerance
                   and av > self.guider_settle_tolerance
                   and time.time() <= start + self.guider_max_settle_time):
                av, rms = self.guider_cycle(self.guider_settle_cycle)
                log.debug('guider AV, RMS = ' + str((av, rms)))
            if time.time() > start + self.guider_max_settle_time:
                log.warning('Guider failed to settle after ' + str(self.guider_max_settle_time) + 's')
                return False
            log.debug('GUIDER SETTLED TO ' + str(self.guider_settle_tolerance) + ' GUIDER PIXELS')
            return True

        def move_with_guide_box(self,
                                dra_ddec,
                                dec=None,
                                guider_astrometry=None):
            """Moves the telescope by moving the guide box.  Guide box position is moved gradually relative to instantaneous guide box position, resulting in a delta move relative to any other guide box motion

            Parameters
            ----------
            dra_ddec : tuple-like array
            delta move in RA and DEC in DEGREES
            guider_astrometry : filename, HDUList, or FITS header 
                Input method for providing an HDUList with WCS
                parameters appropriate for the guider (mainly
                CDELT*).  Defaults to guider_astrometry property 
            """
            # --> Don't bother checking to see if we have commanded 
            if not self.CCDCamera.GuiderRunning:
                log.error('Guider not running, move not performed')
                return False

            if guider_astrometry is None:
                guider_astrometry = self.guider_astrometry
            # --> Is this the right thing to do here?  Say no for now,
            # since I will probably deriving dra_ddec with astrometry.
            #if dec is None:
            #    try:
            #        dec = self.Telescope.Declination
            #    except:
            #        # If the user is using this apart from ACP, they
            #        # might have the scope connected through MaxIm
            #        if not self.Application.TelescopeConnected:
            #            log.warning("Could not read scope declination directly from scope or MaxIm's connection to the scope.  Using value from MaxIm Scope Dec dialog box in Guide tab of Camera Control, which the user has to enter by hand")
            #            dec = self.CCDCamera.GuiderDeclination
            # Change to rectangular tangential coordinates for small deltas
            #dra_ddec[0] = dra_ddec[0]*np.cos(np.radians(dec))

            # Get the rough RA and DEC of our current ("old") guide box
            # position.  !!! Don't forget that pixel coordinates are
            # in !!! TRANSPOSE !!!
            op_coords = (self.CCDCamera.GuiderYStarPosition,
                         self.CCDCamera.GuiderXStarPosition)
            w_coords = self.scope_wcs(op_coords,
                                      to_world=True,
                                      astrometry=guider_astrometry)
            # When moving the scope in a particular direction, the
            # stars appear to move in the opposite direction.  Since
            # we are guiding on one of those stars (or whatever), we
            # have to move the guide box in the opposite direction
            p_coords = self.scope_wcs(w_coords - dra_ddec,
                                      to_pix=True,
                                      astrometry=guider_astrometry)
            # Now we are in pixel coordinates on the guider.
            # Calculate how far we need to move.
            # There is some implicit type casting here since op_coords
            # is a tuple, but p_coords is an np.array
            dp_coords = p_coords - op_coords
            # Calculate the length in pixels of our move and the unit
            # vector in that direction
            norm_dp = np.linalg.norm(dp_coords)
            uv = dp_coords / norm_dp
            
            # Move the guide box slowly but have a threshold
            if norm_dp < self.guider_settle_tolerance:
                num_steps = 1
            else:
                # Guard against guide_box_steps_per_pix < 1 (fast moving)
                num_steps = max((1,
                                 int(self.guide_box_steps_per_pix * norm_dp)))

            step_dp = dp_coords / num_steps
            log.debug('move_with_guide_box: total guider dpix (X, Y): ' + str(dp_coords[::-1]))
            log.debug('norm_dp: ' + str(norm_dp))
            log.debug('Number of steps: ' + str(num_steps))
            log.debug('Delta per step (X, Y): ' + str(step_dp[::-1]))
            for istep in range(num_steps):
                # Just in case someone else is commanding the guide
                # box to move, use its instantaneous position as the
                # starting point of our move !!! TRANSPOSE !!!
                cp_coords = np.asarray((self.CCDCamera.GuiderYStarPosition,
                                        self.CCDCamera.GuiderXStarPosition))
                tp_coords = cp_coords + step_dp
                log.debug('Setting to: ' + str(tp_coords[::-1]))
                # !!! TRANSPOSE !!!
                self.CCDCamera.GuiderMoveStar(tp_coords[1], tp_coords[0])
                self.guider_settle()
            
            ## Give it a few extra cycles to make sure it has stuck
            ## (though even this might be too short)
            #for i in range(self.guide_box_steps_per_pix):
            #    if self.check_guiding() is False:
            #        return False
            return True
                    

        #def check_guiding(self):
        #    # --> the guider doesn't turn off when the star fades
        #    # --> This algorithm could use improvement with respect to
        #    # slowing itself down by looking at the guide errors, but
        #    # it works for now
        #    if self.guider_exptime is None:
        #        # If we didn't start the guider, take a guess at its
        #        # exposure time, since MaxIm doesn't give us that info
        #        exptime = default_guider_exptime
        #    else:
        #        exptime = self.guider_exptime
        #    # --> This needs to include the guide box read time or
        #    # else loop which uses it gets guide box position confused
        #    time.sleep(exptime*3)
        #    if self.CCDCamera.GuiderRunning:
        #        return True
        #    else:
        #        log.error('Guider stopped running while performing move')
        #        return False
            
        def MaxIm_pier_flip_state(self):
            """Return -1 if MaxIm is flipping the sense of the RA axis, 1 for non-flipped"""
            # Using GuiderAutoPierFlip or the interactive PierFlip
            # box, MaxIm can effectively cast the guider image in the
            # correct sense for the FOV of the guide camera across
            # pier flips.  Ultimately MaxIm does this by swapping the
            # RA motor commands.  This is confusing when we know our
            # absolute directions and just want to control the mount
            # through the guide ports on the camera (or other MaxIm
            # connection to the mount).  The purpose of this routine
            # is to figure out if MaxIm is flipping the sense of the
            # RA axis or not.
            flip = 1
            if not self.telescope_connectable:
                # MaxIm does not make the contents of its Pier Flip
                # check box on the Camera Control Guide tab accessible
                # to scripting, so we would have to collect that
                # information and give it to MaxIm in the form of
                # self.CCDCamera.GuiderReverseX et al.  -->
                # Eventually, we might be able to have a user
                # interface which helps with this, but for now, we
                # need to have a telescope that connects.  Don't raise
                # any warnings, since populate_obj has been wordy
                # enough
                pass
            # Here I did some experiments and found that MaxIm knows
            # what side of the pier the guider was calibrated on and
            # uses that to establish calibration in any pier flip
            # state.  Note that as per ASCOM conventions, pierEast
            # (scope on east side of pier looking west) is "normal"
            # and pierWest (scope on west side of pier looking east),
            # is "through the pole," or "flipped."  [Personally, I
            # would define pier flip to be the state I am in after I
            # cross the meridian, which is the opposite of this.]  The
            # only question is: does MaxIm know where the telescope
            # is?  Since ACP does not want MaxIm connected to the
            # telescope, this answer will vary.  Return flip = -1 when
            # MaxIm is sure to be flipping the sense of the RA axis.
            # --> Looks like I can't make this robust: When telescope
            # not connected, I have no idea if user clicked Pier Flip
            # box, so I have to assume one state or another, since
            # MaxIm WILL follow the state of that box when reversing the guider.
            if (self.alignment_mode == ASCOM.algGermanPolar
                and self.Application.TelescopeConnected
                and self.CCDCamera.GuiderAutoPierFlip
                and self.Telescope.SideOfPier == ASCOM.pierWest):
                flip = -1
            # The Camera Control->Guide->Settings->Advanced->Guider
            # Motor Control X and Y refer to connections to the
            # guider, which people are free to hook up however they
            # please to make their guiding experience pleasant (e.g.,
            # align things on their CCD).  Generally X is for RA and Y
            # for DEC, however, MaxIm doesn't really care, since it
            # just calibrates that out.  The only time MaxIm cares is
            # if you have a GEM, in which case it is nice to know
            # which axis corresponds to RA so signals along that axis
            # can be flipped on pier flip.  For non-standard wiring
            # configurations, where guider X and Y outputs are not
            # going to mount RA and DEC inputs, ASCOM_namespace.py is
            # the right place to make changes, since that effectively
            # rewires this program.  In other words, this rewiring
            # needs to take place, but this is the wrong routine to do
            # it, this routine just determines if MaxIm is reversing
            # the sense of RA for a pier flip.
            return flip

        def guider_move(self,
                        dra_ddec,
                        dec=None,
                        guider_astrometry=None):
            """Moves the telescope using guider slews.

            Parameters
            ----------
            dra_ddec : tuple-like array
            delta move in RA and DEC in DEGREES
            """
            if self.CCDCamera.GuiderRunning:
                log.warning('Guider was running, turning off')
                self.CCDCamera.GuiderStop
            log.debug('IN GUIDER_MOVE')
            # I no longer think this is the right thing to do.  We are
            # working in absolute coordinates now that we are using scope_wcs
            #if dec is None:
            #    try:
            #        dec = self.Telescope.Declination
            #    except:
            #        # If the user is using this apart from ACP, they
            #        # might have the scope connected through MaxIm
            #        if not self.Application.TelescopeConnected:
            #            log.warning("Could not read scope declination directly from scope or MaxIm's connection to the scope.  Using value from MaxIm Scope Dec dialog box in Guide tab of Camera Control, which the user has to enter by hand")
            #        dec = self.CCDCamera.GuiderDeclination
            #
            ## Change to rectangular tangential coordinates for small deltas
            #dra_ddec[0] = dra_ddec[0]*np.cos(np.radians(dec))

            # Use our rates to change to time to press E/W, N/S, where
            # E is the + RA direction
            dt = dra_ddec/self.guide_rates
            
            # Do a sanity check to make sure we are not moving too much
            max_t = (self.guider_max_move_multiplier *
                     np.asarray((self.CCDCamera.GuiderMaxMoveX, 
                                 self.CCDCamera.GuiderMaxMoveY)))
            if np.any(np.abs(dt) > max_t):
                log.warning('requested move of ' + str(dra_ddec) + ' arcsec translates into move times of ' + str(np.abs(dt)) + ' seconds.  Limiting move in one or more axes to max t of ' + str(max_t))
                dt = np.minimum(max_t, abs(dt)) * np.sign(dt)
                
            log.info('Seconds to move guider in RA and DEC: ' + str(dt))

            # Keep track of whether or not MaxIm is flipping any
            # coordinates for us and flip them back, since we know our
            # dRA and dDEC in the absolute sense.  
            dt[0] *= self.MaxIm_pier_flip_state()

            if dt[0] > 0:
                # East is positive RA, but east is left (-X) on the sky
                RA_success = self.CCDCamera.GuiderMove(ASCOM.gdMinusX, dt[0])
            elif dt[0] < 0:
                RA_success = self.CCDCamera.GuiderMove(ASCOM.gdPlusX, -dt[0])
            else:
                # No need to move
                RA_success = True
            if not RA_success:
                raise EnvironmentError('RA guide slew command failed')
            # MaxIm seems to be able to press RA and DEC buttons
            # simultaneously, but we can't!
            while self.CCDCamera.GuiderMoving:
                time.sleep(0.1)
            if dt[1] > 0:
                DEC_success = self.CCDCamera.GuiderMove(ASCOM.gdPlusY, dt[1])
            elif dt[1] < 0:
                DEC_success = self.CCDCamera.GuiderMove(ASCOM.gdMinusY, -dt[1])
            else:
                # No need to move
                DEC_success = True
            if not DEC_success:
                raise EnvironmentError('DEC guide slew command failed')
            while self.CCDCamera.GuiderMoving:
                time.sleep(0.1)
                
        def scope_wcs(self,
                      coords_in,
                      to_world=False,
                      to_pix=False,
                      astrometry=None,
                      absolute=False,
                      delta=False):
            """Computes WCS coordinate transformations to/from UNBINNED PIXELS, using scope coordinates if necessary

            Parameters
            ----------
            coords_in : tuple-like array
                (List of) coordinates to transform.  Pixel coordinates
                are in Y, X order, UNBINNED.  World coordinates are in
                RA, DEC order 
            to_world : Boolean
                perform pix to world transformation
            to_pix : Boolean
                perform world to pix transformation
            astrometry : scope name, filename, HDUList, or FITS header 
                Input method for providing an HDUList with WCS
                parameters appropriate for the CCD being used (mainly
                CDELT*).  If scope name provided ("main" or "guide"),
                the appropriate run level default file will be used.
                Can also be a FITS filename or HDUList object.
                Default: "main."  If astrometry image was taken with
                binned pixels, the header keys will be adjusted so the
                WCS transformations will be to/from unbinned pixels
            """
            coords_in = np.asarray(coords_in)
            if coords_in.shape[-1] != 2:
                raise ValueError('coordinates must be specified in pairs')
            if to_world + to_pix != 1:
                raise ValueError('Specify one of to_world or to_pix')
            # Set up our default HDUList
            if astrometry is None:
                astrometry = 'main'
            if isinstance(astrometry, str):
                if astrometry.lower() == 'main':
                    astrometry = self.main_astrometry
                elif astrometry.lower() == 'guide':
                    astrometry = self.guider_astrometry
                if not os.path.isfile(astrometry):
                    raise ValueError(astrometry + ' file not found')
                # If we made it here, we have a file to open to get
                # our astrometry from.  Opening it puts the header
                # into a dictionary we can access at any time
                astrometry = fits.open(astrometry)
                we_opened_file = True
            if isinstance(astrometry, fits.HDUList):
                header = astrometry[0].header
                we_opened_file = False
            elif isinstance(astrometry, fits.Header):
                header = astrometry
                pass
            else:
                raise ValueError('astrometry must be a string, FITS HDUList, or FITS header')
            if we_opened_file:
                astrometry.close()
            if header.get('CTYPE1') is None:
                raise ValueError('astrometry header does not contain a FITS header with valid WCS keys.')

            if not absolute:
                # NOTE: If we were passed a header, this will
                # overwrite its CRVAL keys.  But we are unlikely to be
                # passed a header in the non-absolute case
                # Connects to scope and MaxIm
                try:
                    RA = self.Telescope.RightAscension
                    DEC = self.Telescope.Declination
                except:
                    # If, for some reason the telescope doesn't report
                    # its RA and DEC, we can use the DEC reported by
                    # the user in the Scope Dec. box of the Guide tab,
                    # since DEC is really all we care about for the
                    # cosine effect in calculating deltas
                    RA = 0
                    DEC = self.CCDCamera.GuiderDeclination
                    log.warning('Telescope is not reporting RA and/or DEC.  Setting RA = ' + str(RA) + ' and DEC = ' + str(DEC) + ', which was read from the Scope Dec. box of the Guide tab.')
                # Check to see if our astrometry image was taken on
                # the other side of a GEM pier.  In that case, both RA
                # and DEC are reversed in the astrometry.  For scope
                # motion, the sense of DEC reverses when through the
                # pole, which is why only RA changes when looking at
                # guider images
                pier_flip_sign = 1
                try:
                    if (self.Telescope.AlignmentMode == ASCOM.algGermanPolar
                        and
                        ((header['PIERSIDE'] == 'EAST'
                          and self.Telescope.SideOfPier == ASCOM.pierWest
                          or
                          header['PIERSIDE'] == 'WEST'
                          and self.Telescope.SideOfPier == ASCOM.pierEast))):
                        pier_flip_sign = -1
                except:
                    log.warning('Check of telescope type or PIERSIDE FITS keyword failed.  Unable to determine if telescope is a German Equatorial in flipped state.')

                #log.info('pier_flip_sign= ' + str(pier_flip_sign))
    
                # Make sure RA is on correct axis and in the correct units
                if 'RA' in header['CTYPE1']:
                    header['CRVAL1'] = RA / 24*360 * pier_flip_sign
                    header['CRVAL2'] = DEC * pier_flip_sign
                elif 'DEC' in header['CTYPE1']:
                    header['CRVAL2'] = RA / 24*360 * pier_flip_sign
                    header['CRVAL1'] = DEC * pier_flip_sign
            # Our astrometry header now has the pointing direction we
            # want on the same side of the pier as our current
            # telecsope.  Now fix binning and subframing.  More pixels
            # to the center, but they are smaller
            header['CRPIX1'] = header['XBINNING'] * header['CRPIX1'] + header['XORGSUBF']
            header['CRPIX2'] = header['YBINNING'] * header['CRPIX2'] + header['YORGSUBF']
            header['CDELT1'] /= header['XBINNING']
            header['CDELT2'] /= header['YBINNING']
            header['CD1_1']  /= header['XBINNING']
            header['CD1_2']  /= header['YBINNING']
            header['CD2_1']  /= header['XBINNING']
            header['CD2_2']  /= header['YBINNING']
            # Put our binning and subframe to unbinned values so we
            # don't tweak things again!
            header['XORGSUBF'] = 0
            header['YORGSUBF'] = 0
            header['XBINNING'] = 1
            header['YBINNING'] = 1
            header['HISTORY'] = 'Modified CRPIX*, CD*, XORG*, and *BINNING keywords'

            # Do our desired transformations, only the WCS parts, not
            # distortions, since I haven't mucked with those parameters
            w = wcs.WCS(header)
            if to_world:
                # Our pix coords are in Y, X order.  Transpose using
                # negative striding.  Use the Ellipsis trick to get
                # to the last coordinate, which is, in a row major
                # language, where the coordinate into the pairs
                # resides (would be in the first coordinate in a
                # column major language)
                # https://stackoverflow.com/questions/12116830/numpy-slice-of-arbitrary-dimensions
                coords = coords_in[..., ::-1]
                if delta:
                    # We have left transpose space
                    c0 = np.asarray((header['CRPIX1'], header['CRPIX2']))
                    #log.debug('coords before: ' + str(coords))
                    coords += c0.astype(float)
                    #log.debug('coords after: ' + str(coords))
                # Decide to leave in RA DEC, since we are no longer in
                # our image when we are RA and DEC
                # The 0 is because we number our pixels from 0, unlike
                # FORTRAN which does so from 1

                # ACK!  The WCS package is not smart about checking
                # for single tuple input, so I have to <sigh>
                if coords.size == 2:
                    w_coords = w.wcs_pix2world(coords[0], coords[1], 0)
                else:
                    w_coords = w.wcs_pix2world(coords, 0)
                if delta:
                    w0_coords = w.wcs_pix2world(c0[0], c0[1], 0)
                    #log.debug('w_coords before: ' + str(w_coords))
                    w_coords = (np.asarray(w_coords)
                                - np.asarray(w0_coords))
                    #log.debug('w_coords after: ' + str(w_coords))
                    # for debugging purposes
                    #coords -= c0
                    #log.debug('coords_in[..., ::-1]: ' + str(coords))
                    #log.debug('plate scale: ' +
                    #          str(3600*w_coords/coords))
                return w_coords
            if to_pix:
                # --> This might need to be fixed
                if delta:
                    coords_in += np.asarray((header['CRVAL1'],
                                             header['CRVAL2']))
                if coords_in.size == 2:
                    pix = np.asarray(
                        w.wcs_world2pix(coords_in[0], coords_in[1], 0))
                else:
                    pix = w.wcs_world2pix(coords_in, 0)
                if delta:
                    # Note we have yet to leave transpose space
                    pix -= np.asarray((header['CRPIX1'], header['CRPIX2']))
                # Put out pix back into Y, X order, UNBINNED
                return pix[..., ::-1]

        def rot(self, vec, theta):
            """Rotates vector counterclockwise by theta degrees"""
            np.asarray(vec)
            theta = np.radians(theta)
            c, s = np.cos(theta), np.sin(theta)
            M = np.matrix([[c, -s], [s, c]])
            rotated = np.asarray(np.dot(M, vec))
            return np.squeeze(rotated)
    
        def get_keys(self):
            """Gets list of self.required_FITS_keys from current image"""
            self.FITS_keys = []
            for k in self.required_FITS_keys:
                self.FITS_keys.append((k, self.Document.GetFITSKey(k)))
            
        def set_keys(self, keylist):
            """Write desired keys to current image FITS header"""
            self.getDocument()
            if self.HDUList is None:
                log.warning('Asked to set_keys, but no HDUList is empty')
                return None
            try:
                h = self.HDUList[0].header
                for k in keylist:
                    if h.get(k):
                        # Not sure how to get documentation part written
                        self.Document.SetFITSKey(k, h[k])
            except:
                log.warning('Problem setting keys: ', sys.exc_info()[0])
                return None

        # This will eventually record and analyze guider images and
        # determine the best exposure time to use --> consider
        # combining all image recording to take_im with a camera
        # provided
        def get_guider_exposure(self, exptime=None, filter=None):
            """Returns tuple (exptime, star_auto_selected) since
            taking an image with GuiderAutoSelectStar will select the
            star"""
            if filter is not None:
                try:
                    # --> Do some checking of length of filter, or
                    # --> maybe have a routine that cycles through the
                    # --> guider filter list, since this will bomb
                    # --> with a filter list right now
                    self.CCDCamera.GuiderFilter = filter
                except:
                    raise EnvironmentError('error setting filter to ' + str(filter) + '.  Are you using a valid filter integer?  Is the filter wheel set up for the guider?')
            if exptime is None:
                #log.debug('Code not written yet to get auto exposure, just using default_guider_exptime')
                exptime = default_guider_exptime
                # Here we would do some exposing to figure out what the optimal 
            # --> Eventually write the code that will take the image
            # and figure out what filter to use
            return (exptime, False)

        # This is going to need to take a guider picture
        def set_guider_star_position(self):
            raise ValueError('Code not written yet.  Use GuiderAutoSelectStar for now')

        def guider_start(self, exptime=None, filter=None, star_position=None):
            """Start guider

            Parameters
            ----------
            exptime : float or None
            Exposure time to use

            """
            # --> set declination from scope
            if (self.CCDCamera.GuiderRunning
                and self.guider_commanded_running):
                return
            if (self.CCDCamera.GuiderRunning
                and not self.guider_commanded_running):
                log.warning('Guider was running, restarting')
                # --> May or may not want to propagate existing
                # --> exposure time stuff
                self.guider_stop()

            # GuiderAutoSelectStar is something we set for scripts to
            # have Maxim do the star selection for us
            if star_position is None:
                self.CCDCamera.GuiderAutoSelectStar = True
            else:
                self.CCDCamera.GuiderAutoSelectStar = False
            self.guider_exptime, auto_star_selected \
                = self.get_guider_exposure(exptime=exptime,
                                           filter=filter)
            if not auto_star_selected and star_position is None:
                # Take an exposure to get MaxIm to calculate the guide
                # star postion
                self.CCDCamera.GuiderExpose(self.guider_exptime)
                # --> Consider checking for timout here
                while self.CCDCamera.GuiderRunning:
                    time.sleep(0.1)
            if not self.CCDCamera.GuiderTrack(self.guider_exptime):
                raise EnvironmentError('Attempt to start guiding failed.  Guider configured correctly?')
            # MaxIm rounds pixel center value to the nearest pixel,
            # which can lead to some initial motion in the guider
            self.guider_settle()
            self.guider_commanded_running = True

        def guider_stop(self):
            self.guider_commanded_running = False
            return self.CCDCamera.GuiderStop

        def get_im(self):
            """Puts current MaxIm image (the image with focus) into a FITS HDUList.  If an exposure is being taken or there is no image, the im array is set equal to None"""
            # Clear out HDUList in case we fail
            self.HDUList = None
            if not self.CCDCamera.ImageReady:
                raise EnvironmentError('CCD Camera image is not ready')
            # For some reason, we can't get at the image array or its FITS
            # header through CCDCamera.ImageArray, but we can through
            # Document.ImageArray
            self.getDocument()
            
            # Make sure we have an array to work with
            c_im = self.Document.ImageArray
            if c_im is None:
                raise EnvironmentError('There is no image array')
            # Create a basic FITS image out of this and copy in the FITS
            # keywords we want
    
            # TRANSPOSE ALERT.  Document.ImageArray returns a tuple of
            # tuples shaped just how you would want it for X, Y.  Since
            # Python is written in C, this is stored in memory in in "C
            # order," which is the transpose of how they were intended to
            # be written into a FITS file.  Since all the FITS stuff
            # assumes that we are reading/writing FORTRAN-ordered arrays
            # bytes from/to a C language, we need to transpose our array
            # here so that the FITS stuff has the bytes in the order it
            # expects.  This seems less prone to generating bugs than
            # making users remember what state of transpose they are in
            # when dealing with arrays generated here vs. data read in
            # from disk for debugging routines.  This is also faster than
            # writing to disk and re-reading, since the ndarray order='F'
            # doesn't actually do any movement of data in memory, it just
            # tells numpy how to interpret the order of indices.
            c_im = np.asarray(c_im)
            adata = c_im.flatten()#order='K')# already in C order in memory
            # The [::-1] reverses the indices
            adata = np.ndarray(shape=c_im.shape[::-1],
                               buffer=adata, order='F')
            
            hdu = fits.PrimaryHDU(adata)
            self.get_keys()
            for k in self.FITS_keys:
                hdu.header[k[0]] = k[1]
            self.HDUList = fits.HDUList(hdu)
            return self.HDUList

        # Take im is Bob Denny's nomenclature
        # --> work out all of the binning, subarray, etc. stuff and
        # propagate downstream
        def take_im(self,
                    exptime=None,
                    filt=None,
                    binning=None,
                    camera=None,
                    subarray=None,
                    light=None):
            """Uses MaxIm to record an image
            """
            if exptime is None:
                exptime = default_exptime
            if filt is None:
                filt = self.default_filt
            # Set the filter separately, since some filter wheels need
            # time to rotate
            if self.CCDCamera.Filter != filt:
                self.CCDCamera.Filter = filt
                time.sleep(self.main_filt_change_time)
            # --> Add support for binning, camera, and non-light
            # --> Add support for a pause for the filter wheel
            # Take a light (1) exposure
            self.CCDCamera.Expose(exptime, 1, filt)
            # This is potentially a place for a coroutine and/or events
            time.sleep(exptime)
            # --> Need to set some sort of timeout
            while not self.CCDCamera.ImageReady:
                time.sleep(0.1)
            return(self.get_im())

        def acquire_im(self,
                       fname=None,
                       **kwargs):
            if (not isinstance(fname, str)
                or os.path.isfile(fname)):
                raise ValueError('Specify a valid non-existent file to save the image to')
            HDUList = self.take_im(**kwargs)
            if not self.CCDCamera.SaveImage(fname):
                raise EnvironmentError('Failed to save file ' + fname)
            log.debug('Saved file: ' + fname)
            return HDUList


    class PrecisionGuide():
        """Class containing PrecisionGuide package

    Parameters
    ----------
    MD : MaxImData
        MaxImData object set up with defaults you would like to use
        for guiding, main camera exposure, etc.  Default: MaxImData
        set to defaults of that object

    ObsClassName : str
        (Sub)class name of ObsData which will contain code that calculates 
        obj_center and desired_center coordinates.  Default: ObsData
    
    ObsClassModule : str
        Module (.py file) containing ObsClass definition.  
        Default: current file

    guide_box_command_file : str
        Filename used to send info to GuideBoxMover

    guide_box_log_file : str
        Log file for GuideBoxCommander

    **ObsClassArgs are passed to ObsClassName when instantiated
        """
        def __init__(
                self,
                ObsClassName=None, 
                ObsClassModule=None,
                MD=None,
                guide_box_command_file=default_guide_box_command_file,
                guide_box_log_file=default_guide_box_log_file,
                **ObsClassArgs): # args to use to instantiate ObsClassName
            if ObsClassName is None:
                # Default to plain ObsData
                ObsClassName='ObsData'
            if ObsClassModule is None:
                # Default to finding ObsData subclass in current file
                # https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
                # We are in the same file, so we just want to use the
                # dictionary method of getting the class as a value
                self.ObsDataClass = globals()[ObsClassName]
            else:
                # https://stackoverflow.com/questions/4821104/python-dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported
                # Windows adds the full path as in
                # C:\\asdiasodj\\asdads\\etc LINUX does not, but they
                # both add the .py, which importlib does not want
                # --> I could potentially use some os thing to do the
                # split, but I know I am on Windows at this point
                ObsClassModule = __file__.split('\\')[-1]
                ObsClassModule = ObsClassModule.split('.py')[0]
                self.ObsDataClass \
                    = getattr(importlib.import_module(ObsClassModule),
                              ObsClassName)
            self.ObsClassArgs = ObsClassArgs
            if MD is None:
                self.MD = MaxImData()
            self.guide_box_command_file = guide_box_command_file
            self.guide_box_log_file = guide_box_log_file

            self.center_tolerance = default_cent_tol

            self.ObsDataList = []
            # Number of median exposure times
            self.time_weighting = 5
            # Might not use the next few
            # Make this be able to bring the object back to the center
            self.flex_aggressiveness = 1
            # Used for spotting unusually high flex_pix rates
            self.flex_pix_std_mult = 5
            # These need to be float.  I technically have two things:
            # motion of the object and distance from the center.  -->
            # for now current_flex_pix_rate is the total and
            # current_centering_rate is the eponymous component
            self.current_flex_pix_rate = np.zeros(2)
            self.current_centering_rate = np.zeros(2)
            self.current_flex_pix_TStart = Time.now()
            ### --> See if I can't fix this another way
            ## Don't assume we are running this live, otherwise now()
            ## results in a backward time delta
            ##self.current_flex_pix_TStart = Time('1990-01-01T00:00', format='fits')
            # --> might not use this
            self.last_delta_pix = None
            # --> Current way to keep the guide box moving, may improve
            self._GuideBoxMoverSubprocess = None
            
        def reinitialize(self,
                         keep_flex_pix_rate=False):
            """Resets system for new sequence of observations.  If keep_flex_pix_rate is True, which would be approprite for observations on the same general area of the sky. """ 
            current_motion_rate = (self.current_flex_pix_rate -
                                   self.current_centering_rate)
            self.current_centering_rate = np.zeros(2)
            if not keep_flex_pix_rate:
                self.GuideBoxMoving = False
                # --> consider doing all the calculated variables too
                self.ObsDataList = []
                self.current_flex_pix_rate = np.zeros(2)
                self.current_flex_pix_TStart = Time.now()

        def create_ObsData(self, arg, **ObsClassArgs):
            if ObsClassArgs != {}:
                return self.ObsDataClass(arg, **ObsClassArgs)
            elif self.ObsClassArgs != {}:
                return self.ObsDataClass(arg, **self.ObsClassArgs)
            else:
                return self.ObsDataClass(arg)

        @property
        def GuideBoxMoving(self):
            if self._GuideBoxMoverSubprocess is None:
                return False
            assert isinstance(self._GuideBoxMoverSubprocess, subprocess.Popen)
            return True

        @GuideBoxMoving.setter
        def GuideBoxMoving(self, value=None):
            if not isinstance(value, bool):
                raise ValueError('Set GuideBoxMoving to True or False to start or stop GuideBoxMover')
            if value and self._GuideBoxMoverSubprocess is None:
                # --> I may need a better path to this
                self._GuideBoxMoverSubprocess \
                    = subprocess.Popen(['python',
                                        'ioio.py',
                                        'GuideBoxMover',
                                        self.guide_box_command_file])
                # Make sure we are up and running before moving on
                # --> Needs a timeout
                while self._GuideBoxMoverSubprocess is None:
                    time.sleep(0.2)
                    
                if not os.path.isfile(self.guide_box_log_file):
                    # Pretty print a header for our rates log file
                    # --> put more stuff in here!
                    with open(self.guide_box_log_file, 'w') as l:
                        l.write('Time                        dra, ddec rates (arcsec/hour)\n')
                with open(self.guide_box_log_file, 'a') as l:
                    l.write(Time.now().fits
                            + '---- GuideBoxMover Started ----- \n')
            if not value and not self._GuideBoxMoverSubprocess is None:
                self._GuideBoxMoverSubprocess.terminate()
                with open(self.guide_box_log_file, 'a') as l:
                    l.write(Time.now().fits
                            + '---- GuideBoxMover Stopped ----- \n')

        def GuideBoxCommander(self, pix_rate=np.zeros(2)):
            """Input desired telescope motion rate in main camera pixels per sec, outputs command file for GuideBoxMover in degrees per second"""
            # We are the appropriate place to keep track of the
            # up-to-the minute current pix_rate and
            # current_flex_pix_TStart (see below)
            self.current_flex_pix_rate = pix_rate
            # Convert main camera pix_rate to dra_ddec_rate.  Start by
            # making a vector 10 pixels long.  The conversion factor
            # is effectively the time required for a move of 10 pixels

            log.debug('IN GuideBoxCommander')
            n_pix_rate = np.linalg.norm(pix_rate)
            if n_pix_rate < 1000 * np.finfo(np.float).eps:
                # Avoid zero divide
                # This needs to be float
                dra_ddec_rate = np.zeros(2)
            else:
                # Calculate time it would take to move 10 pixels
                t_move = 10 / n_pix_rate
                dpix = pix_rate * t_move
                dra_ddec \
                    = self.MD.scope_wcs(dpix,
                                        to_world=True,
                                        delta=True,
                                        astrometry=self.MD.main_astrometry)
                dra_ddec_rate = dra_ddec / t_move
            # Convert from degrees/s to arcsec/hour degrees/s *3600
            # arcsec/degree * 3600 s/hr
            dra_ddec_rate *= 3600.**2
            rates_list = dra_ddec_rate.tolist()
            json.dump(rates_list,
                      open(self.guide_box_command_file, 'w'),
                      separators=(',', ':'),
                      sort_keys=True,
                      indent=4)
            # Make sure our GuideBoxMover subprocess is running
            # --> name could be a little better, possibly
            self.GuideBoxMoving = True
            # --> Eventually get a handshake with the precise time we
            # started the new rate
            self.current_flex_pix_TStart = Time.now()

            log.info('Guide box rate: ' +
                     self.current_flex_pix_TStart.fits + ' ' + str(rates_list))
            with open(self.guide_box_log_file, 'a') as l:
                # --> put more stuff in here!  Like HA, DEC, etc. to
                # --> be able to make model from log file.  Consider
                # --> dividing dra by DEC so rates are consistent with
                # --> HORIZONs
                l.write(self.current_flex_pix_TStart.fits + ' '
                        + str(rates_list[::-1]) + '\n')
            return True

        # --> I'll probably want a bunch of parameters for the exposure
        def center(self,
                   HDUList_im_fname_ObsData_or_obj_center=None,
                   desired_center=None,
                   current_astrometry=None,
                   scaling_astrometry=None,
                   ignore_ObsData_astrometry=False,
                   **ObsClassArgs):
            """Move the object to desired_center using guider slews OR
                   guide box moves, if the guider is running.  Takes
                   an image with default  filter and exposure time if
                   necessary
            Parameters
            ----------
            HDUList_im_fname_ObsData_or_obj_center : see name for types

                Specifies default center in some way.  If its and
                HDUList, image, or fname, the ObsData registered with
                PrecisionGuide will be used to derive the current
                object center and desired center.  Default = None,
                which means an image will be recorded and used for the
                ObsData.  If the ObsData calculates absolute
                astrometry, that will end up in its ObsData.header and
                will be used to calculate guider slews.  To ignore the
                astrometry in the ObsData, set
                ignore_ObsData_astrometry=True

            current_astrometry : HDUList or str
                FITS HDUList or file name from which one can be read
                that contains astrometric solution *for the current
                telescope position*

            scaling_astrometry : HDUList or str
                FITS HDUList or file from which one can be read that
                contains astrometric solution for the relevant
                telescope for the purposes of pixel to WCS scaling.
                Actual pointing direction will be taken from telescope
                position or MaxIm guider DEC dialog box

            ignore_ObsData_astrometry : boolean
                Do not use astrometry in ObsData FITS header even if
                present.  Default: False

            """
            # save some typing
            input = HDUList_im_fname_ObsData_or_obj_center
            if input is None:
                # Take an image with the default exposure time and filter
                input = self.MD.take_im()
            try:
                # Check for a simple coordinate pair, which may have
                # been passed in as a tuple or list.  If this is some
                # other input, the exception will pass on through to
                # the other code
                coord = np.asarray(input)
                # But we have to differentiate between this and a full
                # image as ndarray, so throw an intentional error
                assert coord.size == 2
                # If we made it here, we have just a coordinate for
                # our object center.  Set input to None to avoid
                # re-checking for ndarray
                obj_center = coord
                input = None
                # All other cases should provide us a desired_center
                if desired_center is None:
                    log.warning('desired_center not specified.  Using the currently displayed CCD image center')
                    # If these statements bomb, the lack of
                    # desired_center will be caught below
                    self.MD.connect()
                    desired_center \
                        = np.asarray((self.MD.CCDCamera.StartY
                                      + self.MD.CCDCamera.NumY, 
                                      self.MD.CCDCamera.StartX
                                      + self.MD.CCDCamera.NumX)) / 2.
            except:
                pass
            if (isinstance(input, fits.HDUList)
                or isinstance(input, np.ndarray)
                or isinstance(input, str)):
                # The ObsClass base class takes care of reading all of these
                input = self.create_ObsData(input, **ObsClassArgs)
            if isinstance(input, ObsData):
                if input.quality < 5:
                    log.error('Quality of center estimate too low: ' + str(input.quality))
                    return False
                obj_center = input.obj_center
                if desired_center is None:
                    # (Allows user to override desired center)
                    desired_center = input.desired_center
                if current_astrometry is not None:
                    astrometry_from = current_astrometry
                    absolute = True
                elif (input.header.get('CTYPE1')
                      and ignore_ObsData_astrometry == False):
                    astrometry_from = input.header
                    absolute = True
                elif scaling_astrometry is not None:
                    astrometry_from = scaling_astrometry
                    absolute = False
                else:
                    # Default will be determined in scope_wcs
                    astrometry_from = None
                    absolute = False

            if obj_center is None or desired_center is None:
                raise ValueError('Invalid HDUList_im_fname_ObsData_or_obj_center or a problem establishing desired_center from current CCD image (or something else...)')
            
            #log.debug('pixel coordinates (X, Y) of obj_center and desired_center: ' + repr((obj_center[::-1], desired_center[::-1])))
            w_coords = self.MD.scope_wcs((obj_center, desired_center),
                                         to_world=True,
                                         astrometry=astrometry_from,
                                         absolute=absolute)
            #log.debug('world coordinates of obj_center and desired_center: ' + repr(w_coords))

            dw_coords = w_coords[1,:] - w_coords[0,:]
            # --> This would be the right place to check if the dw_coords
            # are too large for a practical move_with_guide_box
            if self.MD.CCDCamera.GuiderRunning:
                log.debug('CENTERING TARGET WITH GUIDEBOX MOVES')
                self.MD.move_with_guide_box(dw_coords)
            else:
                log.debug('CENTERING TARGET WITH GUIDER SLEWS')
                self.MD.guider_move(dw_coords)

        def center_loop(self,
                        exptime=None,
                        filt=None,
                        tolerance=None,
                        max_tries=3,
                        start_PrecisionGuide=False,
                        **ObsClassArgs):
            """Loop max_tries times taking exposures and moving the telescope with guider slews or, if the guider is on, guide box moves to center the object
            """
            if tolerance is None:
                tolerance = self.center_tolerance
            tries = 0
            fails = 0
            while True:
                log.debug('CENTER_LOOP TAKING EXPOSURE')
                HDUList = self.MD.take_im(exptime, filt)
                O = self.create_ObsData(HDUList, **ObsClassArgs)
                if O.quality < 5:
                    log.warning('obj_center quality estimate is too low: ' +
                                str(O.quality))
                    fails += 1
                    if fails > max_tries:
                        log.error(
                            'Maximum number of tries exceeded to get a good center: '
                            + str(fails))
                        return False
                    continue
                if (np.linalg.norm(O.obj_center - O.desired_center)
                    < tolerance):
                    log.debug('Target centered to ' + str(tolerance) +
                              ' pixels')
                    if start_PrecisionGuide:
                        # We have moved the telescope, so our Obs
                        self.reinitialize()
                        self.update_flex_pix_rate(0)
                    return True
                if tries >= max_tries:
                    log.error('Failed to center target to ' +
                              str(tolerance) + ' pixels after '
                              + str(tries) + ' tries')
                    return False
                self.center(O)
                tries += 1

        def pix_rate_to_freeze_motion(self):
            # We have to keep track of the position of our object on a
            # per-filter basis because the filters are unlikely to be
            # perfectly oriented in the same way.  Slight tips in the
            # filter lead to refractive motion.  In the IoIO
            # coronagraph, the field lens focuses the pupil of the
            # telescope onto the camera lens.  When it moves/tips
            # (which is equivalent to tiping the filter), the pupil
            # moves, moving the apparent image of everything in the
            # focal plane.  This is a non-trivial effect because of
            # the length of the instrument.  The ND filter, on the
            # other hand, is not imaged by the field lens onto the
            # camera lens.  It is close enough to basically be part of
            # it.  So movement of the field lens does not effect its
            # apparent position in the focal plane, at least that I
            # have noticed.  It does, however, move as the instrument
            # swings on the telescope.  So we have to keep track of
            # the position of the desired center and motion torward it
            # separtely from the object center.

            # NOTE: All calculations are done in main camera pixel
            # coordinates until we want to move the scope.  When we
            # speak of delta pixels, we mean how many pixels to move
            # our object (obj_center) to get to the place we want it
            # to be (desired_center).

            log.debug('STARTING FLEX_PIX_RATE MOTION CALCULATIONS')

            current_motion_rate = (self.current_flex_pix_rate -
                                   self.current_centering_rate)
            D.say('current_motion_rate ' + str(current_motion_rate))
            # We can only track motion with individual filters
            this_filt = self.ObsDataList[-1].header['FILTER']
            i_this_filt_list = [i for i, O in enumerate(self.ObsDataList)
                                if O.header['FILTER'] == this_filt]
            if len(i_this_filt_list) == 1:
                log.debug('Only one filter of this type measured so far, not doing motion calculations')
                return current_motion_rate

            # Use our stored calculations from update_flex_pix_rate to
            # get the positions of the object, had rate corrections not
            # been done.  FOR OBSERVATIONS THROUGH THIS FILTER ONLY
            effective_obj_centers = (self.obj_centers[i_this_filt_list]
                                     + self.total_flex_dpix[i_this_filt_list])
            #D.say('effective_obj_centers ' + str(effective_obj_centers))
            # Check to see if we have a significant measurement
            depix = effective_obj_centers[-1] - effective_obj_centers
            ndepix = np.asarray([np.linalg.norm(d) for d in depix])
            #D.say('ndepix ' + str(ndepix))
            # Assume Gaussian type errors
            if (np.max(ndepix)
                < np.median(self.obj_center_errs[i_this_filt_list]) * 3):
                log.debug('No significant movement detected, not doing motion calculations')
                return current_motion_rate

            # If we made it here, we have a rate measurement worth
            # making.  Establish our time axis for the linear fit to
            # effective_ob_centers
            d_last_t = np.asarray(
                [(self.Tmidpoints[-1] - D0).sec
                 for D0 in self.Tmidpoints[i_this_filt_list]])
            #D.say('d_last_t ' + str(d_last_t))
            # Next up is establishing our weights.  Basic weight is 1/err
            w = 1/self.obj_center_errs[i_this_filt_list]
            if len(w) > 1:
                # We also want to weight based on how many changes
                # there have been in the flex_pix_rate.
                # return_inverse is a handy return value, since it
                # increments for each unique value in the original
                # array.  We want our last one to have value 1 and go
                # up from there
                u, iidx = np.unique(self.TRateChanges[i_this_filt_list],
                                    return_inverse=True)
                # Working with two separate weights is a bit of a pain
                w /= iidx[-1] + 1 - np.transpose(
                    np.broadcast_to(iidx, (2, len(iidx))))
                # Our final weighting is in time, for which we use the
                # median time between exposures as our basic ruler.
                # Dilute the time weighting factor so that we can have
                # multiple exposures (default 5) before equaling the
                # weight decrement of one rate change --> consider
                # better name for time_weighting
                DTs = (self.Tmidpoints[i_this_filt_list[1:]]
                       - self.Tmidpoints[i_this_filt_list[0:-1]])
                dts = np.asarray([DT.sec for DT in DTs])
                dts = np.transpose(
                    np.broadcast_to(dts, (2, len(dts))))
                # Working with two separate weights is a bit of a pain
                d_last_t2 = np.transpose(
                    np.broadcast_to(d_last_t, (2, len(d_last_t))))
                w /= (d_last_t2 / np.median(dts) + 1) / self.time_weighting
            #D.say('w ' + str(w))
            # --> need a try here?
            # weights can't be specified on a per-fit basis
            # !!! TRANSPOSE !!!
            ycoefs = np.polyfit(d_last_t, effective_obj_centers[:,1], 1,
                                w=w[:,1])
            xcoefs = np.polyfit(d_last_t, effective_obj_centers[:,0], 1,
                                w=w[:,0])
            slopes = np.asarray((ycoefs[0], xcoefs[0]))
            # The slopes are dpix/dt of measured object motion on the
            # main camera.  We want that motion to stop, so we want to
            # apply the negative of that to telescope motion
            new_rate = -1 * slopes * self.flex_aggressiveness
            D.say('NEW RATE before filter: ' + str(new_rate))
            # See if that new rate would result in a significantly
            # different current position over our average exposure
            if np.any(np.abs(new_rate - current_motion_rate)
                      * self.average_exptime > 5 * self.obj_center_errs):
                log.debug('NEW MOTION RATE (main camera pix/s): ' +
                      str(new_rate))
                return new_rate
            log.debug('NO SIGNIFICANT CHANGE IN MOTION')
            return current_motion_rate


            #### O still points to the last object so we can conveniently
            #### fill it in some more.  Recall we are dealing iwth FITS
            #### time objects.
            ###dt = (O.Tmidpoint - self.ObsDataList[-2].Tmidpoint).sec
            ###O.total_flex_dpix = O.flex_pix_rate * dt
            ###
            #### Do our motion canceling calculations
            ###
            ###
            #### Create a list of vectors representing the amount of
            #### guide box motion (in degrees) between each of past
            #### measurements
            ###for O in self.ObsDataList:
            ###    # --> Eventually make this the actual time the
            ###    # --> rate changed in GuideBoxMover
            ###    start_t = np.max(
            ###        (OThisFiltList[-2].Tmidpoint, O.TRateChange))
            ###    # FITS times are in JD, not seconds
            ###    Odpix = (O.flex_pix_rate * u.pix/u.s * (end_t - start_t) 
            ###             + O.delta_pix * u.pix)
            ###    Odpix.decompose()
            ###    log.debug('Odpix: ' + str(Odpix))
            ###    dpix_other_filt += Odpix.value
            ###    end_t = start_t
            ###    #if O == OThisFiltList[-2]:
            ###    # Go back to the very beginning
            ###    if O == OThisFiltList[-2]:
            ###        # When we get back to the first measurement
            ###        # through our filter, we don't include its delta_pix
            ###        dpix_other_filt -= O.delta_pix
            ###        log.debug('dpix_other_filt: ' + str(dpix_other_filt))
            ###        break
            ###
            ###
            ###
            #### --> Not using this yet, but this is the logic that would
            #### --> get the delta_pix into each O in ObsDataList
            ###if self.last_delta_pix is None:
            ###    O.delta_pix = np.zeros(2)
            ###else:
            ###    O.delta_pix = self.last_delta_pix
            ###    self.last_delta_pix = None
            ###
            ###self.ObsDataList.append(O)
            ###if len(self.ObsDataList) == 1:
            ###    log.debug('STARTING GuideBoxCommander OR RECYCLING LIST due to large move')
            ###    return self.GuideBoxCommander(self.current_flex_pix_rate)
            ###
            ###this_filt = self.ObsDataList[-1].header['FILTER']
            ###OThisFiltList = [O for O in self.ObsDataList
            ###                 if O.header['FILTER'] == this_filt]
            ###if len(OThisFiltList) == 1:
            ###    # Start the system self.current_flex_pix_rate should be (0,0)
            ###    log.debug('(old) FIRST CALL TO GuideBoxCommander')
            ###    return self.GuideBoxCommander(self.current_flex_pix_rate)
            ### Create a sequence of measurements since our last
            ### rate change
            ### The following is equivalent to something like this:
            ###obj_centers = []
            ###obj_center_errs = []
            ###Tmidpoints = []
            ###for O in OThisFiltList:
            ###    if O.flex_pix_rate == self.current_flex_pix_rate:
            ###        obj_centers.append(O.obj_center)
            ###        obj_center_errs.append(O.obj_center_err)
            ###        Tmidpoints.append(O.Tmidpoints)
            ### and then the np.asarray conversions
            ##measListOLists = [[O.obj_center,
            ##                   O.obj_center_err,
            ##                   O.Tmidpoint,
            ##                   O.header['EXPTIME']]
            ##                  for O in OThisFiltList
            ##                  if np.all(O.flex_pix_rate
            ##                            == self.current_flex_pix_rate)]
            ### The * unpacks the top level ListOLists (one per O
            ### matching current_flex_pix_rate) to provide zip with a
            ### bunch of lists that have 3 elements each.  Zip then
            ### returns a list of 3 tuples, where the tuples have the
            ### long list of elements we want
            ##measTuples = list(zip(*measListOLists))
            ##obj_centers = np.asarray(measTuples[0])
            ##obj_center_errs = np.asarray(measTuples[1])
            ##Tmidpoints = np.asarray(measTuples[2])
            ##exptimes = np.asarray(measTuples[3])
            ### Estimate effect of seeing on short exposures --> for now
            ### just call seeing 2 pixels.  Eventually we want the
            ### formal seeing in arcsec, measured in real time through a
            ### verity of means
            ##seeing_pix = 2
            ##seeing_freq = 1 # hz, upper limit on frequency of detectable variations
            ##seeing_err = seeing_pix/(1/seeing_freq + exptimes)
            ### Match the shape of our obj_center_errs, which has a list
            ### of coordinates (N, 2).  For broadcast, the last element
            ### of the shape needs to be the length of our original
            ### array.  But that ends up being the transpose of for our
            ### obj_center_errs, hence the swap
            ##seeing_err = np.transpose(
            ##    np.broadcast_to(seeing_err, (2, len(seeing_err))))
            ##obj_center_errs = (obj_center_errs +
            ##                   seeing_err**2)**0.5
            ### --> Might want to come up with another weighting factor
            ### --> to emphasize older data.  But for now I am doing
            ### --> that by just including data since last rate change
            ##
            ### Determine if we have a significant measurement
            ##if np.all(np.abs(obj_centers[-1] - obj_centers)
            ##          <= 3 * obj_center_errs):
            ##    # --> This assumes we will be doing recentering separately
            ##    log.debug('Not enough motion to calculate a reliable flex pix rate')
            ##else:
            ##    log.debug('CALCULATING NEW FLEX PIX RATE TO SEE IF IT IS LARGE ENOUGH TO WARRANT CHANGE')
            ##    # Convert Tmidpoints, which are astropy.fits Time
            ##    # objects with time values in JD into time deltas in
            ##    # seconds.  Have to do the work one at a time, since
            ##    # the time delta doesn't work for ndarrays
            ##    dts = [(T - Tmidpoints[0]).sec for T in Tmidpoints]
            ##    #print(dts, obj_centers, obj_center_errs)
            ##    # --> need a try here?
            ##    # weights can't be specified on a per-fit basis
            ##    # !!! TRANSPOSE !!!
            ##    ycoefs = np.polyfit(dts, obj_centers[:,1], 1,
            ##                        w=obj_center_errs[:,1])
            ##    xcoefs = np.polyfit(dts, obj_centers[:,0], 1,
            ##                        w=obj_center_errs[:,0])
            ##    slopes = np.asarray((ycoefs[0], xcoefs[0]))
            ##    # The slopes are dpix/dt of measured object motion on the
            ##    # main camera.  We want that motion to stop, so we want to
            ##    # apply the negative of that to telescope motion
            ##    new_rate = -1 * slopes * self.flex_aggressiveness
            ##    # See if that new rate would result in a significantly
            ##    # different current position over our average exposure
            ##    if np.any(np.abs(new_rate - self.current_flex_pix_rate)
            ##              * self.average_exptime > 5 * obj_center_errs):            
            ##        log.debug('RATE CHANGE, CALLING GuideBoxCommander')
            ##        self.current_flex_pix_rate = new_rate
            ##
            ### The above stops our object from moving.  Now we want to
            ### get it back into the center.  
            ##dpix = (self.ObsDataList[-1].obj_center
            ##        - self.ObsDataList[-1].desired_center)
            ### Check to see if we have a significant measurement
            ##if np.all(np.abs(dpix) <= 3 * obj_center_errs):
            ##    log.debug('Not enough motion to calculate a reliable recentering rate')
            ##else:
            ##    log.debug('CALCULATING NEW CENTERING RATE TO SEE IF IT IS LARGE ENOUGH TO WARRANT CHANGE')
            ##
            ##    # Pick a time scale that doesn't move our object too much
            ##    # during an exposure --> Also consider doing this for
            ##    # average deltaT between exposures.  The idea is we don't
            ##    # want to move too fast
            ##    self.current_centering_rate = -dpix/self.average_exptime
            ##self.GuideBoxCommander(self.current_flex_pix_rate
            ##                       + self.current_centering_rate)
            ##return True
            
            #### See if our new rate would result in a significantly
            #### different current position over the length of time we
            #### have been correcting at that current rate
            ####dt_current_rate = (Tmidpoints[-1] - 
            ####                   self.current_flex_pix_TStart).sec
            ###dt_current_rate = (Tmidpoints[-1] - Tmidpoints[0]).sec
            ###log.debug('TStart: ' + str(self.current_flex_pix_TStart))
            ###log.debug('Tmidpoints[0]: ' + str(Tmidpoints[0]))
            ###log.debug('dt_current_rate: ' + str(dt_current_rate))
            ###total_corrected = self.current_flex_pix_rate * dt_current_rate
            ###new_corrected = new_rate * dt_current_rate
            ###log.debug('total pixels at current rate: ' +
            ###          str(total_corrected))
            ###log.debug('total pixels at new rate: ' + str(new_corrected))
            #### Do comparison one axis at a time in case user wants to
            #### specify them differently (e.g. RA, DEC mechanical or
            #### target sensitivity)
            ####if np.any(np.abs(total_corrected - new_corrected)
            ####          > self.center_tolerance):
            ###if np.any(np.abs(total_corrected - new_corrected)
            ###          > np.asarray((1,1))):
            ###    log.debug('RATE CHANGE, CALLING GuideBoxCommander')
            ###    self.GuideBoxCommander(new_rate)
            ###
            #### Now check to see if we are too far away from the center
            ###dpix = (self.ObsDataList[-1].obj_center
            ###        - self.ObsDataList[-1].desired_center)
            ####log.debug('IN calc_flex_pix_rate: DPIX FROM CENTER: ' + str(dpix))
            ###log.debug(str(np.abs(dpix)
            ###              > self.ObsDataList[-1].desired_center_tolerance))
            ###if np.any(np.abs(dpix)
            ###          > self.ObsDataList[-1].desired_center_tolerance):
            ###    dra_ddec = self.MD.scope_wcs(dpix,
            ###                                 to_world=True,
            ###                                 delta=True,
            ###                                 astrometry=self.MD.main_astrometry)
            ###    log.warning(
            ###        'calc_flex_pix_rate: Too far from center, moving dra_ddec: '
            ###        + str(dra_ddec))
            ###    self.MD.move_with_guide_box(dra_ddec)
            ###    # Save our dpix to be saved in the ObsData we are
            ###    # currently preparing for
            ###    self.last_delta_pix = dpix
            ###    # Moving guidebox upsets our careful rate measurement,
            ###    # so to be fair, reset the TStart to be after we have
            ###    # settled
            ###    self.current_flex_pix_TStart = Time.now()
            ###    # --> I don't have logic to stop the rate accumulation
            ###    # --> at the last dpix, so just erase the ObjDataList
            ###    # --> for now
            ###    self.ObsDataList = []
            ###
            ###return self.GuideBoxCommander(self.current_flex_pix_rate)
            ###
            #### Below didn't work too well, but was a good start.
            ###if len(OThisFiltList) > 1:
            ###    # We can calculate the obj_center motion from two
            ###    # measurements through the same filter
            ###    dpix = (OThisFiltList[-1].obj_center
            ###            - OThisFiltList[-2].obj_center)
            ###    dt = OThisFiltList[-1].Tmidpoint - OThisFiltList[-2].Tmidpoint
            ###    # For our particular filter, -dpix/dt would give us
            ###    # the pixel rate we want to cancel our object motion.
            ###    # However, we are likely to be interleaving our
            ###    # measurements, so we need to account for telescope
            ###    # recentering and adjustments to the obj_center_rate
            ###    # that the measurements through the other filters
            ###    # induced.  The basic idea is to recalculate the
            ###    # vector that leads from the old obj_center to the new
            ###    # one in the frame of no corrections.  Then we replace
            ###    # the current rate with the new one.  For ease of
            ###    # bookeeping, start from the last measurement and work
            ###    # toward earlier times
            ###    dpix_other_filt = 0
            ###    # The effective time of an obj_center measurement is
            ###    # the midpoint of the observation.  So to get apples
            ###    # in line with apples, we need to calculate our
            ###    # dpix_other_filt begin and end on our two filter
            ###    # measurement's Tmidpoint values.  For the points
            ###    # between, we calculate the total amount of motion for
            ###    # the total elapsed time.  --> Try to extend this to
            ###    # all previous measurements of this filter --> I want
            ###    # some way to put all measurements on the same
            ###    # footing, so I can plot them all on one linear graph.
            ###    # I think I have that already in the O.flex_pix_rate
            ###    end_t = OThisFiltList[-1].Tmidpoint
            ###    for O in self.ObsDataList[::-1]:
            ###        # --> Eventually make this the actual time the
            ###        # --> rate changed in GuideBoxMover
            ###        start_t = np.max(
            ###            (OThisFiltList[-2].Tmidpoint, O.TRateChange))
            ###        # FITS times are in JD, not seconds
            ###        Odpix = (O.flex_pix_rate * u.pix/u.s * (end_t - start_t) 
            ###                 + O.delta_pix * u.pix)
            ###        Odpix.decompose()
            ###        log.debug('Odpix: ' + str(Odpix))
            ###        dpix_other_filt += Odpix.value
            ###        end_t = start_t
            ###        #if O == OThisFiltList[-2]:
            ###        # Go back to the very beginning
            ###        if O == OThisFiltList[-2]:
            ###            # When we get back to the first measurement
            ###            # through our filter, we don't include its delta_pix
            ###            dpix_other_filt -= O.delta_pix
            ###            log.debug('dpix_other_filt: ' + str(dpix_other_filt))
            ###            break
            ###
            ###    # --> Check to se if dpix_other_filt is larger than
            ###    # --> our measurement
            ###
            ###    # Provisionally set our flex_pix_rate.  Again, dt is
            ###    # in time units
            ###    dt = dt.to(u.s).value
            ###    log.debug('dt: ' + str(dt))
            ###    self.current_flex_pix_rate \
            ###        = (-1 * (dpix + dpix_other_filt) / dt
            ###           * self.flex_aggressiveness)
            ###    # Do sanity checks
            ###    if len(self.ObsDataList) > 5:
            ###        flex_pix_diff \
            ###            = np.asarray(
            ###                [np.linalg.norm(
            ###                    self.ObsDataList[-1].flex_pix_rate
            ###                    - self.current_flex_pix_rate)
            ###                 for O in self.ObsDataList[:-1]])
            ###        noise = np.std(flex_pix_diff[1:] - flex_pix_diff[0:-1])
            ###        if (flex_pix_diff[-1] > self.flex_pix_std_mult * noise):
            ###            log.warning('Unusually large flex_pix_rate: ' + str(self.ObsDataList[-1].flex_pix_rate) + '.  Cutting flex_pix_rate down by 1/2')
            ###            self.current_flex_pix_rate *= 0.5

            #### --> Start the GuideBoxCommander/Mover system even if we
            #### have zero rate, since, at the moment, this is the only
            #### entry point for it.
            ####self.GuideBoxCommander(self.current_flex_pix_rate)
            ####           
            #### Do a telecsope move using move_with_guide_box to correct
            #### for not being at desired_center.  For now take the
            #### center of gravity of the accumulated filter offsets as
            #### the desired center position.  --> If this causes
            #### problems with on-off-band subtraction, may wish to use
            #### some sort of neighbor algorithm to measure relative
            #### offsets and position them all into the center with scope
            #### moves before each exposure
            ###flist = []
            ###OUniqFiltList = []
            ###for O in self.ObsDataList[::-1]:
            ###    if O.header['FILTER'] in flist:
            ###        continue
            ###    flist.append(flist)
            ###    OUniqFiltList.append(O)
            #### Calculate the mean center
            ###running_total = np.zeros(2)
            ###for O in OUniqFiltList:
            ###    running_total += O.obj_center
            ###mean_center = running_total / len(OUniqFiltList)
            ###log.debug('mean_center (X, Y): ' + str(mean_center[::-1]))
            ###dpix = self.ObsDataList[-1].desired_center - mean_center
            ###log.debug('dpix (X, Y): ' + str(dpix[::-1]))
            ###if np.linalg.norm(dpix) > self.desired_center_tolerance:
            ###    # Make our scope adjustment only if it is large -->
            ###    # Note that this assumes real time measurement, so the
            ###    # scope is pointed in the correct direction (at least
            ###    # DEC)
            ###    self.last_delta_pix = dpix
            ###    dra_ddec = self.MD.scope_wcs(dpix,
            ###                                 to_world=True,
            ###                                 delta=True,
            ###                                 astrometry=self.MD.main_astrometry)
            ###    log.debug('dra_ddec: ' + str(dra_ddec))
            ###    # --> I might decide that a move is too disruptive to
            ###    # the rate calculations and just start them over
            ###    self.MD.move_with_guide_box(dra_ddec)
            ###
            ###return True





            #dpix_other_filt = 0
            #last_t = OThisFiltList[-1].Tmidpoint
            #for gbfr in self.flex_pix_rates[::-1]:
            #    this_t = gbfr['Tmidpoint']
            #    if this_t < OThisFiltList[-2].Tmidpoint:
            #        # We don't correct for the rate that was in effect
            #        # when we recorded our first point, since we might
            #        # be the next measurement of any kind
            #        break
            #    dpix_other_filt += gbfr['obj_center_rate'] * (last_t - this_t)
            #    last_t = this_t
            #
            #
            #
            #
            #
            ## Let's make the first order assumption that filter tip
            ## doesn't effect the desired center.  However, 
            #desired_center_rate = 1
            #
            #current_rate *= self.desired_center_aggressiveness
            #
            ## I could potentially grep through the ObsDataList to pull
            ## this stuff out each time, but I don't think Python
            ## enough yet to do that.  Figuring out dictionaries was
            ## enough for this lesson
            #filt = O.header['FILTER']
            #if not filt in self.movement:
            #    # On our first entry, all we have is the fact that we
            #    # may be off-center.  Start to head in the correct
            #    # direction
            #    self.movement[filt] = {'T': O.Tmidpoint,
            #                           'dra_ddec': O.dra_ddec}
            #    current_rate = -1 * (self.movement[filt]['dra_ddec']
            #                         * self.desired_center_aggressiveness)
            #else:
            #    # For subsequent measurements, start to build up our
            #    # lists of time and dra_ddec
            #    self.movement[filt]['T'].append(O.Tmidpoint)
            #    self.movement[filt]['dra_ddec'].append(O.dra_ddec)
            #    dt = self.movement[filt]['T']
            #
            #current_rate = -1* movement[-1]/dt[-1] * self.flex_aggressiveness
            #
            #
            ## Do this simply first then add the complexity of multiple
            ## entries and filters
            #D.say(O.dra_ddec)
            ## Slice such than these run from most recent to least
            #dt = (np.asarray(self.ObsDataList[1:].Tmidpoint)
            #      - np.asarray(self.ObsDataList[0:-1].Tmidpoint))
            ## Movement is distinct from distance from distance from
            ## desired center.  We want to cancel out the movement and
            ## move the object to the desired center
            #movement = (np.asarray(self.ObsDataList[1:].dra_ddec)
            #            - np.asarray(self.ObsDataList[0:-1].dra_ddec))
            #D.say('Movement:')
            #D.say(movement)
            ## Movement rate is what we subtract from the current rate
            #current_rate = -1* movement[-1]/dt[-1] * self.flex_aggressiveness
            #self.flex_pix_rates.append(current_rate)
            #D.say('Guide box rates from flexion:')
            #D.say(flex_pix_rate)

        def pix_rate_to_center(self):
            """Calculates portion of flex_pix_rate to center target"""

            log.debug('STARTING CENTERING RATE CALCULATIONS')
            
            dpix = (self.ObsDataList[-1].obj_center -
                    self.ObsDataList[-1].desired_center)

            ### Try to keep it simple, with our rate determined by the
            ### most we would want to move in our max exposure.
            ### Modulo a minus sign I can't keep track of, this alone
            ### sort of worked, keeping the object in the same place,
            ### but off-center.  Rates didn't resemble expected rates
            ### from MaxIm guide box motion. See notes on
            ### Sun Feb 11 05:25:43 2018 EST  jpmorgen@snipe
            ###return self.GuideBoxCommander(-dpix / 10)
            
            # Determine if we have a significant measurement of
            # distance from the center --> possibly make this larger
            # to match the desired_center_tolerance
            if np.all(np.abs(dpix)
                      < self.ObsDataList[-1].desired_center_tolerance):
                log.debug('Close enough to center')
                return np.zeros(2)
            # Calculate the median time between exposures to provide
            # a reasonable rate of motion
            if len(self.Tmidpoints) == 1:
                log.debug('First exposure.  Guessing 60s for time between exposures')
                new_centering_rate = -dpix/60
            else:
                dt = np.median(self.Tmidpoints[1:] - self.Tmidpoints[0:-1])
                new_centering_rate = -dpix/dt.sec
            log.debug('NEW CENTERING RATE (main camera pix/s): ' +
                      str(new_centering_rate))
            return new_centering_rate


        def update_flex_pix_rate(self,
                                 ObsData_or_fname=None):
            """Returns True if rate was updated, False otherwise"""

            # This method sets up the list of ObsData that is used by
            # the methods it calls to actually calculate the rates
            log.debug('PREPARING TO UPDATE FLEX PIX RATE')
            if isinstance(ObsData_or_fname, str):
                Ocurrent = self.ObsDataClass(ObsData_or_fname)
            else:
                Ocurrent = ObsData_or_fname
            assert isinstance(Ocurrent, ObsData)

            dpix = Ocurrent.obj_center - Ocurrent.desired_center
            log.debug('DPIX FROM CENTER: ' +
                      str(Ocurrent.obj_center - Ocurrent.desired_center))

            if Ocurrent.quality < 5:
                log.warning('Skipping flex rate motion calculations because obj_center quality estimate is too low: ' + str(Ocurrent.quality))
                return False

            # Build up our list of ObsData for motion calculations.
            # Note that Ocurrent still points to the last object so we
            # can conveniently fill it in with that reference.
            self.ObsDataList.append(Ocurrent)

            # Record the current_flex_pix_rate that was operating
            # while this exposure was being recorded.  
            Ocurrent.flex_pix_rate = self.current_flex_pix_rate
            # Record the time at which this rate started
            shutter_time = Time(Ocurrent.header['DATE-OBS'], format='fits')
            if len(self.ObsDataList) == 1:
                # For the first exposure in our list, we don't care
                # about motion before the exposure started
                Ocurrent.TRateChange = shutter_time
            else:
                # For observations further along in the sequence, the
                # start time of our rate could have been significantly
                # before the start of the exposure time, since the new
                # rate is started at the end of the previous exposure
                # and the user could have waited to start this
                # exposure.  Nevertheless, we want to track the actual
                # motion of the guide box over the whole period of its
                # motion, so for these calculations, store the actual
                # time the rate changed in the handy box of the
                # ObsData of the observation it primarily affects
                Ocurrent.TRateChange = self.current_flex_pix_TStart
                # Do a sanity check on this in case we are processing
                # files after the fact
                if Ocurrent.TRateChange > shutter_time:
                    Ocurrent.TRateChange = shutter_time
                    log.debug('Detecting after-the-fact run')

            # Now, for this measurement, calculate the total guidebox
            # motion since our very first observation.  We have
            # already done this for previous measurements, but don't
            # worry about that.  For times before this measurement, we
            # just add up the guide box motion without worrying about
            # when it was measured (the total_flex_dpix of the
            # individual measurements will do that).  It would be a
            # little more efficient to store the "all past dpix" and
            # add to it incrementally, but I think this is more clear.
            # Note that for intervals over which there is no change in
            # the rate, this provides no contribution.  But when there
            # is a change, this makes up for it.
            Ocurrent.total_flex_dpix = np.zeros(2)
            for i, O in enumerate(self.ObsDataList[0:-1]):
                #D.say('delta t: ' + str(i) + ' ' +
                #      str((self.ObsDataList[i+1].TRateChange
                #      - O.TRateChange).sec))
                #D.say('flex_pix_rate: ' + str(i) + ' ' +
                #      str(O.flex_pix_rate))
                #D.say('total_flex_dpix: ' + str(i) + ' ' +
                #      str(O.flex_pix_rate
                #           * (self.ObsDataList[i+1].TRateChange
                #              - O.TRateChange).sec))
                Ocurrent.total_flex_dpix \
                    += (O.flex_pix_rate
                        * (self.ObsDataList[i+1].TRateChange
                           - O.TRateChange).sec)
            # The final piece for our current observation is a little
            # different, which is why we can't just do this once and
            # be done.  Guidebox motion lasted through the whole
            # observation, however, the point at which we can actually
            # measure it is the midpoint of the observation.  Note
            # that if there is no change in the rate over past
            # measurements, this does the right thing, since
            # Ocurrent.TRateChange is the start of the first
            # observation
            Ocurrent.total_flex_dpix \
                += (Ocurrent.flex_pix_rate
                   * (Ocurrent.Tmidpoint - Ocurrent.TRateChange).sec)
            # Convert the attributes in the list into numpy arrays and
            # store them in our PrecisionGuide object for use in other
            # routines
            measListOLists = [[O.obj_center,
                               O.obj_center_err,
                               O.Tmidpoint,
                               O.header['EXPTIME'],
                               O.total_flex_dpix,
                               O.TRateChange]
                              for O in self.ObsDataList]
            # The * unpacks the top level ListOLists (one per O
            # matching current_flex_pix_rate) to provide zip with a
            # bunch of lists that have 3 elements each.  Zip then
            # returns a list of 3 tuples, where the tuples have the
            # long list of elements we want
            measTuples = list(zip(*measListOLists))
            self.obj_centers = np.asarray(measTuples[0])
            self.obj_center_errs = np.asarray(measTuples[1])
            self.Tmidpoints = np.asarray(measTuples[2])
            self.exptimes = np.asarray(measTuples[3])
            self.total_flex_dpix = np.asarray(measTuples[4])
            self.TRateChanges = np.asarray(measTuples[5])

            # Use the average so we are weighted by longer exposure
            # times, since we use this when calculating new rate
            # changes
            self.average_exptime = np.average(self.exptimes)

            # Estimate effect of seeing on short exposures --> for now
            # just call seeing 2 pixels.  Eventually we want the
            # formal seeing in arcsec, measured in real time through a
            # verity of means
            seeing_pix = 2
            seeing_freq = 1 # hz, upper limit on frequency of detectable variations
            seeing_err = seeing_pix/(1/seeing_freq + self.exptimes)
            # Match the shape of our obj_center_errs, which has a list
            # of coordinates (N, 2).  For broadcast, the last element
            # of the shape needs to be the length of our original
            # array.  But that ends up being the transpose of for our
            # obj_center_errs
            seeing_err = np.transpose(
                np.broadcast_to(seeing_err, (2, len(seeing_err))))
            self.obj_center_errs = (self.obj_center_errs +
                                    seeing_err**2)**0.5

            # Now that we have populated our object, we can derive our
            # rates from the data
            new_motion_rate = self.pix_rate_to_freeze_motion()
            #new_centering_rate = self.pix_rate_to_center()
            new_centering_rate = np.zeros(2)
            current_motion_rate = (self.current_flex_pix_rate -
                                   self.current_centering_rate)
            if (np.any(current_motion_rate != new_motion_rate)):
                log.debug('Updating motion rate: ' + str(new_motion_rate))
            if (np.any(self.current_centering_rate != new_centering_rate)):
                log.debug('Updating centering rate' + str(new_centering_rate))
                self.current_centering_rate = new_centering_rate
            new_flex_pix_rate = new_motion_rate + new_centering_rate
            log.debug('[NEW_]FLEX_PIX_RATE: ' + str(new_flex_pix_rate))
            if np.any(self.current_flex_pix_rate != new_flex_pix_rate):
                return self.GuideBoxCommander(new_flex_pix_rate)
            return False
            


        def MaxImCollector(self):
            self.MD.CCDCamera.EventMask = 2
            log.debug('MaxIm set to notify when main camera exposure complete')
            #for i in range(3):
            #    event = self.MD.CCDCamera.Notify
            #    log.debug('Exposure ended: ' + str(event))

        # Mostly this just passes parameters through to
        # MaxImData.acquire_im to take the image.  We need to have
        # **ObsClassArgs so that images of different ObsClass type can
        # peacefully coexist in the set of precision guide stuff (-->
        # though if we go to a different object, we will probably need
        # to reinitialize) --> I may need to check for RA and DEC
        # being different
        def acquire_image(self,
                          fname='Test.fits',
                          exptime=None,
                          filt=None,
                          binning=None,
                          subarray=None,
                          ACP_obj=None,
                          **ObsClassArgs):
            """Acquire an image using the PrecisionGuide system."""
            assert self.MD.CCDCamera.GuiderRunning, 'Guider must be running.  You can start it with PrecisionGuide.MD.guider_start()'

            # Here might be where we make the choice to use ACP's
            # TakePicture or record it ourselves based on whether or
            # not ACP's objects are present
            if ACP_obj:
                # Eventually we would read the file from the disk
                # Consider using ACP's TakePicture
                HDUList = fits.open(fname)
                O = self.create_ObsData(HDUList, **ObsClassArgs)
            else:
                HDUList = self.MD.acquire_im(fname=fname,
                                             exptime=exptime,
                                             filt=filt,
                                             binning=binning,
                                             subarray=subarray)
                #HDUList = self.MD.take_im(exptime, filt, binning)
                ## Write image to disk right away in case something goes wrong
                #if not self.MD.CCDCamera.SaveImage(fname):
                #    raise EnvironmentError('Failed to save file ' + fname)
                #log.debug('Saved file: ' + fname)
                # Use the version of our image in HDUList for
                # processing so we don't have to read it off the disk
                # again
                O = self.create_ObsData(HDUList, **ObsClassArgs)
            return self.update_flex_pix_rate(O)

        # Used pythoncom.CreateGuid() to generate this Fired up a
        # Python command line from cmd prompt in Windows.  The
        # followining helped:
        # https://www.pythonstudio.us/introduction-2/implementing-com-objects-in-python.html
        # http://timgolden.me.uk/pywin32-docs/contents.html
        # import pythoncom
        # print pythoncom.CreateGuid()
        # {3E09C890-40C9-4326-A75D-AEF3BF0E099F}


log.setLevel('DEBUG')

#log.setLevel('INFO')
#log.setLevel('WARNING')

#O = CorObsData('/data/io/IoIO/raw/2018-01-31/Problem_R-band.fit')#, plot_dprof=True, plot_ND_edges=True)
#D.say(O.obj_center[::-1])
#D.say(O.desired_center[::-1])

#O = CorObsData('/data/io/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit')#, plot_dprof=True, plot_ND_edges=True)
#D.say(O.obj_center[::-1])
#D.say(O.desired_center[::-1])
#
#
#F = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit')#, plot_dprof=True, plot_ND_edges=True)
#default_ND_params = F.ND_params
#O = CorObsData('/data/io/IoIO/raw/2017-04-20/Filter_sequence-0001_1s_open.fit', default_ND_params=default_ND_params)
#D.say(O.obj_center[::-1])
#D.say(O.desired_center[::-1])

#F = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit')#, plot_dprof=True, plot_ND_edges=True)
#default_ND_params = F.ND_params
#D.say(default_ND_params)
#F2 = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
#default_ND_params = F2.ND_params
#D.say(default_ND_params)
#F3 = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
#default_ND_params = F3.ND_params
#D.say(default_ND_params)
#D.say(F3.obj_center)
#O = CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0014_SII_on-band.fit', default_ND_params=default_ND_params) #plot_prof=True, plot_dprof=True, plot_ND_edges=True, 
#D.say(O.ND_params)
#D.say(O.obj_center)
#D.say(O.desired_center)
#D.say(O.obj_center - O.desired_center)
#D.say(O.obj_to_ND)
#O = CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0028_moving_to_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
#D.say(O.ND_params)
#D.say(O.obj_center)
#D.say(O.desired_center)
#D.say(O.obj_center - O.desired_center)
#D.say(O.obj_to_ND)


#start = time.time()
#rawdir = '/data/io/IoIO/raw/2017-05-28'
#raw_fnames = [os.path.join(rawdir, f)
#              for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
#
#for f in sorted(raw_fnames):
#    D.say(f)
#    try:
#        O = CorObsData(f, default_ND_params=default_ND_params)
#        if O.obj_to_ND > 10:
#            D.say('Large dist: ' + str(int(O.obj_to_ND)))
#    except ValueError as e:
#        log.error('Skipping: ' + str(e))
#
#    
#end = time.time()
#D.say('Elapsed time: ' + str(D.say(end - start)) + 's')


#start = time.time()
#rawdir = '/data/io/IoIO/raw/2017-05-28'
#raw_fnames = [os.path.join(rawdir, f)
#              for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
#
#for f in sorted(raw_fnames):
#    D.say(f)
#    try:
#        O4 = CorObsData(f, n_y_steps=4, default_ND_params=default_ND_params)
#        O8 = CorObsData(f, n_y_steps=8, default_ND_params=default_ND_params)
#        O16 = CorObsData(f, n_y_steps=16, default_ND_params=default_ND_params)
#        D.say(O8.obj_to_ND)
#        dc4 = np.abs(O4.desired_center - O16.desired_center)
#        dc8 = np.abs(O8.desired_center - O16.desired_center)
#        #if (dc4 > 5).any() or (dc8 > 5).any():
#        #D.say(O.obj_center - O.desired_center)
#        print(O4.desired_center - O16.desired_center)
#        print(O4.obj_center - O16.obj_center)
#        print('4^---8>-----')
#        print(O8.desired_center - O16.desired_center)
#        print(O8.obj_center - O16.obj_center)
#    except ValueError as e:
#        log.error('Skipping: ' + str(e))
#
#    
#end = time.time()
#D.say('Elapsed time: ' + str(D.say(end - start)) + 's')

#O = CorObsData('/data/io/IoIO/raw/2017-04-18/IPT-0001_off-band.fit', default_ND_params = ((-0.07286433475733832, -0.068272558665046126), (1251.595679328457, 1357.3942953038429)))
#D.say(O.obj_center)

#default_ND_params = get_default_ND_params('/data/io/IoIO/raw/2017-05-22')
#D.say(default_ND_params)
#O = CorObsData('/data/io/IoIO/raw/2017-05-27/IPT-0007_on-band.fit',
#               default_ND_params=default_ND_params,
#               plot_prof=True, plot_dprof=True, plot_ND_edges=True)
#D.say(O.ND_params)
#D.say(O.obj_center)
#D.say(O.desired_center)

#print(process_dir('/data/io/IoIO/raw/2017-04-26',
#                  default_ND_params=get_default_ND_params(
#                      '/data/io/IoIO/raw/2017-04-24')))



#Na =   CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0007_Na_off-band.fit')
#print(Na.obj_center)
#print(Na.desired_center)
#print(Na.obj_center - Na.desired_center)

#print(Na.get_ND_params())
#print(Na.ND_params)
#print(Na.ND_angle)
#Na.ND_params = (((3.75447820e-01,  3.87551301e-01), 
#                 (1.18163633e+03,   1.42002571e+03)))
#Na.ND_params = ('recalc')


#SII =  CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0035_SII_on-band.fit')
#print(SII.get_ND_params())
#print(SII.ND_angle)

#flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit')
#flat.imshow()
#print(flat.get_ND_params())
#print(flat.ND_angle)


# flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
# flat.imshow()
# print(flat.get_ND_params())
# print(flat.ND_angle)

def cmd_center(args):
    if sys.platform != 'win32':
        raise EnvironmentError('Can only control camera and telescope from Windows platform')
    default_ND_params = None
    if args.ND_params is not None:
        default_ND_params = get_default_ND_params(args.ND_params, args.maxcount)
        P = PrecisionGuide(args.ObsClassName,
                           args.ObsClassModule,
                           default_ND_params=default_ND_params) # other defaults should be good
    else:
        P = PrecisionGuide(args.ObsClassName,
                           args.ObsClassModule) # other defaults should be good
    P.center_loop()

def cmd_guide(args):
    if sys.platform != 'win32':
        raise EnvironmentError('Can only control camera and telescope from Windows platform')
    MD = MaxImData()
    if args.stop:
        MD.guider_stop()
    else:
        MD.guider_start(exptime=args.exptime, filter=args.filter)

def cmd_get_default_ND_params(args):
    print(get_default_ND_params(args.dir, args.maxcount))


# --> Eventually, I would like this to accept input from other
# --> sources, like a flexure model and ephemeris rates
def GuideBoxMover(args):
    log.debug('Starting GuideBoxMover')
    MD = MaxImData()
    last_modtime = 0
    while True:
        # --> Make this sleep time variable based on a fraction of the
        # --> expected motion calculated below
        time.sleep(1)
        # Wait until we have a file
        if not os.path.isfile(args.command_file):
            continue
        # Check to see if the file has changed.  --> Note, when this
        # starts up, it will grab the rate from the last file write
        # time time unless a GuideBoxCommander() is done to zero out
        # the file
        this_modtime = os.path.getmtime(args.command_file)
        if this_modtime != last_modtime:
            # Check to see if the MaxIm guider is on
            last_modtime = this_modtime
            with open(args.command_file, 'r') as com:
                rates_list = json.loads(com.read())
            dra_ddec_rate = np.array(rates_list)
            # Rates were passed in arcsec/hour.  We need degrees/s
            dra_ddec_rate /= 3600**2
            lastt = time.time()
        # Only move when we have a move of more than 0.5 arcsec --> Make
        # this a constant or something poassibly in MD
        now = time.time()  
        dt = now - lastt
        if np.linalg.norm(dra_ddec_rate) * dt >= 0.5/3600:
            log.debug('GuideBoxMover dt(s) = ' + str(dt))
            log.debug('GuideBoxMover is moving the guidebox by ' +
                      str(dra_ddec_rate * dt*3600) + ' arcsec')
            MD.move_with_guide_box(dra_ddec_rate * dt )
            lastt = now

def uniq_fname(basename=None, directory=None, extension='.fits'):
    if directory is None:
        directory = '.'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if basename is None:
        basename = 'unique_fname'
    fnum = 1
    while True:
        fname = os.path.join(directory,
                             basename
                             + '{num:03d}'.format(num=fnum)
                             + extension)
        if os.path.isfile(fname):
            fnum += 1
        else:
            return fname

# Data collector for testing
def data_collector(args):
    d = args.dir
    if d is None:
        today = Time.now().fits.split('T')[0]
        d = os.path.join(raw_data_root, today)
    basename = args.basename
    if basename is None:
        basename = 'PrecisionGuideDataFile'
    P = PrecisionGuide(args.ObsClassName,
                       args.ObsClassModule) # other defaults should be good
    if not P.MD.CCDCamera.GuiderRunning:
        # User could have had guider already on.  If not, center with
        # guider slews and start the guider
        log.debug('CENTERING WITH GUIDER SLEWS') 
        P.center_loop(max_tries=5)
        log.debug('STARTING GUIDER') 
        P.MD.guider_start()
    # Center with guide box moves
    #log.debug('NOT CENTERING WITH GUIDE BOX MOVES WHILE DOING LARGE GUIDE RATE EXPERIMENT ') 
    P.center_loop()
    # Put ourselves in GuideBoxMoving mode (starts the GuideBoxMover subprocess)
    log.debug('STARTING GuideBoxMover')
    # --> this is a confusing name: mover/moving
    P.GuideBoxMoving = True
    while True:
        fname = uniq_fname(basename, d)
        log.debug('COLLECTING: ' + fname)
        P.acquire_image(fname,
                        exptime=args.exptime,
                        filt=args.filt)
        # --> Just for this could eventually do a sleep watchdog or
        # --> guider settle monitor....
        time.sleep(7)


#def IPT_Na_R(args):
#    P = PrecisionGuide("CorObsData") # other defaults should be good
#    d = args.dir
#    if d is None:
#        today = Time.now().fits.split('T')[0]
#        d = os.path.join(raw_data_root, today)
#    basename = args.basename
#    if basename is None:
#        basename = 'IPT_Na_R_'
#    if not P.MD.CCDCamera.GuiderRunning:
#        # User could have had guider already on.  If not, center with
#        # guider slews and start the guider
#        log.debug('CENTERING WITH GUIDER SLEWS') 
#        P.center_loop()
#        log.debug('STARTING GUIDER') 
#        P.MD.guider_start()
#    log.info('Starting with R')
#    P.acquire_image(uniq_fname(basename, d),
#                    exptime=2,
#                    filt=0)
#    
#    # Jupiter observations
#    while True:
#        #fname = uniq_fname(basename, d)
#        #log.debug('data_collector preparing to record ' + fname)
#        #P.acquire_image(fname,
#        #                exptime=7,
#        #                filt=0)
#        # was 0.7, filt 1 for mag 1 stars
#        # 0 = R
#        # 1 = [SII] on-band
#        # 2 = Na on-band
#        # 3 = [SII] off-band
#        # 4 = Na off-band
#
#        log.info('Collecting Na')
#        P.acquire_image(uniq_fname(basename, d),
#                        exptime=60,
#                        filt=4)
#        P.acquire_image(uniq_fname(basename, d),
#                        exptime=300,
#                        filt=2)
#        for i in range(5):
#            log.info('Collecting [SII]')
#            P.acquire_image(uniq_fname(basename, d),
#                            exptime=300,
#                            filt=1)
#            P.acquire_image(uniq_fname(basename, d),
#                            exptime=60,
#                            filt=3)
#            log.info('Collecting R')
#            P.acquire_image(uniq_fname(basename, d),
#                            exptime=2,
#                            filt=0)

def IPT_Na_R(args):
    P = PrecisionGuide("CorObsData") # other defaults should be good
    d = args.dir
    if d is None:
        today = Time.now().fits.split('T')[0]
        d = os.path.join(raw_data_root, today)
    basename = args.basename
    if basename is None:
        basename = 'IPT_Na_R_'
    if not P.MD.CCDCamera.GuiderRunning:
        # User could have had guider already on.  If not, center with
        # guider slews and start the guider
        log.debug('CENTERING WITH GUIDER SLEWS') 
        P.center_loop(max_tries=5)
        log.debug('STARTING GUIDER') 
        P.MD.guider_start(filter=3)
    log.debug('CENTERING WITH GUIDEBOX MOVES') 
    P.center_loop()
    log.info('Starting with R')
    P.MD.acquire_im(uniq_fname(basename, d),
                    exptime=2,
                    filt=0)
    
    # Jupiter observations
    while True:
        #fname = uniq_fname(basename, d)
        #log.debug('data_collector preparing to record ' + fname)
        #P.acquire_image(fname,
        #                exptime=7,
        #                filt=0)
        # was 0.7, filt 1 for mag 1 stars
        # 0 = R
        # 1 = [SII] on-band
        # 2 = Na on-band
        # 3 = [SII] off-band
        # 4 = Na off-band

        log.info('Collecting Na')
        P.MD.acquire_im(uniq_fname(basename, d),
                        exptime=60,
                        filt=4)
        P.MD.acquire_im(uniq_fname(basename, d),
                        exptime=300,
                        filt=2)
        log.debug('CENTERING WITH GUIDEBOX MOVES') 
        P.center_loop()
        
        for i in range(5):
            log.info('Collecting [SII]')
            P.MD.acquire_im(uniq_fname(basename, d),
                            exptime=300,
                            filt=1)
            P.MD.acquire_im(uniq_fname(basename, d),
                            exptime=60,
                            filt=3)
            log.debug('CENTERING WITH GUIDEBOX MOVES') 
            P.center_loop()
            log.info('Collecting R')
            P.MD.acquire_im(uniq_fname(basename, d),
                            exptime=2,
                            filt=0)


def MaxImCollector(args):
    P = PrecisionGuide()
    P.MaxImCollector()

def test_flex(args):
    """Test flex_pix_rate calculations in dir
    """
    directory = os.path.join(raw_data_root, args.dir)
    #if not os.path.isdir(args.dir):
    #    raise ValueError("Specify directory")
    if not os.path.isdir(directory):
        raise ValueError("Specify directory")
    #if args.maxcount is None:
    #    maxcount = 10

    # This could be a general routine for getting objects
    files = [os.path.join(directory, f)
             for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    # The sort method sorts in place, so the return value is None
    files.sort(key=os.path.getmtime)
    obj_fnames = []
    for f in files:
        HDUList = fits.open(f)
        if HDUList[0].header['IMAGETYP'].upper() == 'LIGHT':
            obj_fnames.append(f)
        HDUList.close()

    if len(obj_fnames) == 0:
        log.error('No files found')
        return
    D.say('Starting PrecisionGuide')
    P = PrecisionGuide(args.ObsClassName,
                       args.ObsClassModule) # other defaults should be good
    D.say('Finished starting PrecisionGuide')
    for f in obj_fnames:
        D.say('Processing file ' + f)
        retval = P.update_flex_pix_rate(f)
        if retval:
            D.say('UPDATED FLEX RATE')
        else:
            D.say('DID NOT UPDATE FLEX RATE')
        


#ND=NDData('//snipe/data/io/IoIO/raw/2017-05-29/Sky_Flat-0001_Na_off-band.fit')
#print(ND.get_ND_params())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoIO-related instrument control and image reduction")
    # --> Update this with final list once settled
    subparsers = parser.add_subparsers(dest='one of the subcommands in {}', help='sub-command help')
    subparsers.required = True

    guide_parser =  subparsers.add_parser(
        'guide', help='Start guider (usually used after center)')
    guide_parser.add_argument(
        '--exptime', help='Exposure time to use for guider')
    guide_parser.add_argument(
        '--filter', help='Guider filter (e.g., 0)) or filter search sequence (e.g., "(0,1,2,3)" for auto exposure calculations (start with most transparent filter first')    
    guide_parser.add_argument(
        '--stop', action="store_true", help='Stops guider')
    guide_parser.set_defaults(func=cmd_guide)

    center_parser = subparsers.add_parser(
        'center', help='Record image and center object')
    center_parser.add_argument(
        '--ObsClassName', help='ObsData class name')
    center_parser.add_argument(
        '--ObsClassModule', help='ObsData class module file name')
    # These are specific to the coronagraph --> thinking I might be
    # able to pass package-specific arguments to subclass init in a
    # clever way by capturing the rest of the command line in one
    # argument and then parsing it in init
    center_parser.add_argument(
        '--ND_params', help='Derive default_ND_params from flats in this directory')
    center_parser.add_argument(
        '--maxcount', help='maximum number of flats to process -- median of parameters returned')
    center_parser.set_defaults(func=cmd_center)

    GuideBox_parser = subparsers.add_parser(
        'GuideBoxMover', help='Start guide box mover process')
    GuideBox_parser.add_argument(
        'command_file', help='Full path to file used to pass rates from GuideBoxCommander  to GuideBoxMover')
    GuideBox_parser.set_defaults(func=GuideBoxMover)

    Collector_parser = subparsers.add_parser(
        'MaxImCollector', help='Collect images from MaxIm  for precision guiding')
    Collector_parser.set_defaults(func=MaxImCollector)

    data_collector_parser =  subparsers.add_parser(
        'data_collector', help='Collect images in a file name sequence')
    # --> Other things would need to be here to be useful, like
    # --> potentially reading a file.  But that is ACP, so just keep
    # --> this around for testing
    data_collector_parser.add_argument(
        '--dir', help='directory, default current date in YYYY-MM-DD format')
    data_collector_parser.add_argument(
        '--basename', help='base filename for files, default = PrecisionGuideDataFile')
    data_collector_parser.add_argument(
        '--exptime', help='exposure time, default = default exposure time')
    data_collector_parser.add_argument(
        '--filt', help='filter, default = default filter')
    data_collector_parser.add_argument(
        '--ObsClassName', help='ObsData class name')
    data_collector_parser.add_argument(
        '--ObsClassModule', help='ObsData class module file name')
    data_collector_parser.set_defaults(func=data_collector)

    IPT_Na_R_parser =  subparsers.add_parser(
        'IPT_Na_R', help='Collect IPT, Na, and R measurements')
    IPT_Na_R_parser.add_argument(
        '--dir', help='directory, default current date in YYYY-MM-DD format')
    IPT_Na_R_parser.add_argument(
        '--basename', help='base filename for files, default = IPT_Na_R_')
    IPT_Na_R_parser.set_defaults(func=IPT_Na_R)

    # --> This eventually goes just with ioio.py
    ND_params_parser = subparsers.add_parser(
        'ND_params', help='Get ND_params from flats in a directory')
    ND_params_parser.add_argument(
        'dir', nargs='?', default='.', help='directory')
    ND_params_parser.add_argument(
        'maxcount', nargs='?', default=None,
        help='maximum number of flats to process -- median of parameters returned')
    ND_params_parser.set_defaults(func=cmd_get_default_ND_params)

    test_flex_parser =  subparsers.add_parser(
        'test_flex', help='Collect images in a file name sequence')
    # --> Other things would need to be here to be useful, like
    # --> potentially reading a file.  But that is ACP, so just keep
    # --> this around for testing
    test_flex_parser.add_argument(
        'dir', nargs='?', default='.', help='directory')
    test_flex_parser.add_argument(
        '--ObsClassName', help='ObsData class name')
    test_flex_parser.add_argument(
        '--ObsClassModule', help='ObsData class module file name')
    test_flex_parser.set_defaults(func=test_flex)

    args = parser.parse_args()
    # This check for func is not needed if I make subparsers.required = True
    if hasattr(args, 'func'):
        args.func(args)

    #F = CorObsData('//SNIPE/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit')#, plot_dprof=True, plot_ND_edges=True)
    #default_ND_params = F.ND_params
    #O = CorObsData('//SNIPE/data/io/IoIO/raw/2017-05-28/Na_IPT-0010_SII_on-band.fit', default_ND_params=default_ND_params)
    #P.center(O)
    #P.center('//snipe/data/io/IoIO/raw/2017-05-28/Na_IPT-0007_Na_off-band.fit')
    #P.center()
    #print(P.center_loop())
    #time.sleep(20)

    #print(__file__)
    ##ObsDataClass = getattr(importlib.import_module('ioio'), 'CorObsData')
    #cur_file = __file__.split('.py')[0]
    #ObsDataClass = getattr(importlib.import_module(cur_file), 'CorObsData')
    #F = ObsDataClass('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit') #, plot_dprof=True, plot_ND_edges=True)
    #default_ND_params = F.ND_params
    #D.say(default_ND_params)

    #flat = CorObsData('//SNIPE/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
    #O = CorObsData('//SNIPE/data/io/IoIO/raw/2017-05-28/Na_IPT-0010_SII_on-band.fit', default_ND_params=flat.ND_params)
    #D.say(O.obj_center)
        
    #process_dir('//SNIPE/data/io/IoIO/raw/2017-05-28/', default_ND_params=flat.ND_params)
    #flat = CorObsData('/Users/jpmorgen/byted/xfr/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
    #flat.imshow()
    #flat.get_ND_params()
#     # Start MaxIm
#     print('Getting MaxImData object...')
#     M = MaxImData()
#     print('Done')
#     print('Getting MaxImData camera...')
#     M.getCCDCamera()
#     print('Done')
#     # Keep it alive for these experiments
#     M.CCDCamera.DisableAutoShutdown = True
#     #M.connect()
#     #print(M.calc_main_move((50,50)))
#     #print(M.guider_move(10000,20))
#     print('Getting current image')
#     M.get_im()
#     print('Done')
#     print('Getting object center...')
#     J = CorObsData(M.HDUList)
#     obj_cent = J.obj_center()
#     print('object center = ', obj_cent)
#     print('Getting desired center...')
#     desired_cent = J.desired_center()
#     print('desired center = ', desired_cent)
#     print('Done')
#     print('Calculating movement required to center object...')
#     main_move = M.calc_main_move(obj_cent, desired_cent)
#     print('arcsec of movement required (dDEC, dRA) = ', main_move)
#     print('Result of scope move is:')
#     M.guider_move(main_move)
# 
# # print(M.get_object_center_pix())
# # print('Done')
# # print('Centering Jupiter!')
# # M.center_object()
# # print('Done')
# # print(M.calc_main_move((50,50)))
# # print(M.rot((1,1), 45))
# # print(M.HDUList[0].header)
# # plt.imshow(M.HDUList[0].data)
# # plt.show()
# # # Kill MaxIm
# # #M = None
# 
# #print('Getting jupiter center')
# #print(get_jupiter_center('/Users/jpmorgen/byted/xfr/2017-04-20/IPT-0032_off-band.fit'))
# 
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# # 
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))
# # 
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# # 
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
# # print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))
# #ND=NDData('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit')
# #print(ND.get_ND_params())
# 
# #print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
