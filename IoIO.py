#!/usr/bin/python3
#--> Getting command line to work in Windows.  You need to edit
# registry to make 
# Computer\HKEY_CLASSES_ROOT\Applications\python.exe\shell\open\command
# "C:\ProgramData\Anaconda3\python.exe" "%1" %*
# Thanks to https://stackoverflow.com/questions/29540541/executable-python-script-not-take-sys-argv-in-windows

import numpy as np
from astropy import log
from astropy.io import fits
from astropy.time import Time, TimeDelta
import precisionguide as pg
from scipy import signal, ndimage
import matplotlib.pyplot as plt
import argparse

# Constants for use in code
SII_filt_crop = np.asarray(((350, 550), (1900, 2100)))
run_level_default_ND_params \
    = [[  3.63686271e-01,   3.68675375e-01],
       [  1.28303305e+03,   1.39479846e+03]]

class CorObsData(pg.ObsData):
    """Object for containing coronagraph image data used for centering Jupiter
    """
    # This is for jump-starting the ND_params calculation with flats.
    # Be generous in case the angle is very large.  If it is still
    # hard to get a good ND solution, use more n_y_steps
    def __init__(self,
                 HDUList_im_or_fname=None,
                 default_ND_params=None,
                 recalculate=False,
                 readnoise=5, 
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

        self.y_center = y_center
        self.readnoise = readnoise

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
        # saves us the work of using self.get_ND_params, but allow
        # recalculation if we want to
        if self.recalculate == False and self.header.get('NDPAR00') is not None:
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
        # Note, this assignment dereferences im from HDUList[0].data
        im  = im - self.back_level
        
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
                log.warning('Jupiter (or suitably bright object) not found in image.  This object is unlikely to show up on the ND filter.  Seeting quality to ' + str(self.quality) + ', center to [-99, -99]')
                self._obj_center = np.asarray([-99, -99])
            else:
                self.quality = 6
                # If we made it here, Jupiter is outside the ND filter,
                # but shining bright enough to be found
                # --> Try iterative approach
                ny, nx = im.shape
                y_x = np.asarray(ndimage.measurements.center_of_mass(im))
                print(y_x)
                y = np.arange(ny) - y_x[0]
                x = np.arange(nx) - y_x[1]
                # input/output Cartesian direction by default
                xx, yy = np.meshgrid(x, y)
                rr = np.sqrt(xx**2 + yy**2)
                im[np.where(rr > 200)] = 0
                y_x = np.asarray(ndimage.measurements.center_of_mass(im))
    
                self._obj_center = y_x
                log.info('Object center (X, Y; binned) = ' +
                      str(self.binned(self._obj_center)[::-1]))
        else:
            # Here is where we boost what is sure to be Jupiter, if Jupiter is
            # in the ND filter
            # --> this has trouble when there is bright skys
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
            log.debug('Object center (X, Y; binned) = '
                      + str(self.binned(self._obj_center)[::-1]))
            self.quality = 6
        self.header['OBJ_CR0'] = (self._obj_center[1], 'Object center X')
        self.header['OBJ_CR1'] = (self._obj_center[0], 'Object center Y')
        self.header['QUALITY'] = (self.quality, 'Quality on 0-10 scale of center determination')
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
        self.header['DES_CR0'] = (self._desired_center[1], 'Desired center X')
        self.header['DES_CR1'] = (self._desired_center[0], 'Desired center Y')
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

        # Work with unbinned image
        xs = [] ; ys = []
        us = self.unbinned(self.oshape)
        for iy in np.arange(0, us[0]):
            bounds = (self.ND_params[1,:]
                      + self.ND_params[0,:]*(iy - us[0]/2)
                      + np.asarray((self.edge_mask, -self.edge_mask)))
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
        self.header['OBJ2NDC'] = (d,
                                  'Obj perpendicular dist. to ND filt. center, pix')
        self._obj_to_ND = d
        return self._obj_to_ND

    @property
    def ND_params(self):
        """Returns parameters which characterize the coronagraph ND filter, calculating if necessary.  Parameters are relative to unbinned image"""
        if self._ND_params is not None:
            return self._ND_params

        # If we made it here, we need to do the heavy lifting of
        # finding the ND filter in the image
        assert isinstance(self.HDUList, fits.HDUList)

        # These are unbinned coordinates
        ytop = SII_filt_crop[0,0]
        ybot = SII_filt_crop[1,0]

        # Trying a filter to get rid of cosmic ray hits in awkward
        # places.  Do this only for section of CCD we will be working
        # with, since it is our most time-consuming step.  Also work
        # in our original, potentially binned image, so cosmic rays
        # don't get blown up by unbinning
        if self.isflat:
            # Don't bother for flats, just unbin the image
            im = self.HDU_unbinned()
        else:
            # Make a copy so we don't mess up the primary HDU (see below)
            im = self.HDUList[0].data.copy()
            xtop = self.binned(self.ND_edges(ytop, self.default_ND_params))
            xbot = self.binned(self.ND_edges(ybot, self.default_ND_params))
            # Get the far left and right coords, keeping in mind ND
            # filter might be oriented CW or CCW of vertical
            x0 = int(np.min((xbot, xtop))
                     - self.search_margin / self._binning[1])
            x1 = int(np.max((xbot, xtop))
                     + self.search_margin / self._binning[1])
            x0 = np.max((0, x0))
            x1 = np.min((x1, im.shape[1]))
            # This is the operation that messes with the array in place
            im[ytop:ybot, x0:x1] \
                = signal.medfilt(im[ytop:ybot, x0:x1], 
                                 kernel_size=3)
            # Unbin now that we have removed cosmic rays from the section we
            # care about in native binning
            im = self.HDU_unbinned(im)
            
        # We needed to remove cosmic rays from the unbinned version, but
        # now we may have a copy

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
                bounds = SII_filt_crop[:,1]
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
        self.header['NDPAR00'] = (ND_params[0,0], 'ND filt left side slope')
        self.header['NDPAR01'] = (ND_params[1,0], 'ND filt left side offset at Y center of im')
        self.header['NDPAR10'] = (ND_params[0,1], 'ND filt right side slope')
        self.header['NDPAR11'] = (ND_params[1,1], 'ND filt right side offset at Y center of im')

        return self._ND_params

def IPT_Na_R(args):
    P = pg.PrecisionGuide("CorObsData", "IoIO") # other defaults should be good
    wait_for_horizons = False
    while not P.MD.Telescope.Tracking:
        if not wait_for_horizons:
            log.info('Tracking is off.  Assuming I am waiting for A-P HORIZONS app to acquire Jupiter')
        wait_for_horizons = True
        time.sleep(5)
    if wait_for_horizons:
        # Give mount a couple of minutes to get to Jupiter and for A-P
        # HORIZONS to do its calibration dance
        log.info('Waiting for A-P HORIZONS app to finishing acquiring Jupiter')
        time.sleep(120)
    horizon_message = False
    while P.MD.horizon_limit():
        if not P.MD.Telescope.Tracking:
            log.error('Horizon limit reached.  Shutting down observatory.')
            P.MD.Application.ShutDownObservatory()
            return
        if not horizon_message:
            log.info('Waiting for object to rise above horizon limit')
            horizon_message = True
        time.sleep(5)
    d = args.dir
    if d is None:
        today = Time.now().fits.split('T')[0]
        d = os.path.join(raw_data_root, today)
    basename = args.basename
    #if basename is None:
    #    basename = 'IPT_Na_R_'
    if not P.MD.CCDCamera.GuiderRunning:
        # User could have had guider already on.  If not, center with
        # guider slews and start the guider
        log.debug('CENTERING WITH GUIDER SLEWS') 
        P.center_loop(max_tries=5)
        log.debug('STARTING GUIDER') 
        P.MD.guider_start(filter=3)
    log.debug('TURNING ON GUIDEBOX MOVER SYSTEM')
    P.diff_flex()
    log.debug('CENTERING WITH GUIDEBOX MOVES') 
    P.center_loop()
    if P.MD.horizon_limit():
        log.debug('Horizon limit reached.  Shutting down observatory')
        P.MD.Application.ShutDownObservatory()
        return
    log.info('Starting with R')
    P.MD.acquire_im(uniq_fname('R_', d),
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

        if P.MD.horizon_limit():
            log.debug('Horizon limit reached.  Shutting down observatory')
            P.MD.Application.ShutDownObservatory()
            return
        log.info('Collecting Na')
        P.MD.acquire_im(uniq_fname('Na_off-band_', d),
                        exptime=60,
                        filt=4)
        if P.MD.horizon_limit():
            log.debug('Horizon limit reached')
            return
        P.MD.acquire_im(uniq_fname('Na_on-band_', d),
                        exptime=300,
                        filt=2)
        if P.MD.horizon_limit():
            log.debug('Horizon limit reached.  Shutting down observatory')
            P.MD.Application.ShutDownObservatory()
            return
        log.debug('CENTERING WITH GUIDEBOX MOVES') 
        P.center_loop()
        
        for i in range(4):
            if P.MD.horizon_limit():
                log.debug('Horizon limit reached.  Shutting down observatory')
                P.MD.Application.ShutDownObservatory()
                return
            P.diff_flex()
            log.info('Collecting [SII]')
            P.MD.acquire_im(uniq_fname('SII_on-band_', d),
                            exptime=300,
                            filt=1)
            if P.MD.horizon_limit():
                log.debug('Horizon limit reached.  Shutting down observatory')
                P.MD.Application.ShutDownObservatory()
                return
            P.MD.acquire_im(uniq_fname('SII_off-band_', d),
                            exptime=60,
                            filt=3)
            if P.MD.horizon_limit():
                log.debug('Horizon limit reached.  Shutting down observatory')
                P.MD.Application.ShutDownObservatory()
                return
            P.diff_flex()
            log.debug('CENTERING WITH GUIDEBOX MOVES') 
            P.center_loop()
            if P.MD.horizon_limit():
                log.debug('Horizon limit reached.  Shutting down observatory')
                P.MD.Application.ShutDownObservatory()
                return
            log.info('Collecting R')
            P.MD.acquire_im(uniq_fname('R_', d),
                            exptime=2,
                            filt=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoIO-related instrument control and image reduction")
    # --> Update this with final list once settled
    subparsers = parser.add_subparsers(dest='one of the subcommands in {}', help='sub-command help')
    subparsers.required = True

    IPT_Na_R_parser =  subparsers.add_parser(
        'IPT_Na_R', help='Collect IPT, Na, and R measurements')
    IPT_Na_R_parser.add_argument(
        '--dir', help='directory, default current date in YYYY-MM-DD format')
    IPT_Na_R_parser.add_argument(
        '--basename', help='base filename for files, default = IPT_Na_R_')
    IPT_Na_R_parser.set_defaults(func=IPT_Na_R)

    # Final set of commands that makes argparse work
    args = parser.parse_args()
    # This check for func is not needed if I make subparsers.required = True
    if hasattr(args, 'func'):
        args.func(args)
