# Debugging
import matplotlib.pyplot as plt
from jpm_fns import display

import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta

from scipy import signal, ndimage



# So we have two things here to keep track of.  The 4 parameters that
# characterize the ND filter and the coordinates of the edges of the
# filter at a particular Y value.  Maybe the pos method could return
# either one, depending on whether or not a y coordinate is
# specified.  In that case, what I have as pos now should be
# measure_pos, or something.

def hist_of_im(im):
    """Returns histogram of image and index into centers of bins"""
    
    # Code from west_aux.py, maskgen.

    # Histogram bin size should be related to readnoise
    readnoise = 5
    hrange = (im.min(), im.max())
    nbins = int((hrange[1] - hrange[0]) / readnoise)
    hist, edges = np.histogram(im, bins=nbins,
                               range=hrange, density=True)
    # Convert edges of histogram bins to centers
    centers = (edges[0:-1] + edges[1:])/2
    #plt.plot(centers, hist)
    #plt.show()

    return(hist, centers)

class NDData:
    """Neutral Density Data object"""

    # ND filter position in case none is derivable from flats.  This is from:
    # print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit'))
    previous_params = ((-7.35537190e-02,  -6.71900826e-02), 
                   (1.24290909e+03,   1.34830909e+03))

    # And we can refine it further for a good Jupiter example
    #print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit',
    #                  initial_try=((-7.35537190e-02,  -6.71900826e-02), 
    #                               (1.24290909e+03,   1.34830909e+03))))
    previous_params = ((-6.57640346e-02,  -5.77888855e-02),
                       (1.23532221e+03,   1.34183584e+03))
    n_y_steps = 15
    x_filt_width = 25
    edge_mask = 5
    max_movement=50
    max_delta_pix=10

    im = None
    # The shape of im is really all we need to store for calculations
    im_shape = None
    fname = None
    params = None
    
    def get_im_and_shape(self, im=None):
        """Returns image, reading from fname, if necessary"""
        if im is None:
            if self.im is None:
                if self.fname is None:
                    log.info('No error, just saying that you have no image and asked for one.')
                    return(None)
                H = fits.open(self.fname)
                self.im = np.asfarray(H[0].data)
                H.close()
        else:
            self.im = im
        self.im_shape = self.im.shape
        return(self.im)

    def edges(self, y, external_params=None):
        """Returns x coords of ND filter edges at given y coordinate(s)"""
        if not external_params is None:
            params = external_params
        else:
            if self.params is None:
                self.get_params()
            params = self.params
        params = np.asarray(params)
        if np.asarray(y).size == 1:
            return(params[1,:] + params[0,:]*(y - self.im_shape[0]/2))
        es = []
        for this_y in y:
            es.append(params[1,:] + params[0,:]*(this_y - self.im_shape[0]/2))
        return(es)
        
    def get_params(self):
        """Returns parameters which characterize ND filter (currently 2 lines fir to edges)"""
        if not self.params is None:
            return(self.params)

        # If we made it here, we need to calculate params.  Take
        # n_y_steps and make profiles, take the gradient and absolute
        # value to spot the edges of the ND filter
        nd_edges = [] ; ypts = []
        y_bin = int(self.im_shape[0]/self.n_y_steps)
        yrange = np.arange(0, self.im_shape[0], y_bin)
        for ypt in yrange:
            subim = self.im[ypt:ypt+y_bin, :]
            profile = np.sum(subim, 0)
            smoothed_profile = signal.savgol_filter(profile, self.x_filt_width, 3)
            d = np.gradient(smoothed_profile, 3)
            s = np.abs(d)

            # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
            # points out same problem I had with with cwt.  It is too
            # sensitive to little peaks.  However, I can find the peaks
            # and just take the two largest ones
            peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
            # Need to change peak_idx into an array instead of a list for
            # indexing
            peak_idx = np.array(peak_idx)
            # If available, limit our search to the region max_movement
            # around initial_try.
            bounds = (0,s.size)
            if not self.previous_params is None:
                bounds = self.edges(ypt, self.previous_params) + np.asarray((-self.max_movement, self.max_movement))
                #bounds = previous_params[1,:] + previous_params[0,:]*(ypt - im.shape[0]/2) + np.asarray((-max_movement, max_movement))
                bounds = bounds.astype(int)
                goodc = np.where(np.logical_and(bounds[0] < peak_idx, peak_idx < bounds[1]))
                peak_idx = peak_idx[goodc]
                #print(peak_idx)
                #print(s[peak_idx])
                #plt.plot(s)
                #plt.show()
                if peak_idx.size < 2:
                    continue
    
            # Sort on value
            sorted_idx = np.argsort(s[peak_idx])
            # Unwrap
            peak_idx = peak_idx[sorted_idx]
    
            # Thow out if lower peak is too weak.  Use Carey Woodward's
            # trick of estimating the noise on the continuum To avoid
            # contamination, do this calc just over our desired interval
            ss = s[bounds[0]:bounds[1]]
    
            noise = np.std(ss[1:-1] - ss[0:-2])
            #print(noise)
            if s[peak_idx[-2]] < 3 * noise:
                #print("Rejected")
                continue
    
            # Find top two and put back in index order
            top_two = np.sort(peak_idx[-2:])
            # Accumulate in tuples
            nd_edges.append(top_two)
            ypts.append(ypt)
    
        nd_edges = np.asarray(nd_edges)
        ypts = np.asarray(ypts)
        if nd_edges.size < 2:
            if self.previous_params is None:
                raise ValueError('Not able to find ND filter position')
            log.warning('Unable to improve filter position over initial guess')
            return(previous_params)
        
        #plt.plot(ypts, nd_edges)
        #plt.show()
    
        # Fit lines to our points, making the origin the center of the image in Y
        params = np.polyfit(ypts-self.im_shape[0]/2, nd_edges, 1)
        params = np.asarray(params)
        
        # Check to see if there are any bad points, removing them and
        # refitting
        resid = nd_edges - self.edges(ypts, params)
        # Do this one side at a time, since the points might not be on
        # the same y level and it is not easy to zipper the coordinate
        # tuple apart in Python
        goodc0 = np.where(abs(resid[:,0]) < self.max_delta_pix)
        goodc1 = np.where(abs(resid[:,1]) < self.max_delta_pix)
        #print(ypts[goodc1]-self.im_shape[0]/2)
        #print(nd_edges[goodc1, 1])
        if len(goodc0) < resid.shape[1]:
            params[:,0] = np.polyfit(ypts[goodc0]-self.im_shape[0]/2,
                                     nd_edges[goodc0, 0][0], 1)
        if len(goodc1) < resid.shape[1]:
            params[:,1] = np.polyfit(ypts[goodc1]-self.im_shape[0]/2,
                                     nd_edges[goodc1, 1][0], 1)
        #print(params)
        # Check parallelism by calculating shift of ends relative to each other
        dp = abs((params[0,1] - params[0,0]) * self.im_shape[0]/2)
        if dp > self.max_delta_pix:
            txt = 'ND filter edges are not parallel.  Edges are off by ' + str(dp) + ' pixels.'
            # DEBUGGING
            print(txt)
            plt.plot(ypts, nd_edges)
            plt.show()
    
            if previous_params is None:
                raise ValueError(txt)
            log.warning(txt + ' Returning initial try.')
            params = previous_params

        self.params = params

        #print(self.params)
        return(self.params)

    def coords(self):
        """Returns coordinates of ND filter in im given an ND_filt_pos"""
        if self.params is None:
            self.get_params()
        
        xs = [] ; ys = []
        for iy in np.arange(0, self.im_shape[0]):
            bounds = self.params[1,:] + self.params[0,:]*(iy - self.im_shape[0]/2) + np.asarray((self.edge_mask, -self.edge_mask))
            bounds = bounds.astype(int)
            for ix in np.arange(bounds[0], bounds[1]):
                xs.append(ix)
                ys.append(iy)
    
        # NOTE C order and the fact that this is a tuple of tuples
        return((ys, xs))

    def imshow(self):
        if self.im is None:
            self.get_im_and_shape()
        plt.imshow(self.im)
        plt.show()

    def __init__(self, im_or_fname=None, nd_pos=None,
                 default_nd_pos=previous_params,
                 n_y_steps=n_y_steps, x_filt_width=x_filt_width,
                 edge_mask=edge_mask):
        if im_or_fname is None:
            log.info('No error, just saying that you have no image.')
        elif isinstance(im_or_fname, str):
            self.fname = im_or_fname
            self.get_im_and_shape()
        else:
            self.get_im_and_shape(im_or_fname)
        self.nd_pos = nd_pos
        self.default_nd_pos = default_nd_pos
        self.n_y_steps = n_y_steps
        self.x_filt_width = x_filt_width
        

    #def pos(self, default_nd_pos=self.default_nd_pos,):
        
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
            return(str(width_nd_pos * length_nd_pos))
        else:
            raiseValueError('Use True False or None in variable')
        
    def perimeter(self, width_nd_pos,  length_nd_pos, variable=True):
        if variable is False or variable is None:
            print('The perimeter of the netral density filter is ' +
                  str(width_nd_pos * 2 + 2 *  length_nd_pos) +  '.')
        elif variable is True:
            return(str(width_nd_pos * 2 + 2 *  length_nd_pos) +  '.')
        else:
            raiseValueError('Use True False or None in variable')
            
    def VS(self,v1,value1,v2,value2,v3,value3):
        v1=value1 ; v2=value2 ; v3=value3
        return(v1, v2, v3)


def get_jupiter_center(im_or_fname, y=None):
    """Returns two vectors, the center of Jupiter (whether or not Jupiter
    is on ND filter) and the desired position of Jupiter, assuming we
    want it to be centered on the ND filter at a postion y.  If y not
    specified, y center of image is used."""

    if isinstance(im_or_fname, str):
        H = fits.open(im_or_fname)
        im = np.asfarray(H[0].data)
        H.close()
    else:
        im = im_or_fname
    
    if y is None:
        y = im.shape[0]/2
    # Get our neutral density filter object
    ND = NDData(im)
    ND_center = (np.average(ND.edges(y)), y)
    
    # Use the histogram technique to spot the bias level of the image.
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.  The ND filter provides a
    # similar peak after bias subutraction (or, rather, it is the
    # second such peak)
    im_hist, im_hist_centers = hist_of_im(im)
    im_peak_idx = signal.find_peaks_cwt(im_hist, np.arange(10, 50))
    im -= im_hist_centers[im_peak_idx[0]]

    # Check to see if Jupiter is sticking out significantly from
    # behind the ND filter, in which case we are better off just using
    # the center of mass of the image and calling that good enough
    #print(np.sum(im))
    if np.sum(im) > 1E9: 
        y_x = ndimage.measurements.center_of_mass(im)
        return(y_x[::-1], ND_center)

    # Get the coordinates of the ND filter
    NDc = ND.coords()

    # Filter those by ones that are at least 1 std above the median
    boostc = np.where(im[NDc] > (np.median(im[NDc]) + np.std(im[NDc])))
    boost_NDc0 = np.asarray(NDc[0])[boostc]
    boost_NDc1 = np.asarray(NDc[1])[boostc]
    # Here is where we boost what is sure to be Jupiter, if Jupiter is
    # in the ND filter
    im[boost_NDc0, boost_NDc1] *= 1000
    y_x = ndimage.measurements.center_of_mass(im)
    #print(y_x[::-1])
    #plt.imshow(im)
    #plt.show()
    return(y_x[::-1], ND_center)

def guide_calc(x1, y1, fits_t1=None, x2=None, y2=None, fits_t2=None, guide_dt=10, guide_dx=0, guide_dy=0, last_guide=None, aggressiveness=0.5, target_c = np.asarray((1297, 1100))):
    """ Calculate offset guider values given pixel and times"""

    # Pixel scales in arcsec/pix
    main_scale = 1.59/2
    guide_scale = 4.42
    typical_expo = 385 * u.s
    
    if last_guide == None:
        guide_dt = guide_dt * u.s
        previous_dc_dt = np.asarray((guide_dx, guide_dy)) / guide_dt
    else:
        guide_dt = last_guide[0]
        previous_dc_dt = np.asarray((last_guide[1], last_guide[2])) / guide_dt

    # Convert input time interval to proper units
    
    # time period to use in offset guide file
    new_guide_dt = 10 * u.s

    if fits_t1 == None:
        t1 = Time('2017-01-01T00:00:00', format='fits')
    else:
        t1 = Time(fits_t1, format='fits')
    if fits_t2 == None:
        # Take our typical exposure time to settle toward the center
        t2 = t1 + typical_expo
    else:
        if fits_t1 == None:
            raise ValueError('fits_t1 given, but fits_t1 not supplied')
        t2 = Time(fits_t2, format='fits')
    dt = (t2 - t1) * 24*3600 * u.s / u.day

    c1 = np.asarray((x1, y1))
    c2 = np.asarray((x2, y2))
    
    if x2 == None and y2 == None:
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

    return(new_guide_dt, r[0], r[1])


print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# 
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))
# 
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# 
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
# print(nd_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))
#ND=NDData('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit')
#print(ND.get_params())

#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
