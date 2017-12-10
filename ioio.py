
#from jpm_fns import display
# for testing
import os
import define as D
import time


import sys
import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from scipy import signal, ndimage

##!## # These are needed for MaxImData
##!## import win32com.client
##!## import time
##!## 
##!## 
##!## # So we have two things here to keep track of.  The 4 parameters that
##!## # characterize the ND filter and the coordinates of the edges of the
##!## # filter at a particular Y value.  Maybe the pos method could return
##!## # either one, depending on whether or not a y coordinate is
##!## # specified.  In that case, what I have as pos now should be
##!## # measure_pos, or something.
##!## 
##!## 
##!## #Daniel
##!## if True:
##!##     class MakeList():
##!##         def __init__(self, list_, item, makelist=True):
##!##             if makelist is True:
##!##                 list_=[]
##!##         def append(self, item):
##!##             list_.append(item)
##!## #Daniel
##!## 
##!## 
##!## class MaxImData():
##!##     """Stores data related to controlling MaxIm DL via ActiveX/COM events.
##!## 
##!##     Notes: 
##!## 
##!##     MaxIm camera, guide camera, and telescope must be set up properly
##!##     first (e.g. you have used the setup for interactive observations).
##!##     Even so, the first time this is run, keep an eye out for MaxIm
##!##     dialogs, as this program will hang until they are answered.  To
##!##     fix this, a wathdog timer would need to be used.
##!## 
##!##     Technical note for downstreeam object use: we don't have access to
##!##     the MaxIm CCDCamera.ImageArray, but we do have access to similar
##!##     information (and FITS keys) in the Document object.  The CCDCamera
##!##     object is linked to the actual last image read, where the Document
##!##     object is linked to the currently active window.  This means the
##!##     calling routine could potentially expect the last image read in
##!##     but instead get the image currently under focus by the user.  The
##!##     solution to this is to (carefully) use notify events to interrupt
##!##     MaxIm precisely when the event you expect happens (e.g. exposure
##!##     or guide image acuired).  Then you are sure the Document object
##!##     has the info you expect.  Beware that while you have control,
##!##     MaxIm is stuck and back things may happen, like the guider might
##!##     get lost, etc.  If your program is going to take a long time to
##!##     work with the information it just got, figure out a way to do so
##!##     asynchronously
##!## 
##!##     """
##!## 
##!##     def __init__(self):
##!##         # Create containers for all of the objects that can be
##!##         # returned by MaxIm.  We'll only populate them when we need
##!##         # them.  Some of these we may never use or write code for
##!##         self.Application = None
##!##         self.CCDCamera = None
##!##         self.Document = None
##!##         
##!##         # There is no convenient way to get the FITS header from MaxIm
##!##         # unless we write the file and read it in.  Instead allow for
##!##         # getting a selection of FITS keys to pass around in a
##!##         # standard astropy fits HDUList
##!##         self.FITS_keys = None
##!##         self.HDUList = None
##!##         self.required_FITS_keys = ('DATE-OBS', 'EXPTIME', 'EXPOSURE', 'XBINNING', 'YBINNING', 'XORGSUBF', 'YORGSUBF', 'FILTER', 'IMAGETYP', 'OBJECT')
##!## 
##!##         # Maxim doesn't expose the results of this menu item from the
##!##         # Guider Settings Advanced tab in the object.  It's for
##!##         # 'scopes that let you push both RA and DEC buttons at once
##!##         # for guider movement
##!##         self.simultaneous_guide_corrections = True
##!##         # We can use the CCDCamera.GuiderMaxMove[XY] property for an
##!##         # indication of how long it is safe to press the guider
##!##         # movement buttons
##!##         self.guider_max_move_multiplier = 20
##!## 
##!##         # The conversion between guider button push time and guider
##!##         # pixels is stored in the CCDCamera.Guider[XY]Speed
##!##         # properties.  Plate scales in arcsec/pix are not, though they
##!##         # can be greped out of FITS headers. 
##!## 
##!##         # Main camera plate solve, binned 2x2:
##!##         # RA 12h 55m 33.6s,  Dec +03° 27' 42.6"
##!##         # Pos Angle +04° 34.7', FL 1178.9 mm, 1.59"/Pixel
##!##         self.main_plate = 1.59/2 # arcsec/pix
##!##         self.main_angle = 4.578333333333333 # CCW from N on east side of pier
##!## 
##!##         # Guider (Binned 1x1)
##!##         # RA 07h 39m 08.9s,  Dec +34° 34' 59.0"
##!##         # Pos Angle +178° 09.5', FL 401.2 mm, 4.42"/Pixel
##!##         self.guider_plate = 4.42
##!##         self.guider_angle = 178+9.5/60 - 180
##!## 
##!##         # This is a function that returns two vectors, the current
##!##         # center of the object in the main camera and the desired center 
##!##         #self.get_object_center = None
##!## 
##!##     def getApplication(self):
##!##         if self.Application is not None:
##!##             return True
##!##         try:
##!##             self.Application = win32com.client.Dispatch("MaxIm.Application")
##!##         except:
##!##             raise EnvironmentError('Error creating MaxIM application object.  Is MaxIM installed?')
##!##         # Catch any other weird errors
##!##         return isinstance(self.Application, win32com.client.CDispatch)
##!##         
##!##     def getCCDCamera(self):
##!##         if self.CCDCamera is not None:
##!##             return True
##!##         try:
##!##             self.CCDCamera = win32com.client.Dispatch("MaxIm.CCDCamera")
##!##         except:
##!##             raise EnvironmentError('Error creating CCDCamera object.  Is there a CCD Camera set up in MaxIm?')
##!##         # Catch any other weird errors
##!##         return isinstance(self.CCDCamera, win32com.client.CDispatch)
##!## 
##!##     def getDocument(self):
##!##         """Gets the document object of the current window"""
##!##         # The CurrentDocument object gets refreshed when new images
##!##         # are taken, so all we need is to make sure we are connected
##!##         # to begin with
##!##         if self.Document is not None:
##!##             return True
##!##         self.getApplication()
##!##         try:
##!##             self.Document = self.Application.CurrentDocument
##!##         except:
##!##             raise EnvironmentError('Error retrieving document object')
##!##         # Catch any other weird errors
##!##         return isinstance(self.Document, win32com.client.CDispatch)
##!## 
##!##     def connect(self):
##!##         """Link to telescope, CCD camera(s), filter wheels, etc."""
##!##         self.getApplication()
##!##         self.Application.TelescopeConnected = True
##!##         if self.Application.TelescopeConnected == False:
##!##             raise EnvironmentError('Link to telescope failed.  Is the power on to the mount?')
##!##         self.getCCDCamera()
##!##         self.CCDCamera.LinkEnabled = True
##!##         if self.CCDCamera.LinkEnabled == False:
##!##             raise EnvironmentError('Link to camera hardware failed.  Is the power on to the CCD (including any connection hardware such as USB hubs)?')
##!## 
##!##     def guider_move(self, ddec_dra, dec=None):
##!##         """Moves the telescope using guider slews.  ddec_dra is a tuple with
##!##         values in arcsec.  NOTE ORDER OF COORDINATES: Y, X to conform
##!##         to C ordering of FITS images"""
##!##         self.connect()
##!##         if dec is None:
##!##             dec = self.CCDCamera.GuiderDeclination
##!##         ddec = ddec_dra[0]
##!##         dra = ddec_dra[1]
##!##         # Change to rectangular tangential coordinates for small deltas
##!##         dra = dra*np.cos(np.radians(dec))
##!##         # The guider motion is calibrated in pixels per second, with
##!##         # the guider angle applied separately.  We are just moving in
##!##         # RA and DEC, so we don't need to worry about the guider angle
##!##         dpix = np.asarray((dra, ddec)) / self.guider_plate
##!##         # Multiply by speed, which is in pix/sec
##!##         dt = dpix / np.asarray((self.CCDCamera.GuiderXSpeed, self.CCDCamera.GuiderYSpeed))
##!##         
##!##         # Do a sanity check to make sure we are not moving too much
##!##         max_t = (self.guider_max_move_multiplier *
##!##                  np.asarray((self.CCDCamera.GuiderMaxMoveX, 
##!##                              self.CCDCamera.GuiderMaxMoveY)))
##!##             
##!##         if np.any(np.abs(dt) > max_t):
##!##             #print(str((dra, ddec)))
##!##             #print(str(np.abs(dt)))
##!##             log.warning('requested move of ' + str((dra, ddec)) + ' arcsec translates into move times of ' + str(np.abs(dt)) + ' seconds.  Limiting move in one or more axes to max t of ' + str(max_t))
##!##             dt = np.minimum(max_t, abs(dt)) * np.sign(dt)
##!##             
##!##         log.info('Seconds to move guider in DEC and RA: ' + str(dt))
##!##         if dt[0] > 0:
##!##             RA_success = self.CCDCamera.GuiderMove(0, dt[0])
##!##         elif dt[0] < 0:
##!##             RA_success = self.CCDCamera.GuiderMove(1, -dt[0])
##!##         else:
##!##             # No need to move
##!##             RA_success = True
##!##         # Wait until move completes if we can't push RA and DEC
##!##         # buttons simultaneously
##!##         while not self.simultaneous_guide_corrections and self.CCDCamera.GuiderMoving:
##!##                 time.sleep(0.1)
##!##         if dt[1] > 0:
##!##             DEC_success = self.CCDCamera.GuiderMove(2, dt[1])
##!##         elif dt[1] < 0:
##!##             DEC_success = self.CCDCamera.GuiderMove(3, -dt[1])
##!##         else:
##!##             # No need to move
##!##             DEC_success = True
##!##         while self.CCDCamera.GuiderMoving:
##!##             time.sleep(0.1)
##!##         return RA_success and DEC_success
##!## 
##!##     def rot(self, vec, theta):
##!##         """Rotates vector counterclockwise by theta degrees IN A
##!##         TRANSPOSED COORDINATE SYSTEM Y,X"""
##!##         # This is just a standard rotation through theta on X, Y, but
##!##         # when we transpose, theta gets inverted
##!##         theta = -np.radians(theta)
##!##         c, s = np.cos(theta), np.sin(theta)
##!##         #print(vec)
##!##         #print(theta, c, s)
##!##         M = np.matrix([[c, -s], [s, c]])
##!##         rotated = np.asarray(np.dot(M, vec))
##!##         #print(rotated)
##!##         return np.squeeze(rotated)
##!## 
##!##     def calc_main_move(self, current_pos, desired_center=None):
##!##         """Returns vector [ddec, dra] in arcsec to move scope to
##!##         center object at current_pos, where the current_pos and
##!##         desired_centers are vectors expressed in pixels on the main
##!##         camera in the astropy FITS Pythonic order Y, X"""
##!## 
##!##         self.connect()
##!##         if desired_center is None:
##!##             # --> WARNING!  This uses the current CCD image, not the
##!##             # --> Document object
##!##             desired_center = \
##!##             np.asarray((self.CCDCamera.StartY + self.CCDCamera.NumY, 
##!##                         self.CCDCamera.StartX + self.CCDCamera.NumX)) / 2.
##!##         dpix = np.asarray(desired_center) - np.asarray(current_pos)
##!##         dpix = self.rot(dpix, self.main_angle)
##!##         return dpix * self.main_plate
##!## 
##!##     def get_keys(self):
##!##         """Gets list of self.required_FITS_keys from current image"""
##!##         self.FITS_keys = []
##!##         for k in self.required_FITS_keys:
##!##             self.FITS_keys.append((k, self.Document.GetFITSKey(k)))
##!##         
##!##     # --> Not sure if FITS files get written by MaxIm before or after
##!##     # --> they become Document property after a fresh read.  Could use
##!##     # --> CCDCamera version, but this keeps it consistent
##!##     def set_keys(self, keylist):
##!##         """Write desired keys to current image FITS header"""
##!##         if not getDocument():
##!##             log.warning('Cannot get Document object, no FITS keys set')
##!##             return None
##!##         if self.HDUList is None:
##!##             log.warning('Asked to set_keys, but no HDUList is empty')
##!##             return None
##!##             
##!##         try:
##!##             h = self.HDUList[0].header
##!##             for k in keylist:
##!##                 if h.get(k):
##!##                     # Not sure how to get documentation part written
##!##                     self.Document.SetFITSKey(k, h[k])
##!##         except:
##!##             log.warning('Problem setting keys: ', sys.exc_info()[0])
##!##             return None
##!##             
##!## 
##!##     def get_im(self):
##!##         """Puts current MaxIm image (the image with focus) into a FITS HDUList.  If an exposure is being taken or there is no image, the im array is set equal to None"""
##!##         self.connect()
##!##         # Clear out HDUList in case we fail
##!##         self.HDUList = None
##!##         if not self.CCDCamera.ImageReady:
##!##             log.info('CCD Camera image is not ready')
##!##             return None
##!##         # For some reason, we can't get at the image array or its FITS
##!##         # header through CCDCamera.ImageArray, but we can through
##!##         # Document.ImageArray
##!##         if not self.getDocument():
##!##             log.info('There is no open image')
##!##             return None
##!##         
##!##         # Make sure we have an array to work with
##!##         c_im = self.Document.ImageArray
##!##         if c_im is None:
##!##             log.info('There is no image array')
##!##             return None
##!##         # Create a basic FITS image out of this and copy in the FITS
##!##         # keywords we want
##!## 
##!##         # TRANSPOSE ALERT.  Document.ImageArray returns a tuple of
##!##         # tuples shaped just how you would want it for X, Y.  Since
##!##         # Python is written in C, this is stored in memory in in "C
##!##         # order," which is the transpose of how they were intended to
##!##         # be written into a FITS file.  Since all the FITS stuff
##!##         # assumes that we are reading/writing FORTRAN-ordered arrays
##!##         # bytes from/to a C language, we need to transpose our array
##!##         # here so that the FITS stuff has the bytes in the order it
##!##         # expects.  This seems less prone to generating bugs than
##!##         # making users remember what state of transpose they are in
##!##         # when dealing with arrays generated here vs. data read in
##!##         # from disk for debugging routines.  This is also faster than
##!##         # writing to disk and re-reading, since the ndarray order='F'
##!##         # doesn't actually do any movement of data in memory, it just
##!##         # tells numpy how to interpret the order of indices.
##!##         c_im = np.asarray(c_im)
##!##         adata = c_im.flatten()#order='K')# already in C order in memory
##!##         # The [::-1] reverses the indices
##!##         adata = np.ndarray(shape=c_im.shape[::-1],
##!##                            buffer=adata, order='F')
##!##         
##!##         hdu = fits.PrimaryHDU(adata)
##!##         self.get_keys()
##!##         for k in self.FITS_keys:
##!##             hdu.header[k[0]] = k[1]
##!##         self.HDUList = fits.HDUList(hdu)
##!##         return self.HDUList
##!## 
##!##     ## --> This is in the wrong place in terms of abstraction 
##!##     #def center_object(self):
##!##     #    if self.get_im() is None:
##!##     #        log.warning('No image')
##!##     #        return None
##!##     #    #self.guider_move(self.calc_main_move(self.get_object_center_pix()))
##!##     #    #obj_c = get_jupiter_center(self.HDUList)
##!##     #    obj_c = self.ObsData.obj_center(self.HDUList)
##!##     #    log.info('object center = ', obj_c)
##!##     #    self.guider_move(self.calc_main_move(obj_c))


class ObsData():
    """Base class for observations, enabling object centering, etc.

    This is intended to work in an active obsering setting, so
    generally an image array will be received, the desired properties
    will be calculated from it and those properties will be read by
    the calling code.

    """

    def __init__(self, HDUList_im_or_fname=None):
        if HDUList_im_or_fname is None:
            raise ValueError('No HDUList_im_or_fname provided')
        # Set up our basic FITS image info
        self.fname = None
        self._binning = None
        self._subframe_origin = None
        self._we_opened_file = None
        # This is the most basic observation data to be tracked
        self._obj_center = None
        self._desired_center = None
        # Make the guts of __init__ methods that can be overridden
        # Read our image
        self.read_im(HDUList_im_or_fname)
        # Populate our object
        self.populate_obj()
        self.cleanup()
        del HDUList_im_or_fname
        
    def populate_obj(self):
        """Calculate quantities that will be stored long-term in object"""
        # Note that if MaxIm is not configured to write IRAF-complient
        # keywords, IMAGETYP gets a little longer and is capitalized
        # http://diffractionlimited.com/wp-content/uploads/2016/11/sbfitsext_1r0.pdf
        h = self.HDUList[0].header
        kwd = h['IMAGETYP'].upper()
        if 'DARK' in kwd or 'BIAS' in kwd or 'FLAT' in kwd:
            raise ValueError('Not able to process IMAGETYP = ' + h['IMAGETYP'])
        # Do our work & leave the results in the property
        self.obj_center
        self.desired_center

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
        # --> These should potentially be issubclass
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
            try:
                h = HDUList[0].header
                # Note Astropy Pythonic transpose Y, X order
                self._binning = (h['YBINNING'], h['XBINNING'])
                self._binning = np.asarray(self._binning)
                # This is in binned coordinates
                self._subframe_origin = (h['YORGSUBF'], h['XORGSUBF'])
                self._subframe_origin = np.asarray(self._subframe_origin)
            except:
                log.warning('Could not read binning or subframe origin from image header.  Did you pass a valid MaxIm-recorded image and header?  Assuming binning = 1, subframe_origin = 0,0')
                self._binning = (1,1)
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
        '''Unbin primary HDU image
        '''
        a = self.HDUList[0].data
        # Don't bother if we are already unbinned
        if np.sum(self._binning) == 2:
            return a
        newshape = self._binning * a.shape
        # From http://scipy-cookbook.readthedocs.io/items/Rebinning.html
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
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

    def hist_of_im(self, im, readnoise):
        """Returns histogram of image and index into centers of bins.  
Uses readnoise (default = 5 e- RMS) to define bin widths"""
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
    
        
    def imshow(self, im=None):
        if im is None:
            im = self.HDUList[0].data
        plt.imshow(im)
        plt.show()

    @property
    def obj_center(self):
        """This is a really crummy object finder, since it will be confused
        by cosmic ray hits.  It is up to the user to define an object
        center finder that suits them, for instance one that uses
        PinPoint astrometry
        NOTE: The return order of indices is astropy FITS Pythonic: Y,
        X in unbinned coordinates from the ccd origin."""
    
        if self._obj_center is not None:
            return self._obj_center
        im = self.HDUList[0].data
        im_center = np.unravel_index(np.argmax(im), im.shape)
        self._obj_center = self.unbinned(im_center)
        return self._obj_center

    @property
    def desired_center(self):
        """Returns geometric center of image.
        NOTE: The return order of indices is astropy FITS Pythonic: Y, X"""
        if self._desired_center is not None:
            return self._desired_center
        im = self.HDUList[0].data
        im_center = np.asarray(im.shape)/2
        self._desired_center = self.unbinned(im_center)
        return self._desired_center

class CorObsData(ObsData):
    """Object for containing coronagraph image data used for centering Jupiter"""

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
        h = self.HDUList[0].header
        kwd = h['IMAGETYP'].upper()
        if 'DARK' in kwd or 'BIAS' in kwd:
            raise ValueError('Not able to process IMAGETYP = ' + h['IMAGETYP'])
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
        if h.get('NDPAR00') is not None:
            ND_params = np.zeros((2,2))
            # Note transpose, since we are working in C!
            ND_params[0,0] = h['NDPAR00']
            ND_params[1,0] = h['NDPAR01']
            ND_params[0,1] = h['NDPAR10']
            ND_params[1,1] = h['NDPAR11']
            self._ND_params = ND_params
        else:
            if not self.isflat and self.default_ND_params is None:
                raise ValueError('Set default_ND_params before attempting a run on a non-flat exposure')
                
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
        im = im - self.back_level(im)
        
        # Check to see if Jupiter is sticking out significantly from
        # behind the ND filter, in which case we are better off just using
        # the center of mass of the image and calling that good enough
        #print(np.sum(im))
        # --> This may need some improvement
        #if np.sum(im) > 1E9:
        #    log.warning('Jupiter outside of ND filter')
        #    y_x = np.asarray(ndimage.measurements.center_of_mass(im))
        #    self._obj_center = y_x
        #    return self._obj_center
        
        # Get the coordinates of the ND filter
        NDc = self.ND_coords

        # Come up with a metric for when Jupiter is in the ND filter.
        # Below is my scratch work        
        # Rj = np.asarray((50.1, 29.8))/2. # arcsec
        # plate = 1.59/2 # main "/pix
        # 
        # Rj/plate # Jupiter pixel radius
        # array([ 31.50943396,  18.74213836])
        # 
        # np.pi * (Rj/plate)**2 # Jupiter area
        # array([ 3119.11276312,  1103.54018437])
        # 
        # np.pi * (Rj/plate)**2 * 5 # Jupiter 5 sigma
        # array([ 15595.56381559,   5517.70092183])

        med_ND = np.median(im[NDc])
        std_ND = np.std(im[NDc])
        obj_ND_sigma = np.sum(im[NDc] - med_ND) / (std_ND * len(NDc))

        if obj_ND_sigma < 5000:
            log.warning('Jupiter outside of ND filter')
            med = np.median(im)
            # Outside the ND filter, Jupiter should be saturating, but
            # be conservative about how many pixels
            if np.sum(im - med) < 65000 * 250:
                raise ValueError('Jupiter not found in image')
            # If we made it here, Jupiter is outside the ND filter,
            # but shining bright enough to be found
            y_x = np.asarray(ndimage.measurements.center_of_mass(im))
            self._obj_center = y_x
            return self._obj_center
            
        # Filter ND coords for ones that are at least 1 std above the median
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
        #return (y_x[::-1], ND_center)
        # Stay in Pythonic y, x coords
        self._obj_center = np.asarray(y_x)
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
        
    @property
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter"""
        # Work with unbinned image
        im = self.HDU_unbinned()
        
        xs = [] ; ys = []
        for iy in np.arange(0, im.shape[0]):
            bounds = self.ND_params[1,:] + self.ND_params[0,:]*(iy - im.shape[0]/2) + np.asarray((self.edge_mask, -self.edge_mask))
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

        # NOTE C order and the fact that this is a tuple of tuples
        return (ys, xs)

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
        d = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)**0.5
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
                = signal.medfilt(im[ytop:ybot, x0:x1], \
                                 kernel_size=3)
        im = self.HDU_unbinned()
        h = self.HDUList[0].header

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
                profile = np.sum(im[ypt_top:ypt_top+y_bin, \
                                    bounds[0]:bounds[1]], 0)
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
                default_ND_width = self.default_ND_params[1,1] - \
                                   self.default_ND_params[1,0]
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
                        = im[ypt_top+rowpt, \
                             this_ND_center-subim_hw: \
                             this_ND_center+subim_hw]

                profile = np.sum(subim, 0)
                # This spots the sharp edge of the filter surprisingly
                # well, though the resulting peaks are a little fat
                # (see signal.find_peaks_cwt arguments, below)
                smoothed_profile = signal.savgol_filter(profile, self.x_filt_width, 0)
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
            peak_idx = signal.find_peaks_cwt(s, \
                                             self.cwt_width_arange, \
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
                if de < self.max_ND_width_range[0] or \
                   de > self.max_ND_width_range[1]:
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
                        diff_arr.append((ip, iop, peak_idx[iop] - peak_idx[ip]))
                diff_arr = np.asarray(diff_arr)
                closest = np.abs(diff_arr[:,2] - default_ND_width)
                sorted_idx = np.argsort(closest)
                edge_idx = peak_idx[diff_arr[sorted_idx[0], 0:2]]
                # Sanity check
                de = edge_idx[1] - edge_idx[0]
                if de < self.max_ND_width_range[0] or \
                   de > self.max_ND_width_range[1]:
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
                this_default_ND_center = np.round(np.mean(self.ND_edges(ypts[iy], self.default_ND_params)))
                cshift = int(this_default_ND_center - im.shape[1]/2.)
                es.append(ND_edges[iy,:] + cshift)

                #es.append(self.default_ND_params[1,:] - im.shape[1]/2. + self.default_ND_params[0,:]*(this_y - im.shape[0]/2))
            ND_edges =  np.asarray(es)

        if self.plot_ND_edges:
            plt.plot(ypts, ND_edges)
            plt.show()
        

        # Try an iterative approach to fitting lines to the ND_edges
        ND_edges = np.asarray(ND_edges)
        ND_params0 = self.iter_linfit(ypts-im.shape[0]/2, ND_edges[:,0], self.max_fit_delta_pix)
        ND_params1 = self.iter_linfit(ypts-im.shape[0]/2, ND_edges[:,1], self.max_fit_delta_pix)
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
        h['NDPAR00'] = (ND_params[0,0], 'ND filt left side slope at Y center of im')
        h['NDPAR01'] = (ND_params[1,0], 'ND filt left side offset at Y center of im')
        h['NDPAR10'] = (ND_params[0,1], 'ND filt right side slope at Y center of im')
        h['NDPAR11'] = (ND_params[1,1], 'ND filt right side offset at Y center of im')

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

log.setLevel('INFO')
#log.setLevel('WARNING')

F = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit')#, plot_dprof=True, plot_ND_edges=True)
default_ND_params = F.ND_params
D.say(default_ND_params)
F2 = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
default_ND_params = F2.ND_params
D.say(default_ND_params)
F3 = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
default_ND_params = F3.ND_params
D.say(default_ND_params)
O = CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0014_SII_on-band.fit', default_ND_params=default_ND_params) #plot_prof=True, plot_dprof=True, plot_ND_edges=True, 
D.say(O.ND_params)
D.say(O.obj_center)
D.say(O.desired_center)
D.say(O.obj_center - O.desired_center)
D.say(O.obj_to_ND)
O = CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0028_moving_to_SII_on-band.fit', default_ND_params=default_ND_params)#, plot_prof=True, plot_dprof=True, plot_ND_edges=True)
D.say(O.ND_params)
D.say(O.obj_center)
D.say(O.desired_center)
D.say(O.obj_center - O.desired_center)
D.say(O.obj_to_ND)


##start = time.time()
##rawdir = '/data/io/IoIO/raw/2017-05-28'
##raw_fnames = [os.path.join(rawdir, f)
##              for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
##
##for f in sorted(raw_fnames):
##    D.say(f)
##    try:
##        O4 = CorObsData(f, n_y_steps=4, default_ND_params=default_ND_params)
##        O8 = CorObsData(f, n_y_steps=8, default_ND_params=default_ND_params)
##        O16 = CorObsData(f, n_y_steps=16, default_ND_params=default_ND_params)
##        D.say(O8.obj_to_ND)
##        dc4 = np.abs(O4.desired_center - O16.desired_center)
##        dc8 = np.abs(O8.desired_center - O16.desired_center)
##        #if (dc4 > 5).any() or (dc8 > 5).any():
##        #D.say(O.obj_center - O.desired_center)
##        print(O4.desired_center - O16.desired_center)
##        print(O4.obj_center - O16.obj_center)
##        print('4^---8>-----')
##        print(O8.desired_center - O16.desired_center)
##        print(O8.obj_center - O16.obj_center)
##    except ValueError as e:
##        log.error('Skipping: ' + str(e))
##
##    
##end = time.time()
##D.say('Elapsed time: ' + str(D.say(end - start)) + 's')

#Na =   CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0007_Na_off-band.fit')
#print(Na.obj_center)
#print(Na.desired_center)
#print(Na.obj_center - Na.desired_center)

#print(Na.get_ND_params())
#print(Na.ND_params)
#print(Na.ND_angle)
#Na.ND_params = (((3.75447820e-01,  3.87551301e-01), \
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


#ND=NDData('//snipe/data/io/IoIO/raw/2017-05-29/Sky_Flat-0001_Na_off-band.fit')
#print(ND.get_ND_params())
if __name__ == "__main__":
    flat = CorObsData('/Users/jpmorgen/byted/xfr/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
    #flat.imshow()
    flat.get_ND_params()
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
