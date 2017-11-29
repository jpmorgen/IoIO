
#from jpm_fns import display

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
##!##             return(True)
##!##         try:
##!##             self.Application = win32com.client.Dispatch("MaxIm.Application")
##!##         except:
##!##             raise EnvironmentError('Error creating MaxIM application object.  Is MaxIM installed?')
##!##         # Catch any other weird errors
##!##         return(isinstance(self.Application, win32com.client.CDispatch))
##!##         
##!##     def getCCDCamera(self):
##!##         if self.CCDCamera is not None:
##!##             return(True)
##!##         try:
##!##             self.CCDCamera = win32com.client.Dispatch("MaxIm.CCDCamera")
##!##         except:
##!##             raise EnvironmentError('Error creating CCDCamera object.  Is there a CCD Camera set up in MaxIm?')
##!##         # Catch any other weird errors
##!##         return(isinstance(self.CCDCamera, win32com.client.CDispatch))
##!## 
##!##     def getDocument(self):
##!##         """Gets the document object of the current window"""
##!##         # The CurrentDocument object gets refreshed when new images
##!##         # are taken, so all we need is to make sure we are connected
##!##         # to begin with
##!##         if self.Document is not None:
##!##             return(True)
##!##         self.getApplication()
##!##         try:
##!##             self.Document = self.Application.CurrentDocument
##!##         except:
##!##             raise EnvironmentError('Error retrieving document object')
##!##         # Catch any other weird errors
##!##         return(isinstance(self.Document, win32com.client.CDispatch))
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
##!##         return(RA_success and DEC_success)
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
##!##         return(np.squeeze(rotated))
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
##!##         return(dpix * self.main_plate)
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
##!##             return(None)
##!##         if self.HDUList is None:
##!##             log.warning('Asked to set_keys, but no HDUList is empty')
##!##             return(None)
##!##             
##!##         try:
##!##             h = self.HDUList[0].header
##!##             for k in keylist:
##!##                 if h.get(k):
##!##                     # Not sure how to get documentation part written
##!##                     self.Document.SetFITSKey(k, h[k])
##!##         except:
##!##             log.warning('Problem setting keys: ', sys.exc_info()[0])
##!##             return(None)
##!##             
##!## 
##!##     def get_im(self):
##!##         """Puts current MaxIm image (the image with focus) into a FITS HDUList.  If an exposure is being taken or there is no image, the im array is set equal to None"""
##!##         self.connect()
##!##         # Clear out HDUList in case we fail
##!##         self.HDUList = None
##!##         if not self.CCDCamera.ImageReady:
##!##             log.info('CCD Camera image is not ready')
##!##             return(None) 
##!##         # For some reason, we can't get at the image array or its FITS
##!##         # header through CCDCamera.ImageArray, but we can through
##!##         # Document.ImageArray
##!##         if not self.getDocument():
##!##             log.info('There is no open image')
##!##             return(None)
##!##         
##!##         # Make sure we have an array to work with
##!##         c_im = self.Document.ImageArray
##!##         if c_im is None:
##!##             log.info('There is no image array')
##!##             return(None)
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
##!##         return(self.HDUList)        
##!## 
##!##     ## --> This is in the wrong place in terms of abstraction 
##!##     #def center_object(self):
##!##     #    if self.get_im() is None:
##!##     #        log.warning('No image')
##!##     #        return(None)        
##!##     #    #self.guider_move(self.calc_main_move(self.get_object_center_pix()))
##!##     #    #obj_c = get_jupiter_center(self.HDUList)
##!##     #    obj_c = self.ObsData.get_obj_center(self.HDUList)
##!##     #    log.info('object center = ', obj_c)
##!##     #    self.guider_move(self.calc_main_move(obj_c))


class ObsData():
    """Base class for observations, enabling object centering, etc."""

    def __init__(self, HDUList_im_or_fname=None):
        self.binning = None
        self.subframe_origin = None
        self.open_fits = None
        if HDUList_im_or_fname is None:
            log.info('ObsData initialized with no image')
            self.HDUList = None
        else:
            self.read_im(HDUList_im_or_fname)

    def read_im(self, HDUList_im_or_fname=None):
        """Returns an astropy.fits.HDUList given a filename, image or HDUList"""
        if HDUList_im_or_fname is None:
            log.info('No error, just saying that you have no image.')
            HDUList = None
        # --> These should potentially be issubclass
        elif isinstance(HDUList_im_or_fname, fits.HDUList):
            HDUList = HDUList_im_or_fname
        elif isinstance(HDUList_im_or_fname, str):
            fname = HDUList_im_or_fname
            HDUList = fits.open(fname)
            self.open_fits = True
            # --> when are we going to close the file?
            # --> need to have some property that keeps track of this
            #HDUList.close()
        elif isinstance(HDUList_im_or_fname, np.ndarray):
            hdu = fits.PrimaryHDU(HDUList_im_or_fname)
            HDUList = fits.HDUList(hdu)
        else:
            raise ValueError('Not a valid input, HDUList_im_or_fname')
        if HDUList is not None:
            try:
                h = HDUList[0].header
                # Note Astropy Pythonic transpose Y, X order
                self.binning = (h['YBINNING'], h['XBINNING'])
                self.binning = np.asarray(self.binning)
                self.subframe_origin = (h['YORGSUBF'], h['XORGSUBF'])
                self.subframe_origin = np.asarray(self.subframe_origin)
            except:
                log.warning('Could not read binning or subframe origin from image header.  Did you pass a valid MaxIm-recorded image and header')
                self.binning = None
                self.subframe_origin = None
        self.HDUList = HDUList
    
    def close_fits(self):
        if self.open_fits:
            HDUList.close()        

    def hist_of_im(self, im, readnoise=5):
        """Returns histogram of image and index into centers of bins.  
Uses readnoise (default = 5 e- RMS) to define bin widths"""
        
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
    
        return(hist, centers)

    def back_level(self, im, **kwargs):
        # Use the histogram technique to spot the bias level of the image.
        # The coronagraph creates a margin of un-illuminated pixels on the
        # CCD.  These are great for estimating the bias and scattered
        # light for spontanous subtraction.  The ND filter provides a
        # similar peak after bias subutraction (or, rather, it is the
        # second such peak)
        # --> This is very specific to the coronagraph.  Consider porting first peak find from IDL
        # Pass on readnoise, if supplied
        im_hist, im_hist_centers = hist_of_im(im, kwargs)
        im_peak_idx = signal.find_peaks_cwt(im_hist, np.arange(10, 50))
        return(im_hist_centers[im_peak_idx[0]])
        #im -= im_hist_centers[im_peak_idx[0]]



    # --> This is going to need some improvement
    def get_obj_center(self):
        """This is a really crummy object finder, since it will be confused
        by cosmic ray hits.  It is up to the user to define an object
        center finder that suits them, for instance one that uses
        PinPoint astrometry
        NOTE: The return order of indices is astropy FITS Pythonic: Y,
        X in unbinned coordinates from the ccd origin."""
    
        if self.HDUList is None:
            log.warning('No image')
            return(None)
        if self.binning or self.subframe_origin is None:
            log.warning('No binning and/or subframe origin info')
            return(None)
            
        im = self.HDUList[0].data
        im_center = np.unravel_index(np.argmax(im), im.shape)
        unbinned_center = self.binning * im_center + self.subframe_origin
        return(unbinned_center)

    def get_desired_center(self):
        """Returns geometric center of image.
        NOTE: The return order of indices is astropy FITS Pythonic: Y, X"""
        if self.HDUList is None:
            log.warning('No image')
            return(None)
        im = self.HDUList[0].data
        return(im.shape/2)
    

#class NDData:
#    """Neutral Density Data storage object"""
#
#    def __init__(self, HDUList_im_or_fname=None):
#
#        self.fname = None
#        self.im = None
#        # The shape of im is really all we need to store for
#        # calculations, once we have the ND_params
#        self.im_shape = None
#        self.ND_params = None
#
#        self.ingest_im(HDUList_im_or_fname)
#
#        # ND filter position in case none is derivable from flats.  This is from:
#        # print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit'))
#        self.default_ND_params = ((-7.35537190e-02,  -6.71900826e-02), 
#                               (1.24290909e+03,   1.34830909e+03))
#
#        # And we can refine it further for a good Jupiter example
#        #print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit',
#        #                  initial_try=((-7.35537190e-02,  -6.71900826e-02), 
#        #                               (1.24290909e+03,   1.34830909e+03))))
#        self.default_ND_params = ((-6.57640346e-02,  -5.77888855e-02),
#                               (1.23532221e+03,   1.34183584e+03))
#        self.n_y_steps = 15
#        self.x_filt_width = 25
#        self.edge_mask = 5
#        self.max_movement=50
#        self.max_delta_pix=10
#
#    def ingest_im(self, HDUList_im_or_fname=None):
#        """Returns image, reading from fname, if necessary"""
#        self.HDUList = read_im(HDUList_im_or_fname)
#        if self.HDUList is None:
#            return(None)
#        self.im = np.asfarray(self.HDUList[0].data)
#        self.im_shape = self.im.shape
#        h = self.HDUList[0].header
#        if not h.get('NDPAR00') is None:
#            ND_params = np.zeros((2,2))
#            # Note transpose, since we are working in C!
#            ND_params[0,0] = h['NDPAR00']
#            ND_params[1,0] = h['NDPAR01']
#            ND_params[0,1] = h['NDPAR10']
#            ND_params[1,1] = h['NDPAR11']
#            self.ND_params = ND_params
#
#        return(self.im)
#


class CorObsData(ObsData):
    """Calculates Jupiter current center and desired center given an image"""

    reference_flat = None

    def __init__(self, HDUList_im_or_fname=None, reference_flat=None, y_center=None):
        # Inherit init from base class, which reads the basic FITS
        # image (subject to our roverriding the read_im) and
        # initializes binning and subframe origin
        super().__init__(HDUList_im_or_fname)
        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.  By default this starts as None and gets filled as
        # the midpoint of the image when we read it
        self.y_center = y_center
        
        # The ND_params are the primary property we work hard to
        # generate.  These will be the slopes and intercepts of the
        # two lines defining the edges of the ND filter.  The origin
        # of the lines is the Y center of the unbinned, full-frame
        # chip
        self.ND_params = None
        # Angle is the (average) angle of the lines, useful for cases
        # where the filter is rotated significantly off of 90 degrees
        # (which is where I will run it frequently)
        self.angle = None

        # If we were passed an image, see if it has already been
        # through the system
        if self.HDUList is not None:
            h = self.HDUList[0].header
            if h.get('NDPAR00') is not None:
                ND_params = np.zeros((2,2))
                # Note transpose, since we are working in C!
                ND_params[0,0] = h['NDPAR00']
                ND_params[1,0] = h['NDPAR01']
                ND_params[0,1] = h['NDPAR10']
                ND_params[1,1] = h['NDPAR11']
                self.ND_params = ND_params

        # Make it easy to change the reference flat
        self.reference_flat = reference_flat

        # Define defaults for ND mask finding algorithm.  It is easy
        # to find the ND mask in flats, but not Jupiter images.  We
        # can use the answer from the flats to get the answer for
        # Jupiter.  This was from:
        # '/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit'
        self.default_ND_params = ((-7.35537190e-02,  -6.71900826e-02), 
                                  (1.24290909e+03,   1.34830909e+03))

        # And we can refine it further for a good Jupiter example
        #print(nd_filt_pos('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit',
        #                  initial_try=((-7.35537190e-02,  -6.71900826e-02), 
        #                               (1.24290909e+03,   1.34830909e+03))))
        self.default_ND_params = ((-6.57640346e-02,  -5.77888855e-02),
                                  (1.23532221e+03,   1.34183584e+03))
        
        # flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
        self.default_ND_params = ((3.78040775e-01,  3.84787113e-01),
                                  (1.24664929e+03,   1.35807856e+03))

        # flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
        self.default_ND_params = ((3.75447820e-01,  3.87551301e-01),
                                  (1.18163633e+03,   1.42002571e+03))

        #self.default_ND_params = None
        self.n_y_steps = 15
        self.x_filt_width = 25
        self.edge_mask = 5
        self.max_movement=50
        self.max_delta_pix=25

    def read_im(self, HDUList_im_or_fname=None):
        # Call our super-class read_im method + add to it
        super().read_im(HDUList_im_or_fname)

        if self.HDUList is None:
            return(None)
        # If we made it here, initialize our y_center to the midpoint
        # of the image, unless another was given when the object was
        # created
        self.y_center = self.HDUList[0].data.shape[0]/2

    def get_obj_center(self):
        """Returns center pixel coords of Jupiter whether or not Jupiter is on ND filter"""
        if self.HDUList is None:
            return(None)
        
        im = self.HDUList[0].data
        im -= self.back_level(im)
        
        # Check to see if Jupiter is sticking out significantly from
        # behind the ND filter, in which case we are better off just using
        # the center of mass of the image and calling that good enough
        #print(np.sum(im))
        # --> This may need some improvement
        if np.sum(im) > 1E9: 
            y_x = ndimage.measurements.center_of_mass(im)
            return(y_x)
        
        # Get the coordinates of the ND filter
        NDc = self.ND_coords()

        # Do a sanity check.  Note C order of indices
        badidx = np.where(np.asarray(NDc[0]) > im.shape[0])
        if np.any(badidx[0]):
            log.warning('Y dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')
            return(None)
        badidx = np.where(np.asarray(NDc[1]) > im.shape[1])
        if np.any(badidx[0]):
            log.warning('X dimension of image is smaller than position of ND filter!  Subimaging/binning mismatch?')
            return(None)
        
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
        #return(y_x[::-1], ND_center)
        # Stay in Pythonic y, x coords
        return(y_x)

    def get_desired_center(self, y=None):
        """Returns center of ND filter at position y.  In absence of user
input, default y = ny/2.  Default y can also be set at instantiation of
object"""
        if self.HDUList is None:
            return(None)
        desired_center = np.average(self.ND_edges(self.y_center)), self.y_center
        im = self.HDUList[0].data
        if (desired_center[0] < 0 or desired_center[0] > im.shape[0] or
            desired_center[1] < 0 or desired_center[1] > im.shape[1]):
            log.warning('Desired center is outside of image area!  Subimaging/binning mismatch?')
            return(None)
        else:
            return(desired_center)
        
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter"""
        if self.ND_params is None:
            self.get_ND_params()

        if self.HDUList is None:
            return(None)
        im = self.HDUList[0].data
        
        xs = [] ; ys = []
        for iy in np.arange(0, im.shape[0]):
            bounds = self.ND_params[1,:] + self.ND_params[0,:]*(iy - im.shape[0]/2) + np.asarray((self.edge_mask, -self.edge_mask))
            bounds = bounds.astype(int)
            for ix in np.arange(bounds[0], bounds[1]):
                xs.append(ix)
                ys.append(iy)
                
        # NOTE C order and the fact that this is a tuple of tuples
        return((ys, xs))

    def ND_edges(self, y, external_ND_params=None):
        """Returns x coords of ND filter edges at given y coordinate(s)"""
        if external_ND_params is not None:
            ND_params = external_ND_params
        else:
            if self.ND_params is None:
                self.get_ND_params()
                ND_params = self.ND_params

        ND_params = np.asarray(ND_params)
        im = self.HDUList[0].data
        if np.asarray(y).size == 1:
            return(ND_params[1,:] + ND_params[0,:]*(y - im.shape[0]/2))
        es = []
        for this_y in y:
            es.append(ND_params[1,:] + ND_params[0,:]*(this_y - im.shape[0]/2))
        return(es)
    
    def ND_angle(self):
        """Note this assumes square pixels"""
        if self.angle is not None:
            return(self.angle)
        if self.ND_params is None:
            self.get_ND_params()
        if self.ND_params is None:
            log.warning('Failed to get ND_params')
            return(None)

        # If we made it here, we need to calculate the angle
        # get_ND_params should have caught pathological cases, so we can
        # just use the average of the slopes
        self.angle = np.arctan(np.average(self.ND_params[0,:]))
        return(self.angle)

    # Might eventually want to make this part of a property system
    # https://docs.python.org/3/library/functions.html#property
    # PEP8 suggests I would name ND_params _ND_params and go from there
    def get_ND_params(self):
        """Returns parameters which characterize ND filter (currently 2 lines fit to edges)"""

        if self.ND_params is not None:
            return(self.ND_params)

        if self.HDUList is None:
            return(None)

        # If we made it here, we need to calculate ND_params.  
        im = self.HDUList[0].data
        ##-> clipc = np.where(im > 5000)
        ##-> im[clipc] = np.median(im)

        h = self.HDUList[0].header
        # See if we are a flat, in which case we should have high
        # contrast and can have small step size.
        # --> Plus we don't necessarily have a correct default_ND_params
        #if h['IMAGETYP'] == 'FLAT':
        #    # Note pythonic y dim is [0]
        #    self.n_y_steps = int(im.shape[0]/10)

        # The general method is to take the absolute value of the
        # gradient along each row to spot the edges of the ND filter.
        # Because contrast can be low in the Jupiter images, we need
        # to combine n_y_steps rows.  However, since the ND filter is
        # tilted by ~20 degrees or so, combining rows washes out the
        # edge of the ND filter unless we need to shift the rows in
        # the X direction based on the expected amount from the
        # default_ND_params.  Flats are high contrast, so we can
        # jump-start the process

        ND_edges = [] ; ypts = []

        # Create yrange at y_bin intervals, chopping of the last one
        # if it goes too far
        y_bin = int(im.shape[0]/self.n_y_steps)
        yrange = np.arange(0, im.shape[0], y_bin)
        if yrange[-1] + y_bin > im.shape[0]:
            yrange = yrange[0:-1]
        # picturing the image in C fashion, indexed from the top down,
        # ytop is the top point from which we bin y_bin rows together
        # --> remove me
        newim = im.copy()
        for ytop in yrange:
            # We will be referencing the measured points to the center of the bin
            ycent = ytop+y_bin/2
            # rowpt is each row in the ytop y_bin, which we need to
            # shift to accumulate into a subim that is effectively in
            # the rotated frame of default_ND_params
            subim = np.zeros((y_bin, im.shape[1]))
            for rowpt in np.arange(y_bin):
                # determine how many columns we will shift each row by
                # using the default_ND_params
                if self.default_ND_params is None:
                    this_ND_center = im.shape[1]/2.
                else:
                    this_ND_center = np.round(np.mean(self.ND_edges(rowpt+ytop, self.default_ND_params)))
                cshift = int(this_ND_center - im.shape[1]/2.)
                subim[rowpt, :] = np.roll(im[ytop+rowpt, :], -cshift)

            # --> remove me
            newim[ytop:ytop+y_bin, :] = subim
            #subim = im[ytop:ytop+y_bin, :]

            profile = np.sum(subim, 0)
            if h['IMAGETYP'] == 'FLAT':
                smoothed_profile = signal.savgol_filter(profile, self.x_filt_width, 3)#, deriv=1)
                d = np.gradient(smoothed_profile, 10)
                d2 = np.gradient(d, 10)
                s = np.abs(d2) * profile
            else:
                smoothed_profile = signal.savgol_filter(profile, self.x_filt_width, 0)#, deriv=1)
                d = np.gradient(smoothed_profile, 10)
                s = np.abs(d)
                
            plt.plot(profile)
            plt.show()
            #plt.plot(s)
            #plt.show()

            # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
            # points out same problem I had with with cwt.  It is too
            # sensitive to little peaks.  However, I can find the peaks
            # and just take the two largest ones
            #peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
            peak_idx = signal.find_peaks_cwt(s, np.arange(2, 80), min_snr=2)
            # Need to change peak_idx into an array instead of a list for
            # indexing
            peak_idx = np.array(peak_idx)
            # For flats, it is easy to spot the edge of the ND filter
            if h['IMAGETYP'] == 'FLAT':
                bounds = (0,s.size)
            else:
                # now that we are shifting to line up on the
                # centerline, bounds are easier to calculate
                #bounds = self.ND_edges(ycent, self.default_ND_params) + np.asarray((-self.max_movement, self.max_movement))
                tmp = np.asarray(self.default_ND_params)
                default_ND_width = tmp[1,1] - tmp[1,0]
                bounds = im.shape[1]/2 + np.asarray((-default_ND_width, default_ND_width))/2 + \
                         np.asarray((-self.max_movement, self.max_movement))/2
                bounds = bounds.astype(int)
                print(bounds)
                goodc = np.where(np.logical_and(bounds[0] < peak_idx, peak_idx < bounds[1]))
                peak_idx = peak_idx[goodc]
                print(peak_idx)
                print(s[peak_idx])
                plt.plot(s)
                plt.show()
                # Give up if we don't find two clear edges
                if peak_idx.size < 2:
                    print('No clear two peaks inside bounds')
                    continue
                
            # Assuming we have a set of good peaks, sort on peak size
            sorted_idx = np.argsort(s[peak_idx])
            # Unwrap
            peak_idx = peak_idx[sorted_idx]
            
            # Thow out if lower peak is too weak.  Use Carey Woodward's
            # trick of estimating the noise on the continuum To avoid
            # contamination, do this calc just over our desired interval
            ss = s[bounds[0]:bounds[1]]
            
            noise = np.std(ss[1:-1] - ss[0:-2])
            print(noise)
            if s[peak_idx[-2]] < noise:
                print("Rejected -- not above noise threshold")
                continue
            
            # Find top two and put back in index order
            top_two = np.sort(peak_idx[-2:])
            # Accumulate in tuples
            ND_edges.append(top_two)
            ypts.append(ycent)
            

        # --> remove me
        plt.imshow(newim)
        plt.show()

        ND_edges = np.asarray(ND_edges)
        ypts = np.asarray(ypts)
        if ND_edges.size < 2:
            if self.default_ND_params is None:
                raise ValueError('Not able to find ND filter position')
            log.warning('Unable to improve filter position over initial guess')
            self.ND_params = np.asarray(self.default_ND_params)
            return(self.default_ND_params)
        
        # Put the ND_edges back into the original orientation before
        # we cshifted them with default_ND_params
        if self.default_ND_params is not None:
            self.default_ND_params = np.asarray(self.default_ND_params)
            es = []
            for this_y in ypts:
                es.append(self.default_ND_params[1,:] - im.shape[1]/2. + self.default_ND_params[0,:]*(this_y - im.shape[0]/2))
            ND_edges =  ND_edges + es


        # DEBUGGING
        plt.plot(ypts, ND_edges)
        plt.show()
        
        # Fit lines to our points, making the origin the center of the image in Y
        ND_params = np.polyfit(ypts-im.shape[0]/2, ND_edges, 1)
        ND_params = np.asarray(ND_params)
        #print(ND_params)
        
        # Check to see if there are any bad points, removing them and
        # refitting
        resid = ND_edges - self.ND_edges(ypts, ND_params)
        #print(resid)
        # Do this one side at a time, since the points might not be on
        # the same y level and it is not easy to zipper the coordinate
        # tuple apart in Python.  For some reason, np.where is
        # returning a tuple of arrays
        goodc0 = np.where(abs(resid[:,0]) < self.max_delta_pix)
        goodc1 = np.where(abs(resid[:,1]) < self.max_delta_pix)
        #print(ypts[goodc1]-im.shape[0]/2)
        #print(ND_edges[goodc1, 1])
        #print(ND_edges)
        #print(goodc0)
        #print(goodc1)
        
        if (len(goodc0[0]) < 2) or (len(goodc1[0]) < 2):
            txt = 'Not enough good fit points.' 
            if self.default_ND_params is None:
                raise ValueError(txt + '  No initial try available, raising error.')
            log.warning(txt + '  Returning initial try.')
            ND_params = self.default_ND_params
            return(self.ND_params)
            
        # If we made it here, we should have two lines to fit
        if len(goodc0[0]) < resid.shape[0]:
            ND_params[:,0] = np.polyfit(ypts[goodc0]-im.shape[0]/2,
                                        ND_edges[goodc0, 0][0], 1)
        if len(goodc1[0]) < resid.shape[1]:
            ND_params[:,1] = np.polyfit(ypts[goodc1]-im.shape[0]/2,
                                        ND_edges[goodc1, 1][0], 1)
            #print(ND_params)
            # Check parallelism by calculating shift of ends relative to each other
        dp = abs((ND_params[0,1] - ND_params[0,0]) * im.shape[0]/2)
        if dp > self.max_delta_pix:
            txt = 'ND filter edges are not parallel.  Edges are off by ' + str(dp) + ' pixels.'
            # DEBUGGING
            print(txt)
            plt.plot(ypts, ND_edges)
            plt.show()
            
            if self.default_ND_params is None:
                raise ValueError(txt + '  No initial try available, raising error.')
            log.warning(txt + ' Returning initial try.')
            ND_params = self.default_ND_params

        self.ND_params = ND_params
        # The HDUList headers are objects, so we can do this
        # assignment and the original object property gets modified
        h = self.HDUList[0].header
        # Note transpose, since we are working in C!
        h['NDPAR00'] = (ND_params[0,0], 'ND filt left side slope at Y center of im')
        h['NDPAR01'] = (ND_params[1,0], 'ND filt left side offset at Y center of im')
        h['NDPAR10'] = (ND_params[0,1], 'ND filt right side slope at Y center of im')
        h['NDPAR11'] = (ND_params[1,1], 'ND filt right side offset at Y center of im')

        #print(self.ND_params)
        return(self.ND_params)

##SAVED##    def get_ND_params(self):
##SAVED##        """Returns parameters which characterize ND filter (currently 2 lines fir to edges)"""
##SAVED##        if not self.ND_params is None:
##SAVED##            return(self.ND_params)
##SAVED##
##SAVED##        if self.HDUList is None:
##SAVED##            return(None)
##SAVED##
##SAVED##        # If we made it here, we need to calculate ND_params.  Take
##SAVED##        # n_y_steps and make profiles, take the gradient and absolute
##SAVED##        # value to spot the edges of the ND filter
##SAVED##        im = self.HDUList[0].data
##SAVED##        h = self.HDUList[0].header
##SAVED##        ND_edges = [] ; ypts = []
##SAVED##        y_bin = int(im.shape[0]/self.n_y_steps)
##SAVED##        yrange = np.arange(0, im.shape[0], y_bin)
##SAVED##        for ytop in yrange:
##SAVED##            subim = im[ytop:ytop+y_bin, :]
##SAVED##            profile = np.sum(subim, 0)
##SAVED##            smoothed_profile = signal.savgol_filter(profile, self.x_filt_width, 3)
##SAVED##            d = np.gradient(smoothed_profile, 10)
##SAVED##            s = np.abs(d)
##SAVED##            #plt.plot(profile)
##SAVED##            #plt.show()
##SAVED##            #plt.plot(s)
##SAVED##            #plt.show()
##SAVED##
##SAVED##            # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
##SAVED##            # points out same problem I had with with cwt.  It is too
##SAVED##            # sensitive to little peaks.  However, I can find the peaks
##SAVED##            # and just take the two largest ones
##SAVED##            peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
##SAVED##            # Need to change peak_idx into an array instead of a list for
##SAVED##            # indexing
##SAVED##            peak_idx = np.array(peak_idx)
##SAVED##            # For flats, it is easy to spot the edge of the ND filter
##SAVED##            if h['IMAGETYP'] == 'FLAT':
##SAVED##                bounds = (0,s.size)
##SAVED##            else:
##SAVED##                # --> I am eventually going to want to hard-code in ND
##SAVED##                # ND_params for the various date ranges
##SAVED##                bounds = self.ND_edges(ytop, self.default_ND_params) + np.asarray((-self.max_movement, self.max_movement))
##SAVED##                bounds = bounds.astype(int)
##SAVED##                goodc = np.where(np.logical_and(bounds[0] < peak_idx, peak_idx < bounds[1]))
##SAVED##                peak_idx = peak_idx[goodc]
##SAVED##                #print(peak_idx)
##SAVED##                #print(s[peak_idx])
##SAVED##                #plt.plot(s)
##SAVED##                #plt.show()
##SAVED##                # Give up if we don't find two clear edges
##SAVED##                if peak_idx.size < 2:
##SAVED##                    continue
##SAVED##    
##SAVED##            # Assuming we have a set of good peaks, sort on peak size
##SAVED##            sorted_idx = np.argsort(s[peak_idx])
##SAVED##            # Unwrap
##SAVED##            peak_idx = peak_idx[sorted_idx]
##SAVED##    
##SAVED##            # Thow out if lower peak is too weak.  Use Carey Woodward's
##SAVED##            # trick of estimating the noise on the continuum To avoid
##SAVED##            # contamination, do this calc just over our desired interval
##SAVED##            ss = s[bounds[0]:bounds[1]]
##SAVED##    
##SAVED##            noise = np.std(ss[1:-1] - ss[0:-2])
##SAVED##            #print(noise)
##SAVED##            if s[peak_idx[-2]] < 3 * noise:
##SAVED##                #print("Rejected")
##SAVED##                continue
##SAVED##    
##SAVED##            # Find top two and put back in index order
##SAVED##            top_two = np.sort(peak_idx[-2:])
##SAVED##            # Accumulate in tuples
##SAVED##            ND_edges.append(top_two)
##SAVED##            ypts.append(ytop)
##SAVED##    
##SAVED##        ND_edges = np.asarray(ND_edges)
##SAVED##        ypts = np.asarray(ypts)
##SAVED##        if ND_edges.size < 2:
##SAVED##            if self.default_ND_params is None:
##SAVED##                raise ValueError('Not able to find ND filter position')
##SAVED##            log.warning('Unable to improve filter position over initial guess')
##SAVED##            self.ND_params = np.asarray(self.default_ND_params)
##SAVED##            return(self.default_ND_params)
##SAVED##        
##SAVED##        # DEBUGGING
##SAVED##        #plt.plot(ypts, ND_edges)
##SAVED##        #plt.show()
##SAVED##    
##SAVED##        # Fit lines to our points, making the origin the center of the image in Y
##SAVED##        ND_params = np.polyfit(ypts-im.shape[0]/2, ND_edges, 1)
##SAVED##        ND_params = np.asarray(ND_params)
##SAVED##        #print(ND_params)
##SAVED##        
##SAVED##        # Check to see if there are any bad points, removing them and
##SAVED##        # refitting
##SAVED##        resid = ND_edges - self.ND_edges(ypts, ND_params)
##SAVED##        #print(resid)
##SAVED##        # Do this one side at a time, since the points might not be on
##SAVED##        # the same y level and it is not easy to zipper the coordinate
##SAVED##        # tuple apart in Python
##SAVED##        goodc0 = np.where(abs(resid[:,0]) < self.max_delta_pix)
##SAVED##        goodc1 = np.where(abs(resid[:,1]) < self.max_delta_pix)
##SAVED##        #print(ypts[goodc1]-im.shape[0]/2)
##SAVED##        #print(ND_edges[goodc1, 1])
##SAVED##        #print(ND_edges)
##SAVED##        if len(goodc0) < resid.shape[1]:
##SAVED##            ND_params[:,0] = np.polyfit(ypts[goodc0]-im.shape[0]/2,
##SAVED##                                     ND_edges[goodc0, 0][0], 1)
##SAVED##        if len(goodc1) < resid.shape[1]:
##SAVED##            ND_params[:,1] = np.polyfit(ypts[goodc1]-im.shape[0]/2,
##SAVED##                                     ND_edges[goodc1, 1][0], 1)
##SAVED##        #print(ND_params)
##SAVED##        # Check parallelism by calculating shift of ends relative to each other
##SAVED##        dp = abs((ND_params[0,1] - ND_params[0,0]) * im.shape[0]/2)
##SAVED##        if dp > self.max_delta_pix:
##SAVED##            txt = 'ND filter edges are not parallel.  Edges are off by ' + str(dp) + ' pixels.'
##SAVED##            # DEBUGGING
##SAVED##            print(txt)
##SAVED##            plt.plot(ypts, ND_edges)
##SAVED##            plt.show()
##SAVED##    
##SAVED##            if self.default_ND_params is None:
##SAVED##                raise ValueError(txt)
##SAVED##            log.warning(txt + ' Returning initial try.')
##SAVED##            ND_params = self.default_ND_params
##SAVED##
##SAVED##        self.ND_params = ND_params
##SAVED##        # The HDUList headers are objects, so we can do this
##SAVED##        # assignment and the original object property gets modified
##SAVED##        h = self.HDUList[0].header
##SAVED##        # Note transpose, since we are working in C!
##SAVED##        h['NDPAR00'] = (ND_params[0,0], 'ND filt left side slope at Y center of im')
##SAVED##        h['NDPAR01'] = (ND_params[1,0], 'ND filt left side offset at Y center of im')
##SAVED##        h['NDPAR10'] = (ND_params[0,1], 'ND filt right side slope at Y center of im')
##SAVED##        h['NDPAR11'] = (ND_params[1,1], 'ND filt right side offset at Y center of im')
##SAVED##
##SAVED##        #print(self.ND_params)
##SAVED##        return(self.ND_params)

    def imshow(self):
        if self.HDUList is None:
            return(None)
        im = self.HDUList[0].data
        plt.imshow(im)
        plt.show()
        return(True)

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

    return(new_guide_dt, r[0], r[1])

log.setLevel('INFO')
Na =   CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0007_Na_off-band.fit')
print(Na.get_ND_params())
print(Na.ND_angle())

# SII =  CorObsData('/data/io/IoIO/raw/2017-05-28/Na_IPT-0035_SII_on-band.fit')
# print(SII.get_ND_params())
# print(SII.ND_angle())


flat = CorObsData('/data/io/IoIO/raw/2017-05-28/Sky_Flat-0002_Na_off-band.fit')
#flat.imshow()
#flat.n_y_steps = 100
#print(flat.get_ND_params())
#print(flat.ND_angle())


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
#     obj_cent = J.get_obj_center()
#     print('object center = ', obj_cent)
#     print('Getting desired center...')
#     desired_cent = J.get_desired_center()
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
