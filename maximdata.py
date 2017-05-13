#!/usr/bin/python3

# Class for controlling MaxIm DL via ActiveX/COM events

# Pattern object after ccdproc.CCDData and astropy.io.fits.HDUlist
# such that it is a container of higher-level classes (e.g. HDUlist
# contains numpy.ndarrays), not a subclass of those classes
# (e.g. HDUlist elements don't inherit np.ndarray properties/methods,
# you have to extract them from the HDUlist and then do the operations)

# DEBUGGING
import matplotlib.pyplot as plt

from astropy import log
from astropy.io import fits
import win32com.client
import numpy as np
import time

class MaxImData():
    """Stores data related to controlling MaxIm DL via ActiveX/COM events."""

    def __init__(self):
        # Create containers for all of the objects that can be
        # returned by MaxIm.  We'll only populate them when we need
        # them.  Some of these we may never use or write code for
        self.Application = None
        self.CCDCamera = None
        self.Document = None
        self.main_HDUlist = None
        self.required_FITS_keys = ('DATE-OBS', 'EXPTIME', 'EXPOSURE', 'XBINNING', 'YBINNING', 'XORGSUBF', 'YORGSUBF', 'FILTER', 'IMAGETYP', 'OBJECT')

        # Maxim doesn't expose the results of this menu item from the
        # Guider Settings Advanced tab in the object.  It's for
        # 'scopes that let you push both RA and DEC buttons at once
        # for guider movement
        self.simultaneous_guide_corrections = False
        # We can use the CCDCamera.GuiderMaxMove[XY] property for an
        # indication of how long it is safe to press the guider
        # movement buttons
        self.guider_max_move_multiplier = 20

        # The conversion between guider button push time and guider
        # pixels is stored in the CCDCamera.Guider[XY]Speed
        # properties.  Plate scales in arcsec/pix are not, though they
        # can be greped out of FITS headers. 

        # Main camera plate solve, binned 2x2:
        # RA 12h 55m 33.6s,  Dec +03° 27' 42.6"
        # Pos Angle +04° 34.7', FL 1178.9 mm, 1.59"/Pixel
        self.main_plate = 1.59/2 # arcsec/pix
        self.main_angle = 4.578333333333333 # CCW from N on east side of pier

        # Guider (Binned 1x1)
        # RA 07h 39m 08.9s,  Dec +34° 34' 59.0"
        # Pos Angle +178° 09.5', FL 401.2 mm, 4.42"/Pixel
        self.guider_plate = 4.42
        self.guider_angle = 178+9.5/60 - 180

        # This is a function that returns two vectors, the current
        # center of the object in the main camera and the desired center 
        #self.get_object_center = None

    def getApplication(self):
        if not self.Application is None:
            return(True)
        try:
            self.Application = win32com.client.Dispatch("MaxIm.Application")
        except:
            raise EnvironmentError('Error creating MaxIM application object.  Is MaxIM installed?')
        # Catch any other weird errors
        return(isinstance(self.Application, win32com.client.CDispatch))
        
    def getCCDCamera(self):
        if not self.CCDCamera is None:
            return(True)
        try:
            self.CCDCamera = win32com.client.Dispatch("MaxIm.CCDCamera")
        except:
            raise EnvironmentError('Error creating CCDCamera object')
        # Catch any other weird errors
        return(isinstance(self.CCDCamera, win32com.client.CDispatch))

    def getDocument(self):
        """Gets the document object of the current window"""
        # The CurrentDocument object gets refreshed when new images
        # are taken, so all we need is to make sure we are connected
        # to begin with
        if not self.Document is None:
            return(True)
        self.getApplication()
        try:
            self.Document = self.Application.CurrentDocument
        except:
            raise EnvironmentError('Error retrieving document object')
        # Catch any other weird errors
        return(isinstance(self.Document, win32com.client.CDispatch))

    def connect(self):
        """Link to telescope, CCD camera(s), filter wheels, etc."""
        self.getApplication()
        self.Application.TelescopeConnected = True
        if self.Application.TelescopeConnected == False:
            raise EnvironmentError('Link to telescope failed.  Is the power on to the mount?')
        self.getCCDCamera()
        self.CCDCamera.LinkEnabled = True
        if self.CCDCamera.LinkEnabled == False:
            raise EnvironmentError('Link to camera hardware failed.  Is the power on to the CCD (including any connection hardware such as USB hubs)?')

    def guider_move(self, ddec, dra, dec=None):
        """Moves the telescope using guider slews.  ddec, dra in
        arcsec.  NOTE ORDER OF COORDINATES: Y, X to conform to C
        ordering of FITS images """
        self.connect()
        if dec is None:
            dec = self.CCDCamera.GuiderDeclination
        # Change to rectangular tangential coordinates for small deltas
        dra = dra*np.cos(np.radians(dec))
        # The guider motion is calibrated in pixels per second, with
        # the guider angle applied separately.  We are just moving in
        # RA and DEC, so we don't need to worry about the guider angle
        dpix = np.asarray(([dra, ddec])) / self.guider_plate
        # Multiply by speed, which is in pix/sec
        dt = dpix / np.asarray((self.CCDCamera.GuiderXSpeed, self.CCDCamera.GuiderYSpeed))
        
        # Do a sanity check to make sure we are not moving too much
        max_t = (self.guider_max_move_multiplier *
                 np.asarray((self.CCDCamera.GuiderMaxMoveX, 
                             self.CCDCamera.GuiderMaxMoveY)))
            
        if np.any(np.abs(dt) > max_t):
            #print(str((dra, ddec)))
            #print(str(np.abs(dt)))
            log.warning('requested move of ' + str((dra, ddec)) + ' arcsec translates into move times of ' + str(np.abs(dt)) + ' seconds.  Limiting move in one or more axes to max t of ' + str(max_t))
            dt = np.minimum(max_t, abs(dt)) * np.sign(dt)
            
        log.info('Moving guider ' + str(dt))
        if dt[0] > 0:
            RA_success = self.CCDCamera.GuiderMove(0, dt[0])
        elif dt[0] < 0:
            RA_success = self.CCDCamera.GuiderMove(1, -dt[0])
        else:
            # No need to move
            RA_success = True
        # Wait until move completes if we can't push RA and DEC
        # buttons simultaneously
        while not self.simultaneous_guide_corrections and self.CCDCamera.GuiderMoving:
                time.sleep(0.1)
        if dt[1] > 0:
            DEC_success = self.CCDCamera.GuiderMove(2, dt[1])
        elif dt[1] < 0:
            DEC_success = self.CCDCamera.GuiderMove(3, -dt[1])
        else:
            # No need to move
            DEC_success = True
        while self.CCDCamera.GuiderMoving:
            time.sleep(0.1)
        return(RA_success and DEC_success)
    
    def rot(self, vec, theta):
        """Rotates vector counterclockwise by theta degrees IN A
        TRANSPOSED COORDINATE SYSTEM Y,X"""
        # This is just a standard rotation through theta on X, Y, but
        # when we transpose, theta gets inverted
        theta = -np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        print(vec)
        print(theta, c, s)
        M = np.matrix([[c, -s], [s, c]])
        return(np.dot(M, vec))

    def calc_main_move(self, current_pos, desired_center=None):
        """Returns vector [ddec, dra] in arcsec to move scope to
        center object at current_pos, where the current_pos and
        desired_centers are vectors expressed in pixels on the main
        camera in the astropy FITS Pythonic order Y, X"""

        self.connect()
        if desired_center is None:
            desired_center = \
            np.asarray((self.CCDCamera.StartY + self.CCDCamera.NumY, 
                        self.CCDCamera.StartX + self.CCDCamera.NumX)) / 2.
        dpix = np.asarray(desired_center) - current_pos
        dpix = self.rot(dpix, self.main_angle)
        return(dpix * self.main_plate)

    # --> finish this
    def get_keys(self):
        """Puts current MaxIm image (the image with focus) into a FITS HDUlist.  If an exposure is being taken or there is no image, the im array is set equal to None"""
        
    def get_im(self):
        """Puts current MaxIm image (the image with focus) into a FITS HDUlist.  If an exposure is being taken or there is no image, the im array is set equal to None"""
        self.connect()
        # Clear out main_HDUlist in case we fail
        self.main_HDUlist = None
        if not self.CCDCamera.ImageReady:
            return(None) 
        # For some reason, we can't get at the image array or its FITS
        # header through CCDCamera.ImageArray, but we can through
        # Document.ImageArray
        self.getDocument()
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
        c_im = np.asarray(self.Document.ImageArray)
        adata = c_im.flatten()#order='K')# already in C order in memory
        # The [::-1] reverses the indices
        nddata = np.ndarray(shape=c_im.shape[::-1],
                            buffer=adata, order='F')
        hdu = fits.PrimaryHDU(nddata)
        # --> use get_keys
        for k in self.required_FITS_keys:
            hdu.header[k] = self.Document.GetFITSKey(k)
        self.main_HDUlist = fits.HDUList(hdu)
        return(self.main_HDUlist)        

    # This is a really crummy object finder, since it will be confused
    # by cosmic ray hits.  It is up to the user to define an object
    # center finder that suits them, for instance one that uses
    # PinPoint astrometry
    # NOTE: The return order of indices is astropy FITS Pythonic: Y, X
    def get_object_center_pix(self):
        self.get_im()
        im = self.main_HDUlist[0].data
        return(np.unravel_index(np.argmax(im), im.shape))
        
        
    def center_object(self):
        self.guider_move(self.calc_main_move(self.get_object_center()))

if __name__ == "__main__":
    
    log.setLevel('INFO')

    # Start MaxIm
    M = MaxImData()
    M.getCCDCamera()
    # Keep it alive for these experiments
    M.CCDCamera.DisableAutoShutdown = True
    #M.connect()
    #print(M.calc_main_move((50,50)))
    #print(M.guider_move(10000,20))
    print(M.get_object_center_pix())
    print(M.calc_main_move((50,50)))
    plt.imshow(M.main_HDUlist[0].data)
    plt.show()
    # Kill MaxIm
    #M = None
 
