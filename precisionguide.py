#--> This code was developed using Anaconda3 installed as
#--> administrator and with PATH option selected during the install so
#--> that python can be used from a Windows command line.  NOTE: to
#--> get command line arguments passed, which is essential for this
#--> code, you need need to edit registry to make
# Computer\HKEY_CLASSES_ROOT\Applications\python.exe\shell\open\command
# "C:\ProgramData\Anaconda3\python.exe" "%1" %*
# Thanks to
# https://stackoverflow.com/questions/29540541/executable-python-script-not-take-sys-argv-in-windows
# Alternately, you can just call python with the full path to the
# module and then its arguments


import importlib
import sys
import os
import socket
import time
import subprocess
import argparse
import json

import numpy as np
from scipy import signal
from astropy import log
from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.time import Time, TimeDelta

if sys.platform == 'win32':
    try:
        import win32com.client
    except:
        log.info('You are missing the win32com.client.  This should be in the Anaconda package.  MaxIm/telescope control will not work.')
    else:
        # http://timgolden.me.uk/pywin32-docs/html/com/win32com/HTML/QuickStartClientCom.html
        # Use makepy.py -i to poke around in what might be useful
        try:
            # 'ASCOM Master Interfaces for .NET and COM' constants.
            # Use example: win32com.client.constants.shutterOpen
            win32com.client.gencache.EnsureModule('{76618F90-032F-4424-A680-802467A55742}', 0, 1, 0)
        except:
            log.info('ASCOM does not seem to be installed.  MaxIm/telescope control will not work.')
        else:
            try:
                # MaxIm constants.  The long string is the GUID of MaxIm found by makepy.py
                win32com.client.gencache.EnsureModule('{B4955EC7-F7F2-11D2-AA9C-444553540000}', 0, 1, 0)
            except:
                log.info('MaxIm not found.  MaxIm/telescope control will not work.')
else:
        log.info('You are not on a Windows system.  The MaxIm/telescope control features of this package will not work unless you are on a Windows system.')

import define as D

# --> these are things that eventually I would want to store in a
# --> configuration file
# --> CHANGE ME BACK TO 1s(or 7s) and filter 0 (0.7s or 0.3 on
# --> Vega filter 1 works for day) 
default_exptime = 1
default_filt = 0
default_cent_tol = 5   # Pixels
default_guider_exptime = 1 # chage back to 1 for night, 0.2 for day

# --> I may improve this location or the technique of message passing
if socket.gethostname() == "snipe":
    raw_data_root = '/data/io/IoIO/raw'
elif socket.gethostname() == "puppy" or socket.gethostname() == "gigabyte":
    # --> This doesn't work.  I need Unc?
    #raw_data_root = '//snipe/data/io/IoIO/raw'
    raw_data_root = r'\\snipe\data\io\IoIO\raw'
    default_telescope = 'ScopeSim.Telescope'
elif socket.gethostname() == "IoIO1U1":
    raw_data_root = r'C:\Users\PLANETARY SCIENCE\Desktop\IoIO\data'
    # --> Eventually, it would be nice to have this in a chooser
    default_telescope = 'AstroPhysicsV2.Telescope'
default_guide_box_command_file = os.path.join(raw_data_root, 'GuideBoxCommand.txt')
default_guide_box_log_file = os.path.join(raw_data_root, 'GuideBoxLog.txt')

run_level_main_astrometry = os.path.join(
    raw_data_root, '2019-02_Astrometry/PinPointSolutionEastofPier.fit')
    #raw_data_root, '2019-02_Astrometry/PinPointSolutionWestofPier.fit')
    #raw_data_root, '2018-04_Astrometry/PinPointSolutionEastofPier.fit')

# --> Currently only guider WestofPier (looking east) works properly,
# --> which might indicate that calculations need to be made with true
# --> north of CCD aligned with true north button on mount.  Although
# --> pier flip doesn't affect N/S because tube rolls over too, E/W is
# --> affected
run_level_guider_astrometry = os.path.join(
    #raw_data_root, '2019-02_Astrometry/GuiderPinPointSolutionWestofPier.fit')
    raw_data_root, '2019-02_Astrometry/GuiderPinPointSolutionEastofPier.fit')    
    #raw_data_root, '2018-04_Astrometry/GuiderPinPointSolutionWestofPier.fit')
    #raw_data_root, '2018-01_Astrometry//GuiderPinPointSolutionEastofPier.fit')

horizon_limit = 8.5

def angle_norm(angle, maxang):
    """Normalize an angle to run up to maxang degrees"""
    angle += 360
    angle %= 360
    if angle > maxang: # handles 180 case
        angle -= 360
    return angle

def get_HDUList(HDUList_im_or_fname):
    """Returns an astropy.fits.HDUList given a filename, image or
    HDUList.  If you have a set of HDUs, you'll need to put them
    together into an HDUList yourself, since this can't guess how
    to do that"""
    if isinstance(HDUList_im_or_fname, fits.HDUList):
        return HDUList_im_or_fname
    elif isinstance(HDUList_im_or_fname, str):
        return fits.open(HDUList_im_or_fname)
    elif isinstance(HDUList_im_or_fname, np.ndarray):
        hdu = fits.PrimaryHDU(HDUList_im_or_fname)
        return fits.HDUList(hdu)
    else:
        raise ValueError('Not a valid input, HDUList_im_or_fname, expecting, fits.HDUList, string, or np.ndarray')

def pier_flip_astrometry(header):
    header['CDELT1'] *= -1
    header['CDELT2'] *= -1
    header['CD1_1']  *= -1
    header['CD1_2']  *= -1
    header['CD2_1']  *= -1
    header['CD2_2']  *= -1
    if header.get('PIERSIDE'):
        if header['PIERSIDE'] == 'EAST':
            header['PIERSIDE'] = 'WEST'
        else:
            header['PIERSIDE'] = 'EAST'                    
    header['FLIPAPPL'] = (True, 'Artificially flipped pier side')
    header['HISTORY'] = 'Artificially flipped pier side, modified CD* and PIERSIDE'


# --> Really what I think I want is a PGData for all of the center and
# --> rate stuff.  That will clean up the ObsData property and __init__
class ObsData():
    """Base class for observations, enabling object centering, etc.

    This is intended to work in an active obsering setting, so
    generally an image array will be received, the desired properties
    will be calculated from it and those properties will be read by
    the calling code.

    """

    def __init__(self,
                 HDUList_im_or_fname=None,
                 desired_center=None,
                 recalculate=False,
                 readnoise=5):
        if HDUList_im_or_fname is None:
            raise ValueError('No HDUList_im_or_fname provided')
        self.recalculate = recalculate
        self.readnoise = readnoise
        # Set up our basic FITS image info
        self.header = None
        self._binning = None
        self._subframe_origin = None
        self._we_opened_file = None
        # Keep property for later use/speedy access
        self._hist_of_im = None
        self._back_level = None
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
        # --> Here is where I would make the division between ObsData
        # and PGData.  PGData would init the rates and stuff + read
        # the ObsData.  The ObsData would have a cleanup method that
        # otherwise would not be called
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
        """Populate ObsData with HDUList and associated info"""
        self.HDUList = get_HDUList(HDUList_im_or_fname)
        # Store the original shape of our image so we can do
        # coordinate calculations without it
        self.oshape = np.asarray(self.HDUList[0].data.shape)
        if isinstance(HDUList_im_or_fname, np.ndarray):
            # We don't have any metadata
            return self.HDUList
        # All other options should have HDUList already populated with
        # stuff we need.  Copy stuff into our local property as needed
        if isinstance(HDUList_im_or_fname, str):
            self._we_opened_file = True
        # Store the header in our object.  This is just a
        # reference at first, but after HDUList is deleted, this
        # becomes the only copy
        # https://stackoverflow.com/questions/22069727/python-garbage-collector-behavior-on-compound-objects
        self.header = self.HDUList[0].header
        # Calculate an astropy Time object for the midpoint of the
        # observation for ease of time delta calculations.
        # Account for darktime, if available
        try:
            exptime = self.header.get('DARKTIME') 
            if exptime is None:
                exptime = self.header['EXPTIME']
                # Use units to help with astropy.time calculations
                exptime *= u.s
                self.Tmidpoint = (Time(self.header['DATE-OBS'],
                                       format='fits')
                                  + exptime/2)
        except:
            log.warning('Cannot read DARKTIME and/or EXPTIME keywords from FITS header')
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
        if self.recalculate == True:
            # We don't want to use values stored in the file, this
            # forces recalculate
            return self.HDUList
        try:
            cx = self.header['OBJ_CR0']
            cy = self.header['OBJ_CR1']
            self._obj_center = np.asarray((cy, cx))
            dx = self.header['DES_CR0']
            dy = self.header['DES_CR1']
            self._desired_center = np.asarray((dy, dx))
        except:
            # It was worth a try
            pass
        return self.HDUList

    def unbinned(self, coords):
        """Returns coords referenced to full CCD given internally stored binning/subim info"""
        coords = np.asarray(coords)
        return np.asarray(self._binning * coords + self._subframe_origin)

    def binned(self, coords):
        """Assuming coords are referenced to full CCD, return location in binned coordinates relative to the subframe origin"""
        coords = np.asarray(coords)
        return np.asarray((coords - self._subframe_origin) / self._binning)
        
    def HDU_unbinned(self, a=None):
        """Returns an unbinned version of the primary HDU image or the primary HDU image if it is not binned.  If a is provided, pretend that is the primary HDU (e.g., it may be a modified version) and unbin that
        """
        if a is None:
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
        self.header['OBJ_CR0'] = (self._obj_center[1], 'Object center X')
        self.header['OBJ_CR1'] = (self._obj_center[0], 'Object center Y')
        self.header['QUALITY'] = (self.quality, 'Quality on 0-10 scale of center determination')
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
        self.header['DES_CR0'] = (self._desired_center[1], 'Desired center X')
        self.header['DES_CR1'] = (self._desired_center[0], 'Desired center Y')
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

#Daniel
if True:
    class MakeList():
        def __init__(self, mlist=None):
            if mlist is None:
                self._mlist = []
            else:
                if isinstance(mlist, list):
                    self._mlist = mlist
                else:
                    raise TypeError('Input must be a list.')
        def append(self, item):
            self._mlist.append(item)
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
        if sys.platform != 'win32':
            raise EnvironmentError('Can only control camera and telescope from Windows platform')
        self.main_astrometry = main_astrometry
        if self.main_astrometry is None:
            self.main_astrometry = run_level_main_astrometry
        if isinstance(self.main_astrometry, str):
            HDUList = fits.open(self.main_astrometry)
            self.main_astrometry = HDUList[0].header
            HDUList.close()
        self.guider_astrometry = guider_astrometry
        if self.guider_astrometry is None:
            self.guider_astrometry = run_level_guider_astrometry
        if isinstance(self.guider_astrometry, str):
            HDUList = fits.open(self.guider_astrometry)
            self.guider_astrometry = HDUList[0].header
            HDUList.close()

        self.default_filt = default_filt

        # Mount & guider information --> some of this might
        # eventually be tracked by this application in cases where
        # it is not available from the mount
        self.alignment_mode = None
        self.guider_cal_pier_side = None
        self.ACP_mode = None
        self.previous_GuiderReverseX = None
        self.guide_rates = None # degrees/s
        self.calculated_guide_rates = None # degrees/s
        #self.pinpoint_N_is_up = None
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
        #  --> Too little motion seems to freeze the system, at
        # least sometimes
        self.min_guide_move_time = 0.05
        self.horizon_limit_value = horizon_limit
        self.max_guide_num_steps = 8
        self.connect()
        self.populate_obj()

    #def __del__(self):
    #    # Trying to keep camera from getting disconnected on exit
    #    # --> What seems to work is having FocusMax recycled after
    #    # this firsts connected (though not positive of that)
    #    self.CCDCamera.LinkEnabled == True

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

    def populate_obj(self):
        """Called by init() after connect() to finish init()"""

        # Fill our object with things we know
        if self.telescope_connectable:
            self.alignment_mode = self.Telescope.AlignmentMode
        else:
            # --> Eventually this warning might go away
            log.error("Mount is not connected -- did you specify one in setup [currently the software source code]?  If you have a German equatorial mount (GEM), this software will likely not work properly upon pier flips [because code has not yet been written to let you specify the mode of your telescope on the fly].  Other mount types will work OK, but you should keep track of the Scope Dec. box in MaxIm's Guide tab.")
         # Save GuiderReverseX state so we can put it back when we stop
        # guiding
        self.previous_GuiderReverseX = self.CCDCamera.GuiderReverseX
        # --> DEBUGGING, or maybe long-term.  Clear this here for
        # --> convenience on restart for debugging
        log.info("GuiderReverseX was " + repr(self.CCDCamera.GuiderReverseX))
        self.CCDCamera.GuiderReverseX = False
        if (self.Application.TelescopeConnected
            and self.CCDCamera.GuiderAutoPierFlip):
            # MaxIm is connected to telescope and managing pier flip
            self.ACP_mode = False
        else:
            # Probably in ACP-mode, where it is not recommended to
            # have MaxIm connected to the telescope.  In this case we
            # know we are calibrated on east side looking west with
            # pier flip off.  Unfortunately the opposite of MaxIm's
            # convention, which applies its flip when on east looking
            # west, making the ACP required calibration incompatible
            # with MaxIm pier flip.
            if self.alignment_mode == win32com.client.constants.algGermanPolar:
                self.ACP_mode = True
                log.info('MaxIm is not connected to the telescope or Auto Pier Flip is not set for a GEM. Assuming we are in ACP-mode, with the guider calibrated with the scope on the west looking east.  We need to manage guider pier flip and scope DEC')
            else:
                self.ACP_mode = False

        # Figure out absolute direction for RA & DEC motor motion
        # relative to guide camera.  CCDCamera.GuiderAngle is 0 if the
        # camera is oriented torward north.  The measurement increases
        # CCW
        gang = angle_norm(self.CCDCamera.GuiderAngle, 180)
        aang = angle_norm(self.guider_astrometry['CROTA1'], 180)
        # CROTA* are roll angle perturbations, not camera orientation.
        # If the camera is up-side-down for a pier flip, CDELT2 will
        # be negative and CROTA* will still be small.  Make aang a
        # quantity more like CCDCamera.GuiderAngle
        if np.sign(self.guider_astrometry['CDELT2']) == -1:
            aang = angle_norm(aang + 180, 180)
        log.debug("GuiderAngle: " + repr(gang))
        log.debug("PinPoint solution angle: " + repr(aang))
        if abs(angle_norm(gang - aang, 180)) - 180 > 10:
            raise EnvironmentError('Angle mismatch between Guider PinPoint solution and guider calibration is too large.  Record them both at the same time to ensure match')

        # Check guider calibration rates against mount reported guide rates
        # Create a vector that is as long as we are willing to
        # move in each axis.  The origin of the vector is
        # reference point of the CCD (typically the center)
        # Work in unbinned pixels
        x0 = (self.guider_astrometry['XBINNING']
              * self.guider_astrometry['CRPIX1']
              + self.guider_astrometry['XORGSUBF'])
        y0 = (self.guider_astrometry['YBINNING']
              * self.guider_astrometry['CRPIX2']
              + self.guider_astrometry['YORGSUBF'])
        dt = (self.guider_max_move_multiplier
              * self.CCDCamera.GuiderMaxMoveX)
        dx = (self.CCDCamera.GuiderXSpeed * dt)
              #/ np.cos(np.radians(guider_astrometry['CRVAL2'])))
        dy = self.CCDCamera.GuiderYSpeed * dt
        # GuiderAngle is measured CCW from N according to
        # http://acp.dc3.com/RotatedGuiding.pdf.  But we don't need to
        # use it to calculate the guider rates, since MaxIm has
        # already taken care of that.  Not sure how MaxIm handles
        # non-square pixels, though.   The MaxIm reported guide rates
        # are RA and DEC motor motion in pixels/s.        
        #ang_ccw = self.CCDCamera.GuiderAngle
        ang_ccw = 0
        vec = self.rot((dx, dy), -ang_ccw)
        x1 = x0 + vec[0]
        y1 = y0 + vec[1]
        # Transpose, since we are in pix
        w_coords = self.scope_wcs(((y0, x0), (y1, x1)),
                                  to_world=True,
                                  astrometry=self.guider_astrometry,
                                  absolute=True)
        dra_ddec = w_coords[1, :] - w_coords[0, :]
        self.calculated_guide_rates = np.abs(dra_ddec/dt)
        log.debug("self.calculated_guide_rates: " + repr(self.calculated_guide_rates))
        if not self.telescope_connectable:
            self.guide_rates = abs(self.calculated_guide_rates)
        elif self.Telescope.CanSetGuideRates: 
            # Always assume telescope reported guide rates are
            # correct, but warn if guider rates are off by 10%
            self.guide_rates \
                = np.asarray((self.Telescope.GuideRateRightAscension,
                              self.Telescope.GuideRateDeclination))
            if (np.abs(np.abs(self.calculated_guide_rates[0])
                      - self.Telescope.GuideRateRightAscension)
                > 0.1 * self.Telescope.GuideRateRightAscension):
                log.warning('Calculated RA guide rate is off by more than 10% (scope reported, calculated): ' + str((self.Telescope.GuideRateRightAscension, self.calculated_guide_rates[0])) + '.  Have you specified the correct guider astrometery image?  Have you changed the guide rates changed since calibrating the guider?  Assuming reported telescope guide rates are correct.')
            if (np.abs(np.abs(self.calculated_guide_rates[1])
                      - self.Telescope.GuideRateDeclination)
                > 0.1 * self.Telescope.GuideRateDeclination):
                log.warning('Calculated DEC guide rate is off by more than 10% (scope reported, calculated): ' + str((self.Telescope.GuideRateDeclination, self.calculated_guide_rates[1])) + '.  Have you specified the correct guider astrometery image?  Have you changed the guide rates changed since calibrating the guider?  Assuming reported telescope guide rates are correct.')

        if self.alignment_mode != win32com.client.constants.algGermanPolar:
            return
        
        # The rest of this code deals with getting an absolute sense
        # of the N button relative to our astrometries.  The strategy
        # here is to use the guider calibration, which really does
        # push the N and W buttons to establish directions on the
        # guider.  The Guider astrometry is matched to that doing a
        # pier flip if necessary.  Then the main astrometry is brought
        # to the same side of the pier as the guide.  We then save the
        # result in self.astrometry_pierside
        if abs(angle_norm(gang - aang, 180)) > 10:
            log.debug("Guide calibration and Guider PinPoint solution conducted on opposite sides of the pier")
            self.guider_cal_pinpoint_flip = -1
        else:
            log.debug("Guide calibration and Guider PinPoint solution conducted on same side of the pier")
            self.guider_cal_pinpoint_flip = 1
        # GuiderYSpeed is negative when N is pushed, assuming camera
        # is oriented in the -90 to + 90 range (roughly N).  This is
        # because as the camera points farther north, the image of the
        # star moves farther south.  If camera flips 180,
        # CCDCamera.GuiderAngle = 180 and Yspeed flips sign.  We can
        # use this and our guider_cal_pinpoint_flip to see if we need
        # to flip our guider astrometry so that N is up
        ns = self.CCDCamera.GuiderYSpeed * self.guider_cal_pinpoint_flip
        if ns < 0:
            #self.pinpoint_N_is_up = 1
            log.debug("guider pointpoint up is north")
        else:
            #self.pinpoint_N_is_up = -1
            log.debug("guider pointpoint up is south, flipping guider astrometry")
            pier_flip_astrometry(self.guider_astrometry)
        # --> try to see if this has an effect
        #self.pinpoint_N_is_up = 1
        #log.warning("forcing pinpoint_N_is_up = " + repr(self.pinpoint_N_is_up))

        # Now that the guider N is up [assuming we really need that],
        # get the main on the same side of the pier (which I am sure
        # we need).  It may be at a different angle, but that is OK.
        gps = self.guider_astrometry.get('PIERSIDE')
        mps = self.main_astrometry.get('PIERSIDE')
        if not (gps and mps):
            raise EnvironmentError('PIERSIDE FITS keyword missing from guider and/or main astronemtry')
        if gps == mps:
            log.debug('Main and guider astrometry aligned to same side of pier')
        else:
            log.debug('Main and guider astrometry not aligned on same side of pier, flipping main')
            pier_flip_astrometry(self.main_astrometry)

        if gps == "WEST":
            self.astrometry_pierside = win32com.client.constants.pierWest
            log.debug('N is up in guider when telescope is on West side of pier looking East')
        else:
            self.astrometry_pierside = win32com.client.constants.pierEast
            log.debug('N is up in guider when telescope is on East side of pier looking West')


        #if (np.sign(self.main_astrometry['CDELT2']
        #           * self.guider_astrometry['CDELT2'])
        #    == -1):
        #    log.debug("main pointpoint up is south, flipping main astrometry")
        #    pier_flip_astrometry(self.main_astrometry)



        # I happen to know that east looking west is what
        # Astro-Physics thinks of as normal pointing mode -- north
        # button goes north

        ## Not sure if this matters
        #if (np.abs(a) < 90 or a > 270):
        #    self.guider_north = 1
        #else:
        #    self.guider_north = -1
        #self.guider_north *= np.sign(self.CCDCamera.GuiderYSpeed)
        #if (0 < a and a < 180):
        #    self.guider_east = 1
        #else:
        #    self.guider_east = -1



        ## MaxIm thinks in terms of X and Y, which, if not mirrored in
        ## some way are N = + Y and E = -X.  RA increases to the E,
        ## hence the interest in that coordinate
        #self.guider_east *= -np.sign(self.CCDCamera.GuiderXSpeed)
        #
        ## Reasonable assurance of what side of the pier the
        ## calibration was done on (affects RA sign)
        #if self.ACP_mode:
        #    self.guider_cal_pier_side = win32com.client.constants.pierWest
        #else:
        #    self.guider_cal_pier_side = win32com.client.constants.pierEast


            
        ## --> I don't think it is possible to determine this
        ## Now run the calculation the other way, not taking
        ## out the guider angle, to determine the pier side
        ## when calibration took place
        #ra0 = astrometry['CRVAL1']
        #dec0 = astrometry['CRVAL2']
        #dra_ddec = self.guide_rates * dt
        #ra1 = ra0 + dra_ddec[0]
        #dec1 = dec0 + dra_ddec[1]
        #p_coords = self.scope_wcs((ra1, dec1),
        #                          to_pix=True,
        #                          astrometry=guider_astrometry,
        #                          absolute=True)
        ## remember transpose
        #dp = p_coords[::-1] - np.asarray((x0, y0))
        #
        ## West looking east Manual calibration has X Speed = -3.426 Y Speed 3.4498, angle -179 
        ## East looking west X Speed 3.4284 Y Speed -3.384 angle = 1.097
        #
        ## Calculate our axis flip RELATIVE to guider astrometry.
        ## Note that only one axis is needed, since both flip in
        ## the astrometric sense (only RA flips in the motion
        ## sense, assuming DEC motor commanded consistently in the
        ## same direction)
        #if np.sign(dp[0] / self.CCDCamera.GuiderXSpeed) == 1:
        #    if self.alignment_mode == win32com.client.constants.algGermanPolar:
        #        if astrometry['PIERSIDE'] == 'EAST':
        #            log.debug("self.guider_cal_pier_side = win32com.client.constants.pierEast")
        #        elif astrometry['PIERSIDE'] == 'WEST':
        #            log.debug("self.guider_cal_pier_side = win32com.client.constants.pierWest")
        #else:
        #    if self.alignment_mode != win32com.client.constants.algGermanPolar:
        #        log.error('German equatorial mount (GEM) pier flip detected between guider astrometry data and guider calibration but mount is currently not reporting alignment mode ' + str(self.alignment_mode) + '.  Did you change your equipment?')
        #        # Set our alignment mode, just in case we find
        #        # it useful, but this is a fortuitous catch,
        #        # since clibration and astrometry could have
        #        # been recorded on the same side of the pier.
        #        # --> Ultimately some interface would be
        #        # needed to track mount type and flip state if
        #        # not reported
        #        #self.alignment_mode = win32com.client.constants.algGermanPolar
        #    if astrometry['PIERSIDE'] == 'EAST':
        #        log.debug("self.guider_cal_pier_side = win32com.client.constants.pierWest")
        #    elif astrometry['PIERSIDE'] == 'WEST':
        #        log.debug("self.guider_cal_pier_side = win32com.client.constants.pierEast")
        #    else:
        #        # --> interface would want to possibly record this
        #        log.error('German equatorial mount (GEM) pier flip detected between guider astrometry data and guider calibration but mount not reporting PIERSIDE in guider astrometry file.  Was this file recorded with MaxIm?  Was the mount properly configured through an ASCOM driver when the calibration took place?')

    def horizon_limit(self):
        return (not self.Telescope.Tracking
                or self.Telescope.Altitude < self.horizon_limit_value)


    # For now use self.Application.ShutDownObservatory()
    #def do_shutdown(self):
    #    self.Telescope.Park()
    #    return True
    #
    #def check_shutdown(self):
    #    # if weather: AstroAlert.Weather
    #    #    self.do_shutdown()

    def set_GuiderReverse_and_DEC(self):
        # --> Eventually, I could include GuiderReverseY
        # Tell MaxIm about pier flip and scope DEC manually in case we
        # are in ACP-mode.
        if self.ACP_mode:            
            self.CCDCamera.GuiderReverseX \
                = (self.Telescope.SideOfPier
                   == win32com.client.constants.pierEast)
            log.debug("GuiderReverseX set to " + repr(self.CCDCamera.GuiderReverseX))
            self.CCDCamera.GuiderDeclination = self.Telescope.Declination
            log.debug("Guider DEC set to " + repr(self.CCDCamera.GuiderDeclination))

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
            if self.horizon_limit():
                log.error('Horizon limit reached')
                return False
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
        D.say('old guidebox coords: ' + repr(op_coords[::-1]))
        w_coords = self.scope_wcs(op_coords,
                                  to_world=True,
                                  astrometry=guider_astrometry)
        D.say('world coords of old guidebox: ' + repr(w_coords))
        # When moving the scope in a particular direction, the
        # stars appear to move in the opposite direction.  Since
        # we are guiding on one of those stars (or whatever), we
        # have to move the guide box in the opposite direction
        badp_coords = self.scope_wcs(w_coords - dra_ddec,
                                     to_pix=True,
                                     astrometry=guider_astrometry)
        D.say('badp_coords: ' + repr(badp_coords[::-1]))
        # --> NEW CODE
        # In world coords, we know how far we want to move our guide
        # box.  Calculate the new guidebox position
        p_coords = self.scope_wcs(w_coords + dra_ddec,
                                  to_pix=True,
                                  astrometry=guider_astrometry)
        D.say('New code p_coords: ' + repr(p_coords[::-1]))
        # Now we are in pixel coordinates on the guider.
        # Calculate how far we need to move.
        # There is some implicit type casting here since op_coords
        # is a tuple, but p_coords is an np.array
        dp_coords = p_coords - op_coords
        baddp_coords = badp_coords - op_coords

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
        log.debug('move_with_guide_box: old, bad guider dpix (X, Y): ' + str(baddp_coords[::-1]))
        log.debug('move_with_guide_box: total guider dpix (X, Y): ' + str(dp_coords[::-1]))
        log.debug('norm_dp: ' + str(norm_dp))
        log.debug('Number of steps: ' + str(num_steps))
        if num_steps > self.max_guide_num_steps:
            # We can't do this, so just bomb with a False return
            # and let the caller (usually center) handle it
            log.error('Maximum number of steps (' + str(self.max_guide_num_steps) + ') exceeded: ' + str(num_steps))
            return False
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
            if self.horizon_limit():
                log.error('Horizon limit reached')
                return False
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
        
    def astrometry_pier_flip_state(self):
        """Return -1 if the telecsope is pier flipped relative to astrometry, 1 otherwise"""
        if self.alignment_mode != win32com.client.constants.algGermanPolar:
            return 1
        log.debug('self.Telescope.SideOfPier: ' + repr(self.Telescope.SideOfPier))
        if self.astrometry_pierside == self.Telescope.SideOfPier:
            log.debug('Telescope is on same side as astrometry')
            return 1
        log.debug('Telescope is on opposite side from astrometry')
        return -1

    def MaxIm_pier_flip_state(self):
        """Return -1 if MaxIm is flipping the sense the motors on the RA axis, 1 for non-flipped"""
        # --> I AM NOT SURE THIS IS CORRECT!
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
            # But use CCDCamera.GuiderReverseX anyway, in case it is
            # in use
            if self.CCDCamera.GuiderReverseX:
                flip = -1
            ## --> force
            #flip = 1 
            #log.warning("MaxIm_pier_flip_state forced to = " + repr(flip))
            return flip
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
        if ((self.alignment_mode == win32com.client.constants.algGermanPolar
            and self.Application.TelescopeConnected
            and self.CCDCamera.GuiderAutoPierFlip
            and self.Telescope.SideOfPier == win32com.client.constants.pierWest)
            or self.CCDCamera.GuiderReverseX):
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
        # --> force
        #flip = 1 
        #log.warning("MaxIm_pier_flip_state forced to = " + repr(flip))
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
            
        # Or too little
        dt[np.where(np.abs(dt) < self.min_guide_move_time)] = 0

        log.info('Seconds to move guider in RA and DEC: ' + str(dt))

        # --> TRYING NEW ABSOLUTE dra_ddec and absolute calculated_guide_rates
        # Use our rates to change to time to press E/W, N/S, where
        # E is the + RA direction
        #dt = dra_ddec/self.calculated_guide_rates

        
        dt = dra_ddec/self.guide_rates
        # flip the sense of dt if our absolute astrometry was flipped
        # while making the calculations
        dt *= self.astrometry_pier_flip_state()

        # Do a sanity check to make sure we are not moving too much
        max_t = (self.guider_max_move_multiplier *
                 np.asarray((self.CCDCamera.GuiderMaxMoveX, 
                             self.CCDCamera.GuiderMaxMoveY)))
        if np.any(np.abs(dt) > max_t):
            log.warning('requested move of ' + str(dra_ddec) + ' arcsec translates into move times of ' + str(np.abs(dt)) + ' seconds.  Limiting move in one or more axes to max t of ' + str(max_t))
            dt = np.minimum(max_t, abs(dt)) * np.sign(dt)
            
        # Or too little
        dt[np.where(np.abs(dt) < self.min_guide_move_time)] = 0

        log.info('Seconds to move guider in absolute RA and DEC: ' + str(dt))

        # New code
        #dt[1] *= self.pinpoint_N_is_up
        self.set_GuiderReverse_and_DEC()
        dt[0] *= self.MaxIm_pier_flip_state()

        log.info('Seconds to command MaxIm to move guider in RA and DEC: ' + str(dt))


        # So that we have a consistent begining state, set the
        # GuiderReverse[XY] before we query about motor reverse states
        # in our next statement
        self.set_GuiderReverse_and_DEC()
        # Keep track of whether or not MaxIm is flipping any
        # coordinates for us and flip them back, since we know our
        # dRA and dDEC in the absolute sense.  
        #dt[0] *= self.MaxIm_pier_flip_state()
        # --> using GuiderPinPointSolutionWestofPier.fit now this
        # --> seems to result in another minus sign in ACP mode
        #if self.ACP_mode:
        #    dt[0] *= -1

        if dt[0] > 0:
            # East is positive RA, but east is left (-X) on the sky
            RA_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdMinusX, dt[0])
        elif dt[0] < 0:
            RA_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdPlusX, -dt[0])
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
            DEC_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdPlusY, dt[1])
        elif dt[1] < 0:
            DEC_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdMinusY, -dt[1])
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
        # Set up our astrometry
        we_opened_file = False
        if astrometry is None:
            astrometry = 'main'
        if isinstance(astrometry, str):
            if astrometry.lower() == 'main':
                astrometry = self.main_astrometry
            elif astrometry.lower() == 'guide':
                astrometry = self.guider_astrometry
        if isinstance(astrometry, str):
            if not os.path.isfile(astrometry):
                raise ValueError(astrometry + ' file not found')
            # If we made it here, we have a file to open to get
            # our astrometry from.  Opening it puts the header
            # into a dictionary we can access at any time
            astrometry = fits.open(astrometry)
            we_opened_file = True
        if isinstance(astrometry, fits.HDUList):
            astrometry = astrometry[0].header
        if not isinstance(astrometry, fits.Header):
            raise ValueError('astrometry must be a string, FITS HDUList, or FITS header')
        # Make sure we don't mess up original
        header = astrometry.copy()
        if we_opened_file:
            astrometry.close()
        if header.get('CTYPE1') is None:
            raise ValueError('astrometry header does not contain a FITS header with valid WCS keys.')

        if absolute:
            # In the case of absolute astrometry, we don't have to
            # mess with the astrometry pointing keyword or pier flip
            # relative to our main or guider reference astrometry images
            pier_flip_sign = 1
        else:
            # The non-absolute case means we are going to use the
            # scope's rough RA and DEC to fix the center of the FOV.
            # This is most useful for relative astrometry between two
            # points in the the image
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
            # Check to see if we are pointed on the other side of the
            # mount from our astrometry images
            if self.astrometry_pier_flip_state() == -1:
                pier_flip_astrometry(header)

            # Make sure RA is on correct axis and in the correct units
            if 'RA' in header['CTYPE1']:
                header['CRVAL1'] = RA / 24*360
                header['CRVAL2'] = DEC
            elif 'DEC' in header['CTYPE1']:
                header['CRVAL2'] = RA / 24*360
                header['CRVAL1'] = DEC
        # Fix binning and subframing.  More pixels to the center, but
        # they are smaller.  Also apply pier_flip_sign
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
        header['FLIPAPPL'] = (True, 'Applied pier_flip_sign')
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
        # Since ACP does not want MaxIm connected to the scope, we
        # have to manage all that stuff ourselves
        self.set_GuiderReverse_and_DEC()
        if not self.CCDCamera.GuiderTrack(self.guider_exptime):
            raise EnvironmentError('Attempt to start guiding failed.  Guider configured correctly?')
        # MaxIm rounds pixel center value to the nearest pixel,
        # which can lead to some initial motion in the guider
        self.guider_settle()
        self.guider_commanded_running = True

    def guider_stop(self):
        self.guider_commanded_running = False
        self.CCDCamera.GuiderReverseX = self.previous_GuiderReverseX
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
            # --> __file__ was for testing, I think
            #ObsClassModule = __file__.split('\\')[-1]
            ObsClassModule = ObsClassModule.split('\\')[-1]
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

    # Thanks to
    # https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
    # for instruction.  --> Could make this better to force a with
    def __enter__(self):
        return(self)

    def __exit__(self, exception_type, exception_value, traceback):
        # Turn off the guide box moving system
        self.GuideBoxMoving = False

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
            # https://stackoverflow.com/questions/4152963/get-the-name-of-current-script-with-python
            self._GuideBoxMoverSubprocess \
                = subprocess.Popen(['python',
                                    #'ioio.py',
                                    __file__,
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
            # --> consider making a more gentle exit
            self._GuideBoxMoverSubprocess.terminate()
            with open(self.guide_box_log_file, 'a') as l:
                l.write(Time.now().fits
                        + '---- GuideBoxMover Stopped ----- \n')

    def GuideBoxCommander(self, pix_rate=np.zeros(2)):
        """Input desired telescope motion rate in Y, X main camera pixels per sec, outputs command file for GuideBoxMover in degrees per second"""
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

    def diff_flex(self):
        """-->Eventually this is going to read a map/model file and calculate the differential flexure to feed to GuideBoxCommander.  There may need to be an additional method for concatenation of this flex and the measured flexure.  THIS IS ONLY FOR JUPITER DEC RIGHT NOW"""
        plate_ratio = 4.42/(1.56/2)
        if (-40 < self.MD.Telescope.Declination
            and self.MD.Telescope.Declination < +10
            and self.MD.Telescope.Altitude < 30):
            # Change from guider pixels per 10s to main camera pixels per s
            dec_pix_rate = -0.020/10 * plate_ratio
            # Note Pythonic transpose
            return self.GuideBoxCommander(np.asarray((dec_pix_rate, 0)))
        # For now, don't guess, though I could potentially put
        # Mercury in here
        return self.GuideBoxCommander(np.asarray((0, 0)))

    # --> I'll probably want a bunch of parameters for the exposure
    def center(self,
               HDUList_im_fname_ObsData_or_obj_center=None,
               desired_center=None,
               current_astrometry=None,
               scaling_astrometry=None,
               ignore_ObsData_astrometry=False,
               recursive_count=0,
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
        if self.MD.horizon_limit():
            log.error('Horizon limit reached')
            return False

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
                  and not ignore_ObsData_astrometry):
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
        log.debug('world coordinates of obj_center and desired_center: ' + repr(w_coords))

        dw_coords = w_coords[1,:] - w_coords[0,:]
        if self.MD.CCDCamera.GuiderRunning:
            log.debug('CENTERING TARGET WITH GUIDEBOX MOVES')
            if not self.MD.move_with_guide_box(dw_coords):
                if recursive_count > 1:
                    log.error('center: Failed to center target using guidebox moves after two tries')
                    return False                        
                log.debug('TURNING GUIDER OFF AND CENTERING WITH GUIDER SLEWS')
                self.MD.guider_stop()
                self.MD.guider_move(dw_coords)
                # --> Need to add logic to capture guider stuff,
                # though filter should be the same.  It is just
                # the exposure time that I might want to have
                # carried over, though I will eventually figure
                # that out myself.
                log.debug('RESTARTING GUIDER AND CENTERING WITH GUIDEBOX MOVES')
                self.MD.guider_start()
                recursive_count += 1
                self.center(recursive_count=recursive_count)
        else:
            log.debug('CENTERING TARGET WITH GUIDER SLEWS')
            self.MD.guider_move(dw_coords)
        return True

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
            if self.MD.horizon_limit():
                log.error('Horizon limit reached')
                return False
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
            # Here is where we actually call the center algorithm
            if not self.center(O):
                log.error('center_loop: could not center target')
                return False
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

def cmd_test_center(args):
    if sys.platform != 'win32':
        raise EnvironmentError('Can only control camera and telescope from Windows platform')
    P = PrecisionGuide(args.ObsClassName,
                       args.ObsClassModule) # other defaults should be good
    if args.x and args.y:
        # NOTE TRANSPOSE!
        desired_center = (args.y, args.x)
    else:
        desired_center = None
    log.debug('Desired center (X, Y; binned) = ' + str(desired_center[::-1]))

    P.center(desired_center=desired_center)
    log.debug('STARTING GUIDER') 
    P.MD.guider_start()
    log.debug('CENTERING WITH GUIDEBOX MOVES') 
    P.center(desired_center=desired_center)


def cmd_guide(args):
    MD = MaxImData()
    if args.stop:
        MD.guider_stop()
    else:
        MD.guider_start(exptime=args.exptime, filter=args.filter)

# --> Eventually, I would like this to accept input from other
# --> sources, like a flexure model and ephemeris rates

# --> This doesn't work when the file changes faster than it would do
# --> a guidebox command
def GuideBoxMover(args):
    log.debug('Starting GuideBoxMover')
    MD = MaxImData()
    last_modtime = 0
    while True:
        if MD.horizon_limit():
            log.error('GuideBoxMover: Horizon limit reached')
            return False

        # --> Make this sleep time variable based on a fraction of the
        # --> expected motion calculated below
        time.sleep(1)
        # Wait until we have a file
        # --> Consider making lack of file an exit condition
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
            if not MD.move_with_guide_box(dra_ddec_rate * dt ):
                log.error('GuideBoxMover MD.move_with_guide_box failed')
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
    log.debug('TURNING ON (PASSIVE) GUIDEBOX MOVER SYSTEM')
    P.diff_flex()
    # Center with guide box moves
    #log.debug('NOT CENTERING WITH GUIDE BOX MOVES WHILE DOING LARGE GUIDE RATE EXPERIMENT ') 
    log.debug('CENTERING WITH GUIDEBOX MOVES') 
    P.center_loop()
    # Put ourselves in GuideBoxMoving mode (starts the GuideBoxMover subprocess)
    #log.debug('STARTING GuideBoxMover')
    # --> this is a confusing name: mover/moving
    #P.GuideBoxMoving = True
    while True:
        fname = uniq_fname(basename, d)
        log.debug('COLLECTING: ' + fname)
        # --> change this back to P.acquire_image to test measurement system
        P.MD.acquire_im(fname,
                        exptime=args.exptime,
                        filt=args.filt)
        log.debug('UPDATING (PASSIVE) GUIDEBOX MOVER SYSTEM')
        P.diff_flex()
        # --> Just for this could eventually do a sleep watchdog or
        # --> guider settle monitor....
        time.sleep(7)

def MaxImCollector(args):
    P = PrecisionGuide()
    P.MaxImCollector()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command-line control of MaxIm")
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

    test_center_parser = subparsers.add_parser(
        'test_center', help='test center code')
    test_center_parser.add_argument(
        'x', type=float, nargs='?', default=None, help='desired_center X')
    test_center_parser.add_argument(
        'y', type=float, nargs='?', default=None, help='desired_center Y')
    test_center_parser.add_argument(
        '--ObsClassName', help='ObsData class name')
    test_center_parser.add_argument(
        '--ObsClassModule', help='ObsData class module file name')
    test_center_parser.set_defaults(func=cmd_test_center)




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

    # Final set of commands that makes argparse work
    args = parser.parse_args()
    # This check for func is not needed if I make subparsers.required = True
    if hasattr(args, 'func'):
        args.func(args)
