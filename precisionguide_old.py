#--> This code was developed using Anaconda3 installed as
#--> administrator and with PATH option selected during the install so
#--> that python can be used from a Windows command line.  NOTE: to
#--> get command line arguments passed, which is essential for this
#--> code, you need need to edit registry to make
# Computer\HKEY_CLASSES_ROOT\Applications\python.exe\shell\open\command
# "C:\ProgramData\Anaconda3\python.exe" "%1" %*
# Thanks to
# https://stackoverflow.com/questions/29540541/executable-python-script-not-take-sys-argv-in-windows

# Alternately, you can just call python (with the full path to python
# if needed) and then specify the full path to the module and then the
# module's arguments


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
from astropy.coordinates import EarthLocation, AltAz, get_sun

if sys.platform == 'win32':
    # --> also check out pythonnet
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


# In principle, it is possible to use only MaxIm guiding stuff for
# this, alleviating the need for us to connect directly to the
# telescope.  In practice, with a GEM and for setting DEC conveniently
# in the guider, MaxImControl really needs to be connected to the scope.
# Since unexpected behavior may result if there is not a hard failure
# when a GEM is not connected, indicate here whether or not you want
# that hard failure
telescope_must_be_connectable = True

# These are necessary for GEMs because MaxIm does not reveal the
# contents of the Camera Control -> Guide Tab -> Settings dialog box
# -> Advanced Tab -> Guider Motor Control radio buttons to the
# scripting interface.  As explained in guider_motor_reverse_setup,
# when using ACP or otherwise not having MaxIm connected to the
# telescope, we need to manage motor reversal ourselves
guider_motor_control_reverseX = True
guider_motor_control_reverseY = False
# Misalignment in deg
guider_cal_astrometry_max_misalignment = 10

horizon_limit = 8.5 # Reasonable western horion until you get south
sundown_limit = -12 # Nautical twilight

# --> I may improve this location or the technique of message passing
hostname = socket.gethostname()
if hostname == "snipe" or hostname == "byted":
    raw_data_root = '/data/io/IoIO/raw'
elif hostname == "greyhound" or hostname == "gigabyte":
    # --> This doesn't work.  I need Unc?
    #raw_data_root = '//snipe/data/io/IoIO/raw'
    raw_data_root = r'\\snipe\data\io\IoIO\raw'
    default_telescope = 'ScopeSim.Telescope'
elif socket.gethostname() == "IoIO1U1":
    raw_data_root = r'C:\Users\PLANETARY SCIENCE\Desktop\IoIO\data'
    # --> Eventually, it would be nice to have this in a chooser
    default_telescope = 'AstroPhysicsV2.Telescope'
# For weather synchronization with ACP
ACPUtil = 'ACP.Util'
default_guide_box_command_file = os.path.join(raw_data_root, 'GuideBoxCommand.txt')
default_guide_box_log_file = os.path.join(raw_data_root, 'GuideBoxLog.txt')

run_level_main_astrometry = os.path.join(
    raw_data_root, '2023-08_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2023-07_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2022-09_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2022-05_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2021-04_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2021-03_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2020-09_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2020-03_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2019-04_Astrometry/Main_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2019-04_Astrometry/Main_Astrometry_West_of_Pier.fit')
    #raw_data_root, '2019-02_Astrometry/PinPointSolutionEastofPier.fit')
    #raw_data_root, '2019-02_Astrometry/PinPointSolutionWestofPier.fit')
    #raw_data_root, '2018-04_Astrometry/PinPointSolutionEastofPier.fit')

run_level_guider_astrometry = os.path.join(
    raw_data_root, '2023-08_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2023-07_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2022-09_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2022-05_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2021-04_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2021-03_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2020-09_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2020-03_Astrometry/Guider_Astrometry_East_of_Pier.fit')
    #raw_data_root, '2019-04_Astrometry/Guider_Astrometry_West_of_Pier.fit')
    #raw_data_root, '2019-02_Astrometry/GuiderPinPointSolutionWestofPier.fit')
    #raw_data_root, '2019-02_Astrometry/GuiderPinPointSolutionEastofPier.fit')    
    #raw_data_root, '2018-04_Astrometry/GuiderPinPointSolutionWestofPier.fit')
    #raw_data_root, '2018-01_Astrometry//GuiderPinPointSolutionEastofPier.fit')

def angle_norm(angle, maxang):
    """Normalize an angle to run up to maxang degrees"""
    angle += 360
    angle %= 360
    if angle > maxang: # handles 180 case
        angle -= 360
    return angle

def iter_linfit(x, y, max_resid=None):
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
            coefs = iter_linfit(x[goodc], y[goodc])
    return coefs
    
# I am either phasing this out or I could potentially make it work
# with __enter__ and __exit__ for a context manager
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

def pier_flip_astrometry(header_in):
    """Adjust FITS astrometry CD* keywords to emulate a pier flip (rotate FOV 180 deg)
        header_in : input FITS header
	return value : copy of header_in with CD* keywords adjusted"""
    header = header_in.copy()
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
    return header

# --> Really what I think I want is a PGData for all of the center and
# --> rate stuff.  That will clean up the ObsData property and
# --> __init__
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
        self._HDU_unbinned = None
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
        """Close open file, deference large arrays"""
        if self._we_opened_file:
            self.close_fits()
        del self.HDUList
        del self._HDU_unbinned

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
        
    def im_unbinned(self, a):
        """Returns an unbinned version of a.  a must be same shape
        as the primary HDU image
        """
        assert a.shape == self.HDUList[0].data.shape
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
        
    @property
    def HDU_unbinned(self):
        """Returns an unbinned version of the primary HDU image or the primary HDU image if it is not binned.
        """
        if self._HDU_unbinned is not None:
            return self._HDU_unbinned
        self._HDU_unbinned = self.im_unbinned(self.HDUList[0].data)
        return self._HDU_unbinned

    def close_fits(self):
        if self.HDUList.fileinfo is not None:
            self.HDUList.close()
            self._we_opened_file = None

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
    ## Worst case scenario, we calculate them with MaxImControl.scope_wcs
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

class FakeWeather():
    def __init__(self):
        self.Safe = True

# --> This really needs to be in MaxImControl
class ObsTerminator():
    """Determines when an observation will end"""
    def __init__(self,
                 Tend=None, # unix seconds (time.time() + exposure_time)
                 horizon_limit=horizon_limit,
                 sundown_limit=sundown_limit):
        self.Tend = Tend
        self.horizon_limit = horizon_limit
        self.sundown_limit = sundown_limit

    def below_horizon(self, MC, horizon_limit):
        if (not MC.Telescope.Tracking
            or MC.Telescope.Altitude < horizon_limit):
            log.debug(f'Telescope Altitude = {MC.Telescope.Altitude} Telescope Tracking: {MC.Telescope.Tracking}')
            if (not MC.Telescope.Tracking and
                MC.Telescope.Altitude > horizon_limit + 3):
                log.error('Telescope is not tracking for some weird reason, ignoring')
                return False
            log.info('Telescope horizon limit reached')
            return True
        return False

        # return (not MC.Telescope.Tracking
        #         or MC.Telescope.Altitude < MC.horizon_limit_value)

    def end_of_obs(self, MC):
        t = time.time()
        if self.Tend is not None and t > self.Tend:
            return True
        # --> Hope telescope location is set correctly.  If not able
        # --> to do this, should have a property somewhere for user
        e_loc = EarthLocation.from_geodetic(
            MC.Telescope.SiteLongitude,
            MC.Telescope.SiteLatitude,
            MC.Telescope.SiteElevation,
            'WGS84')
        t = Time(t, format='unix', location=e_loc)
        sc =  get_sun(t)
        alt_az = sc.transform_to(
            AltAz(obstime=t, location=e_loc))
        return (self.below_horizon(MC, self.horizon_limit)
                or alt_az.alt.value > self.sundown_limit)

class MaxImControl():
    """Controls MaxIm DL via ActiveX/COM events.

    Notes: 

    MaxIm camera, guide camera, and telescope must be set up properly
    first (e.g. you have used the setup for interactive observations).
    Even so, the first time this is run, keep an eye out for MaxIm
    dialogs, as this program will hang until they are answered.  To
    fix this, a wathdog timer would need to be used.

    Technical note for downstream object use: we don't have access to
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
                 default_filt=default_filt,
                 obs_terminator=None):
        if sys.platform != 'win32':
            raise EnvironmentError('Can only control camera and telescope from Windows platform')
        self.main_astrometry = main_astrometry
        if self.main_astrometry is None:
            self.main_astrometry = run_level_main_astrometry
        if isinstance(self.main_astrometry, str):
            with fits.open(self.main_astrometry) as HDUList:
                self.main_astrometry = HDUList[0].header
        self.guider_astrometry = guider_astrometry
        if self.guider_astrometry is None:
            self.guider_astrometry = run_level_guider_astrometry
        if isinstance(self.guider_astrometry, str):
            with fits.open(self.guider_astrometry) as HDUList:
                self.guider_astrometry = HDUList[0].header

        self.default_filt = default_filt
        self.obs_terminator = obs_terminator or ObsTerminator()

        self.alignment_mode = None
        self.guider_cal_pierside = None
        self.pier_flip_on_side = None
        # Pattern this after Telescope.GuideRateRightAscension and
        # Telescope.GuideRateDeclination, which are the telescope
        # guide rates in deg/s, assuming they can be read from the
        # telescope.  Here we can store them as an np.array
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
        
        # --> Kind of a hack to have this here.  Eventually want to
        # --> integrate MaxImControl object with ACP better
        self.ACPUtil = None
        self.weather_server = None

        # Create containers for all of the objects that can be
        # returned by MaxIm.  We'll only populate them when we need
        # them.  Some of these we may never use or write code for
        self.Application = None
        self.CCDCamera = None
        self.Document = None
        self.Telescope = None
        # This helps with logic that I don't use yet
        self.telescope_connectable = None
        # This helps PrecisionGuide dance around ACP controlling the
        # focuser directly
        self.focuser_previously_connected = None
        self.previous_guider_filter = None
        self.original_GuiderAutoPierFlip = None

        # There is no convenient way to get the FITS header from MaxIm
        # unless we write the file and read it in.  Instead allow for
        # getting a selection of FITS keys to pass around in a
        # standard astropy fits HDUList
        self.FITS_keys = None
        self.HDUList = None
        self.required_FITS_keys = ('DATE-OBS', 'EXPTIME', 'EXPOSURE', 'XBINNING', 'YBINNING', 'XORGSUBF', 'YORGSUBF', 'FILTER', 'IMAGETYP', 'OBJECT')

        # We can use the CCDCamera.GuiderMaxMove[XY] property for an
        # indication of how long it is safe to press the guider
        # movement buttons, but "Max" is not very much -- 0.1 - 3.0s
        # according to MaxIm documentation, so be liberal with this
        self.guider_max_move_multiplier = 20
        #  --> Too little motion seems to freeze the system, at
        # least sometimes
        self.max_guide_num_steps = 8
        self.connect()
        if self.telescope_connectable:
            self.alignment_mode = self.Telescope.AlignmentMode
        else:
            # --> Eventually this warning might go away as we might
            # --> use some property of our own to track the mount type
            log.error("Mount is not connected -- did you specify one in setup [currently the software source code]?  If you have a German equatorial mount (GEM), this software will likely not work properly upon pier flips [because code has not yet been written to let you specify the mode of your telescope on the fly].  Other mount types will work OK, but you should keep track of the Scope Dec. box in MaxIm's Guide tab.")

        # This sets self.pier_flip_on_side and makes sure the guider
        # astrometry is aligned with the guider cal
        self.guider_motor_reverse_setup()
        # Now line up our main astrometry with the guider in case it
        # was recorded on the opposite side of a GEM flip
        main_astrometry_pierside = self.main_astrometry.get('PIERSIDE')
        if self.alignment_mode == win32com.client.constants.algGermanPolar:
            if main_astrometry_pierside is None:
                raise EnvironmentError('Connected to GEM mount yet no PIERSIDE was recorded in main astrometry FITS header.  Was MaxIm connected to the telescope when the astrometry was recorded?')
            if main_astrometry_pierside != self.guider_cal_pierside:
                self.main_astrometry \
                    = pier_flip_astrometry(self.main_astrometry)
        self.check_guider_speeds()
        self.previous_GuiderReverseX = self.CCDCamera.GuiderReverseX
        self.previous_GuiderReverseY = self.CCDCamera.GuiderReverseY

    def __enter__(self):
        return(self)

    def __exit__(self, exception_type, exception_value, traceback):
        # --> Try to get telescope disconnected properly so APCC can
        # --> exit without having to kill it
        if self.telescope_connectable:
            self.Telescope.Connected = False
        # This kills the link to the CCD camera unless
        # self.CCDCamera.DisableAutoShutdown = True by someone.  ACP
        # sets this to True but FocusMax does not.  See notes in
        # IoIO.notebk about C-c dance if you want to have
        # CCDCamera.DisableAutoShutdown = False and not kill camera
        # link
        if self.CCDCamera:
            self.guider_stop()
            # --> consider putting this back to 0 instead of previous
            # --> and really this should be in the coronagraph
            # --> subclass instead of here, but that is a project for
            # --> a later date
            self.CCDCamera.GuiderFilter = self.previous_guider_filter
            #self.CCDCamera.LinkEnabled = False
        # Put the MaxIm focuser connection back to its previous state
        self.Application.FocuserConnected = self.focuser_previously_connected
        self.CCDCamera.GuiderAutoPierFlip = self.original_GuiderAutoPierFlip
        # --> Not sure if I need to do these or if they mess it up worse
        self.Application = None
        self.CCDCamera = None
        self.Telescope = None

    def connect(self):
        """Link to weather safety monitor, telescope, CCD camera(s), filter wheels, etc."""
        try:
            self.ACPUtil = win32com.client.Dispatch(ACPUtil)
            self.weather_server = self.ACPUtil.Weather
            test = self.ACPUtil.Weather.Safe
        except Exception as e:
            log.error('Received the following error: ' + str(e))
            log.warning('Seems to be an ACP weather server problem, forging ahead with no weather protection!')
            self.weather_server = FakeWeather()

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
        # --> ACP doesn't like MaxIm being connected to the focuser,
        # but some of my IoIO things do need that for the time being
        # --> This is really at the wrong level -- I should have
        # ACP_IPT_Na_R take care of this as an object
        self.focuser_previously_connected = self.Application.FocuserConnected 
        self.getCCDCamera()
        self.CCDCamera.LinkEnabled = True
        if self.CCDCamera.LinkEnabled == False:
            raise EnvironmentError('Link to camera hardware failed.  Is the power on to the CCD (including any connection hardware such as USB hubs)?')
        # Let the guider filter and AutoPierFlip be put back to
        # previous states after we use it
        self.previous_guider_filter = self.CCDCamera.GuiderFilter
        self.original_GuiderAutoPierFlip = self.CCDCamera.GuiderAutoPierFlip
        # Keep CCD link up after script exits (thanks to Daniel!)
        self.CCDCamera.DisableAutoShutdown = True

    def getTelescope(self):
        if self.Telescope is not None:
            return
        try:
            self.Telescope = win32com.client.Dispatch(default_telescope)
            self.telescope_connectable = True
        except:
            if telescope_must_be_connectable:
                raise EnvironmentError('Error instantiating telescope control object ' + default_telescope + '.  Is the telescope on and installed?')
            else:
                log.warning('Not able to connect to telescope.  Some features like auto pier flip for German equatorial mounts (GEMs) and automatic declination compensation for RA motions will not be available. --> eventually make some sort of menu to select mount type or grep that from ACP config')
                self.telescope_connectable = False
        else:
            # Catch any other weird errors
            assert isinstance(self.Telescope, win32com.client.CDispatch)
        
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

    def guider_motor_reverse_setup(self):
        """Set up property for guiding and moving the telescope with guider
        slews.

        Details: Set up property so guiding works regardless of
        whether or not MaxIm is connected to the telescope and
        regardless of how the guider was calibrated.  Also set up
        property so we can use use MaxIm's control of the guider
        inputs of the mount to do small motions to center our target.

        To briefly review how autoguiding works, MaxIm measures the
        current X and Y position of the guide star and calculates
        the number of pixels it needs to move to get to the desired
        guide position.  Of course, MaxIm doesn't physically move
        pixels in the camera, it moves the telescope.  So there
        needs to be some sort of calibration that goes between the
        two.  GuiderXSpeed, GuiderYSpeed, and GuiderAngle is that
        calibration.

        MaxIm calibrates the guider by finding the brightest star in
        the FOV.  The +X and +Y buttons are operated such that the
        star is moved in an "L" shape.  First in the +-X direction
        and then in the +/-Y direction.  It is up to the user to
        connect the X and Y leads that MaxIm is operating to the RA
        and DEC leads of the mount.  Generally +X is considered +RA,
        which is E, or *left* as you look at the sky with no camera
        mirroring or rotation effects and the +Y lead is north.
        Because it is the mount that moves E and N, the star, which
        is fixed on the sky, appears to move W and S.  MaxIm, of
        course, knows this and applies the appropriate minus sign
        when calculating the GuiderXSpeed and GuiderYSpeed
        quantities.  These are the speed in guider camera pixels per
        second the mount moves when X and Y are pressed.  Note some
        confusion may arise because MaxIm pixel coordinates (0,0)
        are at the *top* left of the image, which is the up/down
        mirror from normal Cartesian coordinates.  Thus, if N is up
        as displayed by MaxIm, +N ends up being -Y in MaxIm
        GuiderYSpeed.  A final quantity is necessary: GuiderAngle,
        the angle CCW from N of the +Y telescope motion direction.

        Whew!

        Now add the complication of a GEM.

        A GEM effectively operates in two modes because it has to
        carry the telescope on one side or the other of the pier to
        be able to view all of the sky
        https://ascom-standards.org/Help/Platform/html/P_ASCOM_DeviceInterface_ITelescopeV3_SideOfPier.htm
        has a discussion of how this is somewhat awkwardly
        implemented in ASCOM.  Add to this the complication that
        different GEM mounts do different things to their coordinate
        systems when the mount is on the two different sides.  RA is
        generally left alone, since on either side of the pier, the
        RA axis still has to rotate toward the west.  But DEC
        conventions vary.  When the mount flips and points the
        telscope at the same place on the sky (issues of
        counterweight up aside), RA flips 180 and DEC flips 180,
        resulting in the telescope tube *rotating* 180.  Some
        mounts, like my Astro-Physics 1100, leave the DEC motors
        connected in the same sense on both sides of the mount.
        This results in somewhat counter-intuitive behavior of the
        telescope tube when you press N.  When near the equator, on
        one side of the mount (ASCOM pierWest, looking east), N
        moves you torward N, on the other (e.g. when I am in Park
        4), it moves you toward S.  Other mounts (Paramount, I
        think) define the preferred N goes N side as ASCOM pierEast.
        Still other mounts (PlaneWave, I think) flip N and S when
        you flip the mount, so N always moves the tube N.

        No matter what the mount does with N/S, one thing never
        changes on pier flip: the tube rotates 180 degrees such that
        if N was up before, it is now down.  And if E was left in
        the camera, it is now right.  When doing absolute
        astrometry, this matters a lot, but when guiding, it gets a
        little simpler.  On mounts that don't reverse the N/S motor
        sense on flip (e.g. Astro-Physics, Paramount), because both
        astrometric N/S and motion N/S have flipped, the guider can
        blithly press the same N or S button to get the star to move
        the expected direction in camera Y coordinates.  X is not so
        lucky.  RA has really flipped in the camera but not on the
        mount.  Someone needs to keep track of that flip.  As
        described below, MaxIm has that capacity both for
        interactive and scripted use.
        
        For interactive use, MaxIm uses the Pier Flip box in the
        Camera->Guide tab together with the Guider Motor Control On
        Pier Flip radio buttons in the Guide -> Settings -> Advanced
        tab to let the user fix guider pier flip issues.  These
        controls tell MaxIm to swap the RA and/or DEC connections on
        pier flip.  For mounts that properly report their pier flip
        state via ASCOM, the Auto Pier Flip box can be checked,
        which basically keeps track of checking the Pier Flip box
        for you.  For scripting use, the GuiderReverseX and
        GuiderReverseY property can be used.  More on those later.

        The problem with the "Pier Flip" nomenclature is, of course,
        that it is not absolute.  "Pier Flip" relative to what?  At
        some level it doesn't matter.  If you calibrated the guider on
        one side of the pier and then do a pier flip, then you need to
        tell the guider you have pier flipped.  Fiddle with the Pier
        Flip radio buttons in the Guider Settings Advanced tab until
        guiding works and you are done.  This also works for
        AutoPierFlip, but it happens to have an absolute:

        MaxIm "Pier Flip" is ASCOM pierWest looking east (through the
        pole).  MaxIm normal (no pier flip) is pierEast looking west

        So if your mount reports its pierside state properly via
        ASCOM, it is probably best to do your first guider
        calibration on pierEast looking west (normal).  Then enable
        MaxIm's AutoPierFlip, do a pier flip, guide, and poke the
        radio buttons in Guider Settings Advanced until guiding
        works.  Once you have those radio buttons set, you can
        recalibrate the guider on any side of the mount because
        MaxIm will automatically reverse the motor connections as
        per your specifications as it is doing the calibration.
        Sure you will get GuiderAngle of +180 on opposite sides of
        the mount, but if you take that into consideration with the
        guider speeds, everything will end up looking like it is
        calibrated relative to the normal ASCOM pointing state.

        Whew!

        Now enter ACP.  

        It is not clearly stated why but
        http://acp.dc3.com/RotatedGuiding.pdf explains that ACP has
        chosen the opposite of the mount from MaxIm to act as the
        guider calibration standard.  ACP users are instructed to
        put the mount on pierWest looking east, turn AutoPierFlip
        off, and calibrate.  This is unfortunate, since it
        completely breaks MaxIm guiding.  Furthermore, MaxIM
        profiles can't be used to fully change between ACP guiding
        mode and MaxIm guiding mode since the profile doesn't
        preserve the AutoPierFlip state button state.

        Since the ACP rotated guiding document states that the ACP
        pierWest guider calibration standard is not going to change,
        we need to deal with it here.  We also want to make it
        convenient to be used in normal MaxIm mode.  Fortunately, we
        can use the hints summarized below to let us figure out which
        mode we are in.  We can then use the GuiderReverseX and
        GuiderReverseY property to manually reverse the sense of the
        guider motor conenctions.

        Before describing how we can figure out what side of the mount
        the guider was calibrated on and what mode of guiding we are
        in, we need to describe the other thing we do with this
        software: move the mount so that in the main camera, our
        target is centered where we want it.  The main camera pixel
        coordinates of these quantities are stored in the obj_center
        and desired_center property of the ObsData object or its
        subclass (e.g. CorObsData).  Using the main camera PinPoint
        sample astrometry solution, run_level_main_astrometry, we can
        translate these pixel coordinates into absolute astrometric
        coordinates and figure out how far to move the mount to get
        our target centered.  

        There are several ways we can move the mount: enter in new
        coordinates, push mount motion buttons with ASCOM direct, or
        push mount motion buttons with whatever MaxIm uses for the
        guider.  Since we the user has already figured out how to
        connect MaxIm to the mount to make the guider work and there
        is an easy way to command MaxIm to use that connection via the
        scripting interface (CCDCamera.GuiderMove()), we will do the
        later.  The problem with this is that motor reverse commands
        effect these motor control commands.  So if we want to move
        the telescope in astrometric coordinate space, we need to keep
        track of the motor flips and undo them when.  --> Alternately,
        we could translate our astrometric coordinates into guider
        pixel space, apply a guider speed and angle coordinate
        translation and the flips would take care of themselves.

        Whew!

        Now we can write down the various cases of what mode we are
        in, how the guider was calibrated, how pier flipping is
        tracked and how we need to track that pier flipping to reverse
        its effect when we want to use the MaxIm guider motion commands.

        (1) MaxIm guiding mode:
            Telescope connected to MaxIm
            AutoPierFlip enabled
            if cal pierEast, directions match astrometry rotated to pierEast
            [Guider motor connections can be grepped in this configuration]
            if cal pierWest, E/W and/or N/S will be flipped,
                depending on Guider Motor Control state
            MaxIm will manage all pier flip for the guider
            Observing pierWest, unflip E/W and/or N/S for guider moves

        (1a) MaxIm guiding mode but scope not connected for some reason
             We have to manage all pier flip stuff for the guider via
             GuiderReverseX and GuiderReverseY
             Observing pierWest, unflip E/W and/or N/S for guider moves

        (2) ACP guiding mode:
            telescope should not be connected
            AutoPierFlip probably disabled
            Astrometry directions flipped to pierWest should match
                guider cal directions
            We have to manage all pier flip stuff for the guider via
            GuiderReverseX and GuiderReverseY
            Observing pierEast, unflip E/W and/or N/S for guider moves

        The MaxIm scripting interface does not provide any property
        for the Guider Motor Control radio buttons, so we duplicate
        them here as global variables
        (guider_motor_control_reverse[XY]), though there is one case
        when we can grep them out of the guider cal.  If this gets
        closely integrated with ACP, we might also be able to get
        these from its configuration options.

        """
        # The guider astrometry calibration gives us absolute knowledge
        # of which directions are which in the guide camera, assuming
        # the images were recorded with MaxIm connected to the telescope
        # and the telecsope reports pier side.  The side on which guider
        # calibration is done is not noted by MaxIm.  However, we can
        # compare the GuiderAngle and the north angle in the guider
        # astrometry and use the guider astrometry PIERSIDE to tell us
        # exactly which side the guider was calibrated on.  With the
        # guider astrometry lined up with the guider calibration, we can
        # see which way the cal directions point to figure out whether
        # or not MaxIm was doing any motor reversing while the guider
        # calibration was being done
        
        # This will be None if the mount is not a GEM or was not
        # connected when astrometry was recorded.  Save it for later...
        guider_astrometry_pierside = self.guider_astrometry.get('PIERSIDE')

        # Find the angle of N in the astrometry image relative to
        # MaxIm's N = -Y convention.  This is very confusing because
        # for MaxIm, in a nominal N up, E left configuration, N is
        # toward -Y, since MaxIm plots Y increasing down.  In the FITS
        # standard, if the cardinal directions are aligned to pixel X,
        # Y, increasing Y means moving N.  Ultimately, which direction
        # N is plotted doesn't really matter: the user tweaks things
        # so N is up for them and the FITS CDELT will adjust.  If
        # plotted in Cartesian coordinates (e.g. IDL, IRAF), the user
        # will align N up so CDELT2 is positive.  For MaxIm and other
        # (0,0) top left oriented displays the user will align N up so
        # CDELT2 is negative.  But here it matters!  We need to
        # emulate a MaxIm N vector with the FITS standard so we can
        # compare our astrometry image to MaxIm's guider calibration.
        # Start with the regular FITS WCS standard, using a
        # healthy-size angle to avoid numeric problems
        dp = self.scope_wcs((0, 10.),
                            to_pix=True,
                            astrometry=self.guider_astrometry,
                            absolute=True,
                            delta=True)
        # Now negate Y to align with MaxIm's -Y = up standard, keeping
        # in mind that scope_wcs returns Y,X.
        dp[0] *= -1
        # arctan2 takes y, x and uses the normal math angle definition
        # (+X axis = 0 deg)
        aang = np.degrees(np.arctan2(dp[0], dp[1]))
        # Convert angle to N up, +/-180 (already increasing CCW)
        aang = angle_norm(aang-90, 180)
        # Now compare to guider calibration N up angle.  This has yet
        # to match precisely on, probably because Bob has private
        # distortions
        gang = angle_norm(self.CCDCamera.GuiderAngle, 180)
        log.debug("PinPoint solution angle: " + repr(aang))
        log.debug("GuiderAngle: " + repr(gang))
        dang = abs(angle_norm(gang - aang, 180))
        if ((dang < 90
             and dang > guider_cal_astrometry_max_misalignment)
            or ((dang > 90)
                and dang < 180 - guider_cal_astrometry_max_misalignment)):
            raise EnvironmentError('Angle mismatch between Guider PinPoint solution and guider calibration is too large.  Record them both at the same time to ensure match')
        if dang < 90:
            guider_cal_astrometry_aligned = True
        else:
            # In the case that guider cal and guider pinpoint solution
            # are on opposite sides of the pier, we can check for some
            # errors
            if not self.telescope_connectable:
                raise EnvironmentError('GEM pier flip between guider calibration and guider PinPoint solution detected, yet telescope is not connectable.  GEM mounts really need to be connected for all of this to work.  Alternately, you could calibrate on one side and just stick to that side, or I may write code to specify GEM flips on the fly like MaxIm does')
            if self.alignment_mode != win32com.client.constants.algGermanPolar:
                raise EnvironmentError('GEM pier flip between guider calibration and guider PinPoint solution detected yet mount is not reporting that it is a GEM')
            if guider_astrometry_pierside is None:
                raise EnvironmentError('GEM pier flip between guider calibration and guider PinPoint solution detected, yet no PIERSIDE was recorded in guider astrometry FITS header.  Was MaxIm connected to the telescope when the astrometry was recorded?')
            # If we made it here, there are no errors and we know that
            # the guider cal and guider pinpoint solution are on
            # opposite sides of the pier
            guider_cal_astrometry_aligned = False
        if self.alignment_mode != win32com.client.constants.algGermanPolar:
            log.debug('non-GEM mount, guider astrometry and guider cal are lined up')
            return

        # If we made it here, we are a GEM Rotate our guider
        # astrometry to line up with the guider calibration direction
        if guider_astrometry_pierside is None:
            raise EnvironmentError('Currently connected mount reports it is a GEM, but PIERSIDE was not recorded in guider astrometry FITS header.  Was MaxIm connected to the telescope when the astrometry was recorded?')
        if guider_cal_astrometry_aligned:
            log.debug('Guider astrometry and calibration performed on same side of pier')
            self.guider_cal_pierside = guider_astrometry_pierside
        else:
            log.debug('Guider astrometry and calibration performed on opposite side of pier.  Flipping guider astrometry to line up with guider cal')
            if guider_astrometry_pierside == 'EAST':
                self.guider_cal_pierside = 'WEST'
            if guider_astrometry_pierside == 'WEST':
                self.guider_cal_pierside = 'EAST'
            # --> this tweaks our guider image header in memory
            self.guider_astrometry \
                = pier_flip_astrometry(self.guider_astrometry)
            # Don't forget to flip aang, our astrometry N angle, since
            # it is used below
            aang = angle_norm(aang+180, 180)
        log.debug('guider calibrated on pier ' + self.guider_cal_pierside)

        # Now see what direction E turns out to be relative to N.
        dp = self.scope_wcs((10., 0),
                            to_pix=True,
                            astrometry=self.guider_astrometry,
                            absolute=True,
                            delta=True)
        # Just in case there is a sizable component of up/down in E,
        # make sure we cast our synthetic E vector onto our MaxIm
        # coordinate system, where Y is in the opposite sense from the
        # WCS conventions
        dp[0] *= -1
        eang = np.degrees(np.arctan2(dp[0], dp[1]))
        # Convert angle to N up, +/-180 (already increasing CCW)
        eang = angle_norm(eang-90, 180)
        # Rotate into N up coordinate
        eang = angle_norm(eang - aang, 180)
        log.debug('aang, eang = ' + repr((aang, eang)))
        # At this point eang will be +90 if E goes left on the camera,
        # -90 if it goes W.  ASSUMING GUIDER +X CONNECTED TO E
        if np.sign(eang*self.CCDCamera.GuiderXSpeed) == -1:
            guider_cal_Xflip = False
        else:
            guider_cal_Xflip = True
        # ASSUMING GUIDER +Y CONNECTED TO N and keeping in mind +Y is
        # _down_ in MaxIm
        if np.sign(self.CCDCamera.GuiderYSpeed) == -1:
            guider_cal_Yflip = False
        else:
            guider_cal_Yflip = True

        # Now do the best we can with our table of what mode we are in
        if self.guider_cal_pierside == 'WEST':
            if guider_cal_Xflip or guider_cal_Yflip:
                log.debug('Assuming normal motor connections of E = -X, N = +Y, motor reversal(s) detected on pierWest guider cal.  Setting to pier flip on pierWest.  This is almost certainly MaxIm mode.')
                self.pier_flip_on_side = win32com.client.constants.pierWest
            else:
                log.debug('Assuming normal motor connections of E = -X, N = +Y, no motor reversals were detected on pierWest guider cal.  Setting to pier flip on pierEast.  This is almost certainly ACP mode.')
                self.pier_flip_on_side = win32com.client.constants.pierEast

        if self.guider_cal_pierside == 'EAST':
            if guider_cal_Xflip or guider_cal_Yflip:
                log.warning('Pier flip detected for pierEast guider cal.  This is not a valid MaxIm pier flip state and ACP does not allow calibration on pierEast.  Do you have your motor control leads hooked up in the normal way: E = -X, N = +Y?  For now I will assume the connections are normal and just set pier_flip_onside pierEast.  If the guider pushes the star out or precision guide moves the telecsope in the wrong way and it is not possible for you to change the motor control leads, contact the developer and ask for an abstraction layer to be added')
                self.pier_flip_on_side = win32com.client.constants.pierEast
            else:                
                log.debug('Assuming normal motor connections of E = -X, N = +Y, no motor reversal(s) detected on pierEast guider cal.  Setting to pier flip on pierWest.  This is almost certainly MaxIm mode.')
                self.pier_flip_on_side = win32com.client.constants.pierWest

    def set_guider_motor_reverse_and_DEC(self):
        # --> I am not sure the effect of this if the telescope is
        # --> connected and Auto Scope Dec is selected
        self.CCDCamera.GuiderDeclination = self.Telescope.Declination
        if self.alignment_mode != win32com.client.constants.algGermanPolar:
            log.debug('Not GEM, no motor reversal')
            return
        if self.pier_flip_on_side == win32com.client.constants.pierEast:
            log.debug("ACP guider calibration mode detected.  Don't let MaxIm manage pier flip...")
            self.CCDCamera.GuiderAutoPierFlip = False
        elif (self.Application.TelescopeConnected
              and self.CCDCamera.GuiderAutoPierFlip):
            # ACP does pier flips on east side
            log.debug('Let MaxIm manage pier flip state: guider calibrated in MaxIm mode, MaxIm is connected to the telescope and Auto Pier Flip is on.')
            # Set these to False, since otherwise they would confuse
            # us and MaxIm
            self.CCDCamera.GuiderReverseX = False
            self.CCDCamera.GuiderReverseY = False
            return
        else:
            log.debug("MaxIm is not managing pier flip...")
        if self.Telescope.SideOfPier == self.pier_flip_on_side:
            self.CCDCamera.GuiderReverseX = guider_motor_control_reverseX
            self.CCDCamera.GuiderReverseY = guider_motor_control_reverseY
            log.debug("... flip detected...")
        else:
            self.CCDCamera.GuiderReverseX = False
            self.CCDCamera.GuiderReverseY = False
            log.debug("... no flip detected...")
        log.debug("CCDCamera.GuiderReverseXY = " + repr((self.CCDCamera.GuiderReverseX, self.CCDCamera.GuiderReverseY)))
        
    def guider_motor_reverse_state(self):
        """Return np.array indicating motor reverse state"""
        # See guider_motor_reverse_setup for explanation
        revXY = np.asarray([1,1])
        if self.alignment_mode != win32com.client.constants.algGermanPolar:
            log.debug('Not GEM, no motor reversal')
            return revXY
        if self.Application.TelescopeConnected:
            log.debug("MaxIm is connected to the telescope...")
            if self.pier_flip_on_side == win32com.client.constants.pierEast:
                log.debug("... but guider calibrated in ACP mode, so inspecting GuiderReverseXY...")
            elif self.CCDCamera.GuiderAutoPierFlip:
                log.debug("...and managing pier flip...")
                if (self.Telescope.SideOfPier
                      == win32com.client.constants.pierEast):
                    log.debug("...but telescope is on pierEast, so no motor reversal")
                elif (self.Telescope.SideOfPier
                    == win32com.client.constants.pierWest):
                    log.debug("... and reversing motor motion because the telescope is on pierWest")
                    if guider_motor_control_reverseX:
                        revXY *= np.asarray([-1,1])
                    if guider_motor_control_reverseY:
                        revXY *= np.asarray([1,-1])
                else:
                    raise EnvironmentError('Inconsistent Telescope.SideofPier ' + repr(self.Telescope.SideOfPier))
            else:
                log.debug('... but not managing pier flip, so inspecting GuiderReverseXY...')
        else:
            log.debug('MaxIm is not connected to the telescope, so inspecting GuiderReverseXY...')
        if self.CCDCamera.GuiderReverseX:
            log.debug('... GuiderReverseX is set')
            revXY *= np.asarray([-1,1])
        if self.CCDCamera.GuiderReverseY:
            log.debug('... GuiderReverseY is set')
            revXY *= np.asarray([1,-1])
        log.debug('Motor reversal state = ' + repr(revXY))
        return revXY

    def check_guider_speeds(self):
        """Check guider X and Y speeds against telescope guide rates.
        This also sets the self.guide_rates property"""

        # Do this by creating synthetic guider calibrations.  CDELT*
        # are the same regardless of DEC, but DEC must be considered
        # when doing coordinate transformation from pixels to world,
        # so just set the guider astrometry DEC to 0, saving current
        # value so we can put it back later
        save_crval2 = self.guider_astrometry['CRVAL2']
        self.guider_astrometry['CRVAL2'] = 0
        # Create a notional time to move in pixel space.  It would be
        # great to use the actual guider cal times, which defaults to
        # 10s, but MaxIm does not make those available.  So just use
        # 10s
        dt = 10
        dx = self.CCDCamera.GuiderXSpeed * dt
        dy = self.CCDCamera.GuiderYSpeed * dt
        # Transpose, since we are in pix
        dra_ddec = self.scope_wcs((dy, dx),
                                  to_world=True,
                                  astrometry=self.guider_astrometry,
                                  absolute=True,
                                  delta=True)
        guider_cal_guide_rates = dra_ddec/dt
        # Use cal speeds by default but see if we can refine them with
        # scope rates
        self.guide_rates = guider_cal_guide_rates
        if self.telescope_connectable and self.Telescope.CanSetGuideRates:
            # Always assume telescope reported guide rates are
            # correct, but warn if guider rates are off by 10%,
            # keeping in mind that MaxIm seems to consider a pixel
            # size to be its diameter
            self.guide_rates \
                = np.asarray((self.Telescope.GuideRateRightAscension,
                              self.Telescope.GuideRateDeclination))
            dgrp = np.linalg.norm(self.guide_rates)
            dgcgrp = np.linalg.norm(guider_cal_guide_rates)
            dr = np.abs(dgrp - dgcgrp)
            if dr > 0.1 * dgrp:
                log.warning('Guider calibration rate is off by more than 10% of the scope reported rate: ' + repr((self.guide_rates, np.abs(guider_cal_guide_rates))) + '.  Norm of these: ' + repr((np.linalg.norm(self.guide_rates), np.linalg.norm(guider_cal_guide_rates))) + '.  Have you specified the correct guider astrometery image?  Have you changed the guide rates changed since calibrating the guider?  Assuming reported telescope guide rates are correct.')

        self.guider_astrometry['CRVAL2'] = save_crval2  

    # For now use self.Application.ShutDownObservatory()
    #def do_shutdown(self):
    #    self.Telescope.Park()
    #    return True
    #
    #def check_shutdown(self):
    #    # if weather: AstroAlert.Weather
    #    #    self.do_shutdown()



    #def set_GuiderReverse_and_DEC(self):
    #    # --> Eventually, I could include GuiderReverseY
    #    # Tell MaxIm about pier flip and scope DEC manually in case we
    #    # are in ACP-mode.
    #    if self.ACP_mode:            
    #        self.CCDCamera.GuiderReverseX \
    #            = (self.Telescope.SideOfPier
    #               == win32com.client.constants.pierEast)
    #        log.debug("GuiderReverseX set to " + repr(self.CCDCamera.GuiderReverseX))
    #        self.CCDCamera.GuiderDeclination = self.Telescope.Declination
    #        log.debug("Guider DEC set to " + repr(self.CCDCamera.GuiderDeclination))

    def guider_cycle(self, n=1):
        """Returns average and RMS guider error magnitude after n guider cycles

        Parameters
        ----------
        n : int like
            Number of guider cycles.  Default = 1

        norm : boolean
            Return norm of guider error.  Default False


        """
        if not self.CCDCamera.GuiderRunning:
            log.warning('Guider not running')
            return None
        this_norm = 0
        running_total = 0
        running_sq = 0
        for i in range(n):
            assert self.weather_server.Safe, ('Weather is not safe!')
            # --> Need a timeout
            while self.CCDCamera.GuiderNewMeasurement is False:
                time.sleep(self.loop_sleep_time)
            # As per MaxIm documentation, reading these clears
            # GuiderNewMeasurement
            this_norm = np.linalg.norm(
                (self.CCDCamera.GuiderYError,
                 self.CCDCamera.GuiderXError))
            running_total += this_norm
            running_sq += this_norm**2
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
            if self.obs_terminator.end_of_obs(self):
                log.error('End of observation reached, terminating guider_settle')
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
        """Moves the telescope by moving the guide box.  Guide box position is moved gradually relative to instantaneous guide box position, resulting in a delta move relative to any other guide box motion.  NOTE: use guider_stop to make sure guider position is properly set after guider is stopped (see that function's documentation).

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
        # In world coords, we know how far we want to move our guide
        # box.  Calculate the new guidebox position
# --> I think this is where the MaxIm people and I differ by a minus
# sign in MaxIm 6.20
        p_coords = self.scope_wcs(w_coords + dra_ddec,
                                  to_pix=True,
                                  astrometry=guider_astrometry)
        D.say('New code p_coords: ' + repr(p_coords[::-1]))
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
            if self.obs_terminator.end_of_obs(self):
                log.error('End of observation reached, terminating move_with_guuide_box')
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
        if self.guider_cal_pierside == self.Telescope.SideOfPier:
            log.debug('Telescope is on same side as astrometry')
            return 1
        log.debug('Telescope is on opposite side from astrometry')
        return -1

    def move_with_guider_slews(self,
                               dra_ddec,
                               guider_astrometry=None):
        """Moves the telescope using guider slews.

        Parameters
        ----------
        dra_ddec : tuple-like array delta move in RA and DEC in DEGREES
        guider_astrometry : FITS header.  Defaults to guider
            astrometry in object
        """
        log.debug('Preparing to move with guider slews by dRA, dDEC = ' + repr(dra_ddec))
        if self.CCDCamera.GuiderRunning:
            log.warning('Guider was running, turning off')
            self.CCDCamera.GuiderStop
        # Use our guide rates to change to time to press E/W, N/S.
        # Note that we deal with sign below
        dt = dra_ddec/self.guide_rates
        # Do a sanity check to make sure we are not moving too much
        max_t = (self.guider_max_move_multiplier *
                 np.asarray((self.CCDCamera.GuiderMaxMoveX, 
                             self.CCDCamera.GuiderMaxMoveY)))
        if np.any(np.abs(dt) > max_t):
            log.warning('requested move of ' + str(dra_ddec) + ' arcsec translates into move times of ' + str(np.abs(dt)) + ' seconds.  Limiting move in one or more axes to max t of ' + str(max_t))
            dt = np.minimum(max_t, abs(dt)) * np.sign(dt)
        # Or too little
        bad_idx = np.where(np.abs(dt) <
                           np.asarray((self.CCDCamera.guiderMinMoveX,
                                      self.CCDCamera.guiderMinMoveY)))
        dt[bad_idx] = 0
        log.debug('Seconds to move guider in RA and DEC: ' + str(dt))
        # Now we need to translate this into how long to command MaxIm
        # to operate the + or - X and Y motors.  self.guide_rates gets
        # us the right absolute time to push the buttons, but we have
        # to work with the sign of the X and Y speed calibrations and
        # any pier flips to get the directions right
        XYsign = np.sign(np.asarray((self.CCDCamera.GuiderXSpeed,
                                    self.CCDCamera.GuiderYSpeed)))
        # First assume we are on the same side of the pier as the
        # guider calibration.  Recall from guider_motor_reverse_setup
        # documentation that +RA is typically connected to the +X
        # motor, but +RA is east, east is left on the sky and left is
        # -X in a nominally oriented MaxIm camera.  Hence the nominal
        # GuiderXSpeed is negative for a properly connected +RA to +X
        # motor.  To have a +RA command the +X motor, we therefore
        # need to negate the X part of XYsign.  Similarly +DEC is
        # connected to the +Y motor, but Y increases down in MaxIm, so
        # GuiderYSpeed often comes out negative.  Thus:
        XYsign = -XYsign
        # Now apply our pier flip, since after all that is said and
        # done, MaxIm (or we) may be doing some motor reversals.
        self.set_guider_motor_reverse_and_DEC()
        dt *= XYsign * self.guider_motor_reverse_state()
        log.debug('Seconds to command MaxIm to move X and Y guider motors, where +X connects to +RA (nominally -X on the CCD) and +Y connects to +DEC (nominally -Y on the CCD): ' + str(dt))        
        if dt[0] > 0:
            RA_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdPlusX, dt[0])
        elif dt[0] < 0:
            RA_success = self.CCDCamera.GuiderMove(win32com.client.constants.gdMinusX, -dt[0])
        else:
            # No need to move
            RA_success = True
        if not RA_success:
            raise EnvironmentError('RA guide slew command failed')
        # MaxIm seems to be able to press RA and DEC buttons
        # simultaneously, but we can't!
        while self.CCDCamera.GuiderMoving:
            assert self.weather_server.Safe, ('Weather is not safe!')
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
            assert self.weather_server.Safe, ('Weather is not safe!')
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
        absolute : ignore scope position and use position from astrometry
        delta : assume coords_in are delta from center of CCD
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
            # --> this is going to change
            if self.astrometry_pier_flip_state() == -1:
                header = pier_flip_astrometry(header)

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
        # --> don't be fancy, having trouble getting initial guide star
        # --> not sure if this is the trouble or if the initial
        # time.sleep(0.3) before the while solved the problem
        #if not auto_star_selected and star_position is None:
        if True:
            # Take an exposure to get MaxIm to calculate the guide
            # star postion --> consider temporary code to select lower
            # filter number if guide star not found
            self.CCDCamera.GuiderExpose(self.guider_exptime)
            time.sleep(self.guider_exptime)
            # --> Consider checking for timout here
            while self.CCDCamera.GuiderRunning:
                time.sleep(0.3)
        # Since ACP does not want MaxIm connected to the scope, we
        # have to manage all that stuff ourselves
        self.set_guider_motor_reverse_and_DEC()
        if not self.CCDCamera.GuiderTrack(self.guider_exptime):
            raise EnvironmentError('Attempt to start guiding failed.  Guider configured correctly?')
        # MaxIm rounds pixel center value to the nearest pixel,
        # which can lead to some initial motion in the guider
        self.guider_settle()
        self.guider_commanded_running = True

    def guider_stop(self):
        """Stops the guider, sets the guider position to the last position used by move_with_guidebox and sets GuiderReverse[XY] values to previous.  Returns value of CCDCamera.GuiderStop unless guider wasn't running, in which case False is returned
        The guider position update is required as per discussion in https://forum.diffractionlimited.com/threads/ccdcamera-guiderxstarposition-guiderystarposition-no-longer-updated-after-ccdcamera-guidermovestar.6048/page-2 """
        retval = False
        self.guider_commanded_running = False
        if self.CCDCamera.GuiderRunning:
            # As per documentation above, turning off the guider makes
            # the guide point revert to the original position before
            # any GuiderMoveStar was done
            x = self.CCDCamera.GuiderXStarPosition
            y = self.CCDCamera.GuiderYStarPosition
            autoselect = self.CCDCamera.GuiderAutoSelectStar
            retval = self.CCDCamera.GuiderStop
            if autoselect:
                # MaxIm won't GuiderSetStarPosition when AutoSelectStar is on
                self.CCDCamera.GuiderAutoSelectStar = False
            # Set our guide star position to the actual position we
            # have just moved the star to, just in case we want to
            # turn the guider back on again
            self.CCDCamera.GuiderSetStarPosition(x, y)
            if autoselect:
                # If, instead, our next step is to take an exposure
                # with the guider, the AutoSelectStar algorithm will
                # be used instead of the old star position
                self.CCDCamera.GuiderAutoSelectStar = True
        # Reset reverse to previous values in case we are handing off to ACP(?)
        self.CCDCamera.GuiderReverseX = self.previous_GuiderReverseX
        self.CCDCamera.GuiderReverseY = self.previous_GuiderReverseY
        return retval
    
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
        if binning is None:
            binning = 1
        # Set the filter separately, since some filter wheels need
        # time to rotate
        if self.CCDCamera.Filter != filt:
            self.CCDCamera.Filter = filt
            time.sleep(self.main_filt_change_time)
        self.CCDCamera.BinX = binning
        if not self.CCDCamera.XYBinning:
            self.CCDCamera.BinY = binning
        # --> Add support for camera, and non-light
        # Take a light (1) exposure
        self.CCDCamera.Expose(exptime, 1, filt)
        # This is potentially a place for a coroutine and/or events
        assert self.weather_server.Safe, ('Weather is not safe!')
        timeleft = exptime
        while timeleft > 60:
            time.sleep(60)
            assert self.weather_server.Safe, ('Weather is not safe!')
            timeleft -= 60
        if timeleft > 1:
            time.sleep(timeleft)
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
MC : MaxImControl
    MaxImControl object set up with defaults you would like to use
    for guiding, main camera exposure, etc.  Default: MaxImControl
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
            MC=None,
            obs_terminator=None,            
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
        if MC is None:
            self.MC = MaxImControl()
        self.obs_terminator = obs_terminator or ObsTerminator()
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
        # Turn off everything
        self.GuideBoxMoving = False
        self.MC.__exit__(exception_type, exception_value, traceback)

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
            Tend = Tend or ''
            self._GuideBoxMoverSubprocess \
                = subprocess.Popen(['python',
                                    #'ioio.py',
                                    __file__,
                                    'GuideBoxMover',
                                    self.guide_box_command_file,
                                    Tend])
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
                = self.MC.scope_wcs(dpix,
                                    to_world=True,
                                    delta=True,
                                    astrometry=self.MC.main_astrometry)
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
        """-->Eventually this is going to read a map/model file and calculate the differential flexure to feed to GuideBoxCommander.  There may need to be an additional method for concatenation of this flex and the measured flexure.  THIS IS ONLY FOR JUPITER DEC RIGHT NOW  --> WARNING THIS HAS GUIDER PLATE SCLAE HARD-CODED IN!"""
        # Lodestar X2
        #plate_ratio = 4.42/(1.56/2)
        # 
        plate_ratio = 3.01/(1.56/2)
        # 2018
        #if (-40 < self.MC.Telescope.Declination
        #    and self.MC.Telescope.Declination < +10
        #    and self.MC.Telescope.Altitude < 30):
        #    # Change from guider pixels per 10s to main camera pixels per s
        #    dec_pix_rate = -0.020/10 * plate_ratio
        #    # Note Pythonic transpose
        #    return self.GuideBoxCommander(np.asarray((dec_pix_rate, 0)))
        # 2019 post 9-position filter wheel.  Jupiter only got up to
        # 35 and this was variable near the meridian but never much larger.
        # Thu May 09 10:28:09 2019 EDT  jpmorgen@snipe
        # This may be messing up flips for ACP
        #if (-40 < self.MC.Telescope.Declination
        #    and self.MC.Telescope.Declination < +10
        #    and self.MC.Telescope.Altitude < 35):
        #    # Change from guider pixels per 10s to main camera pixels per s
        #    dec_pix_rate = -0.005/10 * plate_ratio
        #    # Note Pythonic transpose
        #    return self.GuideBoxCommander(np.asarray((dec_pix_rate, 0)))
        # 2020 early
        # if (-40 < self.MC.Telescope.Declination
        #     and self.MC.Telescope.Declination < +10
        #     and self.MC.Telescope.Altitude < 30):
        #     # Change from guider pixels per 10s to main camera pixels per s
        #     ra_pix_rate = -0.014/10 * plate_ratio
        #     dec_pix_rate = -0.020/10 * plate_ratio
        #     # Note Pythonic transpose
        #     return self.GuideBoxCommander(np.asarray((dec_pix_rate, ra_pix_rate)))
        # 2021 after guider change
        if (-40 < self.MC.Telescope.Declination
            and self.MC.Telescope.Declination < +40
            and self.MC.Telescope.Altitude < 40):
            # Change from guider pixels per 10s to main camera pixels per s
            ra_pix_rate = -0.010/10 * plate_ratio
            # Fri May 14 10:24:04 2021 EDT  jpmorgen@snipe
            # This is the formal calculation, but intuition tells me
            # it is too big of a swing to make at once
            #dec_pix_rate = +0.051/10 * plate_ratio
            #dec_pix_rate = +0.030/10 * plate_ratio
            # Indeed, this should probably get pretty close
            dec_pix_rate = +0.037/10 * plate_ratio
            # Note Pythonic transpose
            return self.GuideBoxCommander(np.asarray((dec_pix_rate, ra_pix_rate)))


        return self.GuideBoxCommander(np.asarray((0, 0)))

    # --> I'll probably want a bunch of parameters for the exposure
    def center(self,
               HDUList_im_fname_ObsData_or_obj_center=None,
               desired_center=None,
               current_astrometry=None,
               scaling_astrometry=None,
               ignore_ObsData_astrometry=False,
               recursive_count=0,
               dead_zone=None, # Pixels
               dead_zone_move=None, # Pixels
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
        if self.obs_terminator.end_of_obs(self.MC):
            log.error('End of observation reached, terminating center')
            return False

        # save some typing
        input = HDUList_im_fname_ObsData_or_obj_center
        if input is None:
            # Take an image with the default exposure time and filter
            input = self.MC.take_im()
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
                self.MC.connect()
                desired_center \
                    = np.asarray((self.MC.CCDCamera.StartY
                                  + self.MC.CCDCamera.NumY, 
                                  self.MC.CCDCamera.StartX
                                  + self.MC.CCDCamera.NumX)) / 2.
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
        w_coords = self.MC.scope_wcs((obj_center, desired_center),
                                     to_world=True,
                                     astrometry=astrometry_from,
                                     absolute=absolute)
        log.debug('world coordinates of obj_center and desired_center: ' + repr(w_coords))

        dw_coords = w_coords[1,:] - w_coords[0,:]
        if self.MC.CCDCamera.GuiderRunning:
            log.debug('CENTERING TARGET WITH GUIDEBOX MOVES')
            if not self.MC.move_with_guide_box(dw_coords):
                if recursive_count > 1:
                    log.error('center: Failed to center target using guidebox moves after two tries')
                    return False                        
                log.debug('TURNING GUIDER OFF AND CENTERING WITH GUIDER SLEWS')
                self.MC.guider_stop()
                # --> Consider using center here instead of move_with_guider_slews
                self.center_loop(dead_zone=dead_zone,
                                 dead_zone_move=dead_zone_move)
                #self.MC.move_with_guider_slews(dw_coords)
                # --> Need to add logic to capture guider stuff,
                # though filter should be the same.  It is just
                # the exposure time that I might want to have
                # carried over, though I will eventually figure
                # that out myself.
                log.debug('RESTARTING GUIDER AND CENTERING WITH GUIDEBOX MOVES')
                self.MC.guider_start()
                recursive_count += 1
                self.center(recursive_count=recursive_count,
                            dead_zone=dead_zone,
                            dead_zone_move=dead_zone_move)
        else:
            log.debug('CENTERING TARGET WITH GUIDER SLEWS')
            self.MC.move_with_guider_slews(dw_coords)
            if self.MC.guider_commanded_running:
                log.debug('Guider was turned off unexpectedly.  Turning it back on and recentering with guidebox moves')
                self.MC.guider_start()
                self.center(recursive_count=recursive_count,
                            dead_zone=dead_zone,
                            dead_zone_move=dead_zone_move)
        return True

    def center_loop(self,
                    exptime=None,
                    filt=None,
                    tolerance=None,
                    max_tries=3,
                    start_PrecisionGuide=False,
                    dead_zone=None, # Pixels
                    dead_zone_move=None, # Pixels
                    was_guiding=None,
                    **ObsClassArgs):
        """Loop max_tries times taking exposures and moving the telescope with guider slews or, if the guider is on, guide box moves to center the object
        """
        if tolerance is None:
            tolerance = self.center_tolerance
        tries = 0
        fails = 0
        while True:
            if self.obs_terminator.end_of_obs(self.MC):
                log.error('End of observation reached, terminating center_loop')
                return False
            log.debug('CENTER_LOOP TAKING EXPOSURE')
            HDUList = self.MC.take_im(exptime, filt)
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
            d_center = np.linalg.norm(O.obj_center - O.desired_center)
            if (d_center < tolerance):
                log.debug('Target centered to ' + str(tolerance) +
                          ' pixels')
                if was_guiding:
                    # Turn our guider back on after a dead_zone
                    # incident in the case that it happened to be
                    # running while we wondered out (hopefully rare --
                    # usually a wind or cloud event)
                    log.debug('TURNING GUIDER BACK ON AND...')
                    self.MC.guider_start()
                    log.debug('CENTERING WITH GUIDEBOX MOVES') 
                    return self.center_loop(exptime=exptime,
                                            filt=filt,
                                            tolerance=tolerance,
                                            max_tries=max_tries)
                if start_PrecisionGuide:
                    # We have moved the telescope, so our Obs
                    self.reinitialize()
                    self.update_flex_pix_rate(0)
                return True
            if (dead_zone is not None
                and dead_zone[0] < d_center
                and d_center < dead_zone[1]):
                # Kick ourselves out of the dead zone and get a running
                # start from outside
                if self.MC.CCDCamera.GuiderRunning:
                    was_guiding = True
                    log.debug('TURNING GUIDER OFF AND...')
                    self.MC.guider_stop()
                log.debug('MOVING OUT OF DEAD ZONE')
                if dead_zone_move is None:
                    dead_zone_move = (dead_zone[1], dead_zone[1])
                # These need to be float
                dead_zone = np.asarray(dead_zone)
                dead_zone = dead_zone.astype(float)
                dead_zone_move = np.asarray(dead_zone_move)
                dead_zone_move = dead_zone_move.astype(float)
                dra_ddec = self.MC.scope_wcs(dead_zone_move,
                                             to_world=True,
                                             delta=True)
                self.MC.move_with_guider_slews(dra_ddec)
                # Try only once to kick ourselves out, otherwise
                # unpleasant looping occurs!
                return self.center_loop(exptime=exptime,
                                        filt=filt,
                                        tolerance=tolerance,
                                        max_tries=max_tries,
                                        was_guiding=was_guiding)
                
            if tries >= max_tries:
                log.error('Failed to center target to ' +
                          str(tolerance) + ' pixels after '
                          + str(tries) + ' tries')
                return False
            # Here is where we actually call the center algorithm
            if not self.center(O,
                               dead_zone=dead_zone,
                               dead_zone_move=dead_zone_move):
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
        ###    dra_ddec = self.MC.scope_wcs(dpix,
        ###                                 to_world=True,
        ###                                 delta=True,
        ###                                 astrometry=self.MC.main_astrometry)
        ###    log.warning(
        ###        'calc_flex_pix_rate: Too far from center, moving dra_ddec: '
        ###        + str(dra_ddec))
        ###    self.MC.move_with_guide_box(dra_ddec)
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
        ###    dra_ddec = self.MC.scope_wcs(dpix,
        ###                                 to_world=True,
        ###                                 delta=True,
        ###                                 astrometry=self.MC.main_astrometry)
        ###    log.debug('dra_ddec: ' + str(dra_ddec))
        ###    # --> I might decide that a move is too disruptive to
        ###    # the rate calculations and just start them over
        ###    self.MC.move_with_guide_box(dra_ddec)
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
        self.MC.CCDCamera.EventMask = 2
        log.debug('MaxIm set to notify when main camera exposure complete')
        #for i in range(3):
        #    event = self.MC.CCDCamera.Notify
        #    log.debug('Exposure ended: ' + str(event))

    # Mostly this just passes parameters through to
    # MaxImControl.acquire_im to take the image.  We need to have
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
        assert self.MC.CCDCamera.GuiderRunning, 'Guider must be running.  You can start it with PrecisionGuide.MC.guider_start()'

        # Here might be where we make the choice to use ACP's
        # TakePicture or record it ourselves based on whether or
        # not ACP's objects are present
        if ACP_obj:
            # Eventually we would read the file from the disk
            # Consider using ACP's TakePicture
            HDUList = fits.open(fname)
            O = self.create_ObsData(HDUList, **ObsClassArgs)
        else:
            HDUList = self.MC.acquire_im(fname=fname,
                                         exptime=exptime,
                                         filt=filt,
                                         binning=binning,
                                         subarray=subarray)
            #HDUList = self.MC.take_im(exptime, filt, binning)
            ## Write image to disk right away in case something goes wrong
            #if not self.MC.CCDCamera.SaveImage(fname):
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
    P.center(desired_center=desired_center)
    log.debug('STARTING GUIDER') 
    P.MC.guider_start()
    log.debug('CENTERING WITH GUIDEBOX MOVES') 
    P.center(desired_center=desired_center)

def cmd_guide(args):
    MC = MaxImControl()
    if args.stop:
        MC.guider_stop()
    else:
        MC.guider_start(exptime=args.exptime, filter=args.filter)

# --> Eventually, I would like this to accept input from other
# --> sources, like a flexure model and ephemeris rates

# --> This doesn't work when the file changes faster than it would do
# --> a guidebox command
def GuideBoxMover(args):
    log.debug('Starting GuideBoxMover')
    Tend = args.Tend or None
    obs_terminator = ObsTerminator(Tend=Tend,
                                   horizon_limit=horizon_limit,
                                   sundown_limit=sundown_limit)
    
    with MaxImControl() as MC:
        last_modtime = 0
        while True:
            if obs_terminator(MC):
                # Returning false seemed to confuse process that
                # popened us
                # --> This might be the cause of the
                # confusion at the end of observations that requires
                # long wait times for things to exit!
                log.error('GuideBoxMover: End of observation time reached.  Doing nothing')
                time.sleep(3)
                continue
    
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
            # this a constant or something poassibly in MC
            now = time.time()  
            dt = now - lastt
            if np.linalg.norm(dra_ddec_rate) * dt >= 0.5/3600:
                log.debug('GuideBoxMover dt(s) = ' + str(dt))
                log.debug('GuideBoxMover is moving the guidebox by ' +
                          str(dra_ddec_rate * dt*3600) + ' arcsec')
                if not MC.move_with_guide_box(dra_ddec_rate * dt ):
                    log.error('GuideBoxMover MC.move_with_guide_box failed')
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
    if not P.MC.CCDCamera.GuiderRunning:
        # User could have had guider already on.  If not, center with
        # guider slews and start the guider
        log.debug('CENTERING WITH GUIDER SLEWS') 
        P.center_loop(max_tries=5)
        log.debug('STARTING GUIDER') 
        P.MC.guider_start()
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
        P.MC.acquire_im(fname,
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
    GuideBox_parser.add_argument(
        'Tend', help='Time in UNIX seconds past which observations are invalid',
        default='')
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
