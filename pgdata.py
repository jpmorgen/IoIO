"""Define lightweight data containers for precisionguide system"""

import numpy as np
from astropy.io import fits

class PGCenter():
    """Base class for containing object center and desired center


    Parameters
    ----------
    obj_center : 
    """
    def __init__(self,
                 obj_center=None,
                 desired_center=None):
        self.obj_center = obj_center
        self.desired_center = desired_center

    @property
    def obj_center(self):
        return self._obj_center
        
    @obj_center.setter
    def obj_center(self, value):
        if value is None:
            self._obj_center = None
        else:
            self._obj_center = np.asarray(value)

    @property
    def desired_center(self):
        if self._desired_center is not None:
            return self._desired_center
        
        
    @desired_center.setter
    def desired_center(self, value):
        if value is None:
            self._desired_center = None
        else:
            self._desired_center = np.asarray(value)

class PGData():
    """Base class for image data in the `precisionguide` system

    This class contains an individual image data and calculates and/or
    stores two quantities: :prop:`desired_center` and
    :prop:`obj_center`.  The center quantities are intended to be
    returned in a :class:`PGCenter` object for subsequent lightweight
    storage and use in the precisionguide system.  Because
    precisionguide controls the absolute position of an object (or FOV
    center) on the CCD, :prop:`desired_center` and :prop:`obj_center`
    always read in *unbinned* pixel values referenced to the origin of
    the CCD itself.  Thus, the image input to :class:`PGData` must
    include both the image FITS header and image array(s)

    """

    def __init__(self,
                 input_im=None,
                 desired_center=None,
                 center_offset=(0, 0)):
        if input_im is None:
            raise ValueError('input_im must be specified')
        self._data = None
        self._data_unbinned = None
        self._meta = None
        self._obj_center = None
        self._desired_center = None
        self._center_offset = None
        self._read(input_im)
        self.center_offset = center_offset
        self.desired_center = desired_center

    def _read(self, input_im):
        # Reading invalidates all of our calculations.  Make this
        # private so it only gets done once
        if isinstance(input_im, str):
            with fits.open(input_im) as HDUList:
                self._data = HDUList[0].data
                self._meta = HDUList[0].header
        elif isinstance(input_im, fits.HDUList):
            self._data = input_im[0].data
            self._meta = input_im[0].header
        else:
            # Assume input_im is CCDData-like
            try:
                self._data = input_im.data
                self._meta = input_im.meta
            except:
                raise ValueError('Not a valid input, input_im')
        return self

    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self, value):
        if value.shape != self._data.shape:
            # Unfortunately, we can't check for subframe origin shift
            raise ValueError('New data array must be same shape as old array')
        self._data = value
        self._data_unbinned = None
        
    @property
    def meta(self):
        return self._meta
        
    @meta.setter
    def meta(self, value):
        self._meta = value
        self._obj_center = None
        
    @property
    def header(self):
        return self._meta
        
    @header.setter
    def header(self, value):
        self.meta = value

    @property
    def binning(self):
        return np.asarray((1,1))
        # Note, this needs to be overwritten with the actual FITS
        # binning keywords used (which are unufortunately not in any
        # FITS standard)
        #return = np.asarray((self.meta['YBINNING'],
        #                     self.meta['XBINNING']))
        
    @property
    def subframe_origin(self):
        # Overwrite like binning.  Note, ZERO reference.  Make sure to
        # adjust values read from subframe origin keywords if
        # necessary
        return np.asarray((0,0))

    def unbinned(self, coords):
        """Returns coords referenced to full CCD given internally stored binning/subim info"""
        coords = np.asarray(coords)
        return np.asarray(self.binning * coords + self.subframe_origin)

    def binned(self, coords):
        """Assuming coords are referenced to full CCD, return location in binned coordinates relative to the subframe origin"""
        coords = np.asarray(coords)
        return np.asarray((coords - self.subframe_origin) / self.binning)
        
    def im_unbinned(self, a):
        """Returns an unbinned version of a.  a must be same shape as data
        """
        assert a.shape == self.data.shape
        # Don't bother if we are already unbinned
        if np.sum(self.binning) == 2:
            return a
        newshape = self.binning * a.shape
        # From http://scipy-cookbook.readthedocs.io/items/Rebinning.html
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new)
                   for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        unbinned = a[tuple(indices)]
        # Check to see if we need to make a larger array into which to
        # plop unbinned array
        if np.sum(self.subframe_origin) > 0:
            # Note subframe origin reads in binned pixels
            origin = self.unbinned(self.subframe_origin)
            full_unbinned = np.zeros(origin + np.asarray(unbinned.shape))
            full_unbinned[origin[0]:, origin[1]:] = unbinned
            unbinned = full_unbinned
        return unbinned

    @property
    def data_unbinned(self):
        """Returns an unbinned version of data
        """
        if self._data_unbinned is not None:
            return self._data_unbinned
        self._data_unbinned = self.im_unbinned(self.data)
        return self._data_unbinned

    @property
    def obj_center(self):
        if self._obj_center is not None:
            return self._obj_center
        # This obviously would be overwritten
        self._obj_center = self.desired_center + self.center_offset
        return self._obj_center

    @property
    def center_offset(self):
        return self._center_offset

    @center_offset.setter
    def center_offset(self, value):
        self._obj_center = None
        if value is None:
            self.center_offset = (0,0)
        else:
            self._center_offset = np.asarray(value)

    @property
    def desired_center(self):
        if self._desired_center is not None:
            return self._desired_center

        # Here is where the complicated desired_center calculation is
        # done:
        desired_center = np.asarray(self.data.shape)/2

        # After the calculation is done, use the setter to set the
        # keywords in the FITS header
        self.desired_center = desired_center
        return self._desired_center

    @desired_center.setter
    def desired_center(self, value=None):
        if value is None:
            value = self.desired_center
        self._desired_center = value
        # Note pythonic y, x coordinate ordering
        self.header['DES_CR0'] = (self._desired_center[1], 'Desired center X')
        self.header['DES_CR1'] = (self._desired_center[0], 'Desired center Y')
        return self._desired_center
        
    
#pgc = PGCenter()
#pgc = PGCenter((1,2), (3,4))
#pgd = PGData()
fname = '/data/io/IoIO/raw/2020-07-15/HD87696-0016_Na_off.fit'
pgd = PGData(fname)
