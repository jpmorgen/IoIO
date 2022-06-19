"""Base Photometry object that provides interface with photutils"""

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.coordinates import Angle, SkyCoord

from photutils import (make_source_mask, detect_sources,
                       deblend_sources)
from photutils import SourceCatalog
from photutils.background import Background2D
from photutils.utils import calc_total_error

from astroquery.simbad import Simbad
from astroquery.sdss import SDSS

from reproject.mosaicking import find_optimal_celestial_wcs

from bigmultipipe import assure_list

from precisionguide import pgproperty

EXPTIME_KEY = 'EXPTIME'
EXPTIME_UNIT = u.s
SOURCE_TABLE_COLS = ['label',
                     'xcentroid',
                     'ycentroid',
                     'elongation',
                     'eccentricity',
                     'equivalent_radius',
                     'min_value',
                     'max_value',
                     'segment_flux',
                     'segment_fluxerr']
SOLVE_TIMEOUT = 60 # s
VOTABLE_FIELDS = ['flux(U)', 'flux(B)', 'flux(V)', 'flux(R)',
                  'flux(I)']
PHOTOOBJ_FIELDS = ['ra', 'dec', 'probPSF', 'aperFlux7_u',
                   'aperFlux7_g', 'aperFlux7_r', 'aperFlux7_i',
                   'aperFlux7_z']

# I am not using this just yet, I will when I implement Simbad UBVRI
# and Sloan ugriz search results
JOIN_TOLERANCE = 5
JOIN_TOLERANCE_UNIT = u.arcsec

class control_property(pgproperty):
    """Decorator for Photometry.  

    Handles simple set and get cases and runs :meth:`init_calc()` when
    a value is changed

    """
    def __set__(self, obj, val):
        obj_dict = obj.__dict__
        if self.fset:
            # Just in case there is an extra thing to do.  In general,
            # these simple properties don't need custom setters
            val = self.fset(obj, val)
        old_val = obj_dict.get(self._key)
        if old_val is val or old_val == val:
            # Just in case == doesn't check "is" first  If nothing
            # changes, nothing changes
            return
        if old_val is not None:
            # This is our initial set or changing from nothing to something
            #log.debug(f'resetting {self._key} requires re-initialization of calculated quantities')
            obj.init_calc()
        obj_dict[self._key] = val

def create_rmat(angle):
    """Returns MATHEMATICAL 2D rotation matrix of angle.  This needs to be
    transposed if being used with FITS

    """ 
    if not isinstance(angle, u.Quantity):
        raise ValueError('angle must be a Quantity')
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s),
                       (s, c)))
    
def rot_wcs(wcs, angle, return_cd_or_pc=False):
    """Rotate WCS by angle (CCW)"""
    rmat = create_rmat(angle)
    # Keeping in mind we are in FITS space, which is transposed
    rmat = np.transpose(rmat)

    # OOPS!  This was a subtle bug.  The nice thing about standards is
    # there are so many to choose from!  the other astropy stuff &
    # numpy use copy() to do deepcopy()
    r_wcs = wcs.deepcopy()
    if r_wcs.wcs.has_cd():
        cd_or_pc = np.matmul(r_wcs.wcs.cd, rmat)
        r_wcs.wcs.cd = cd_or_pc
    elif r_wcs.wcs.has_pc():
        cd_or_pc = np.matmul(r_wcs.wcs.pc, rmat)
        r_wcs.wcs.pc = cd_or_pc
    else:
        raise ValueError('ccd.wcs.wcs does not have expected cd or pc '
                         'transformation matrix')
    if return_cd_or_pc:
        return (r_wcs, cd_or_pc)
    return r_wcs

def rot_angle_by_wcs(angle, ccd):
    """Rotates angle referenced to N, CCW positive by wcs transformation
    such that returned angle reads in pixel space

    NOTE: There may be a more elegant and accurate way to do this but
    I am not finding it.  This is good enough for plotting purposes

    """
    # Rotate the wcs by our angle so now the WCS N is pointing toward
    # our desired direction
    wcs = rot_wcs(ccd.wcs, angle)

    # Create a northward vector that is not too long with which to
    # measure our angle.

    # There are some circumstances I have found that wcs doesn't have
    # _naxis defined, so guess.  Note naxis is full field.  I am
    # trying to get 1/2 way to the edge
    naxis = ccd.shape[::-1]
    cdelt = wcs.proj_plane_pixel_scales()
    d_north = cdelt[1]*naxis[1]/4
    w_rot = wcs.wcs.crval * u.deg
    w_rot[1] += d_north
    coord = SkyCoord(*list(w_rot))
    p_rot = wcs.world_to_pixel(coord)
    v_rot = p_rot - (wcs.wcs.crpix - 1)
    v_up = np.asarray((0,1))
    # This is the angle to N, but not N CCW oriented
    angle = np.arccos(np.dot(v_up, v_rot)
                      / (np.linalg.norm(v_up)
                         * np.linalg.norm(v_rot)))
    angle *= u.rad
    angle = angle.to(u.deg)
    if v_rot[0] > 0:
        angle = 360*u.deg - angle
    return angle

def wcs_project(ccd, target_wcs, target_shape=None, order='bilinear'):
    """
    Given a CCDData image with WCS, project it onto a target WCS and
    return the reprojected data as a new CCDData image.

    Any flags, weight, XXor uncertaintyXX are ignored in doing the
    reprojection. (tweak from ccdproc.wcs_project)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        Data to be projected.

    target_wcs : `~astropy.wcs.WCS` object
        WCS onto which all images should be projected.

    target_shape : two element list-like or None, optional
        Shape of the output image. If omitted, defaults to the shape of the
        input image.
        Default is ``None``.

    order : str, optional
        Interpolation order for re-projection. Must be one of:

        + 'nearest-neighbor'
        + 'bilinear'
        + 'biquadratic'
        + 'bicubic'

        Default is ``'bilinear'``.

    {log}

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        A transformed CCDData object.
    """
    from astropy.nddata.ccddata import _generate_wcs_and_update_header
    from astropy.wcs.utils import proj_plane_pixel_area
    from reproject import reproject_interp

    if not (ccd.wcs.is_celestial and target_wcs.is_celestial):
        raise ValueError('one or both WCS is not celestial.')

    if target_shape is None:
        target_shape = ccd.shape

    projected_image_raw, _ = reproject_interp((ccd.data, ccd.wcs),
                                              target_wcs,
                                              shape_out=target_shape,
                                              order=order)

    reprojected_mask = None
    if ccd.mask is not None:
        reprojected_mask, _ = reproject_interp((ccd.mask, ccd.wcs),
                                               target_wcs,
                                               shape_out=target_shape,
                                               order=order)
        # Make the mask 1 if the reprojected mask pixel value is non-zero.
        # A small threshold is included to allow for some rounding in
        # reproject_interp.
        reprojected_mask = reprojected_mask > 1e-8

    reprojected_uncert = None
    if ccd.uncertainty is not None:
        reprojected_uncert, _ = reproject_interp(
            (ccd.uncertainty.array, ccd.wcs),
            target_wcs,
            shape_out=target_shape,
            order=order)
        
    # The reprojection will contain nan for any pixels for which the source
    # was outside the original image. Those should be masked also.
    output_mask = np.isnan(projected_image_raw)

    if reprojected_mask is not None:
        output_mask = output_mask | reprojected_mask

    # Need to scale counts by ratio of pixel areas
    area_ratio = (proj_plane_pixel_area(target_wcs) /
                  proj_plane_pixel_area(ccd.wcs))

    # If nothing ended up masked, don't create a mask.
    if not output_mask.any():
        output_mask = None

    # If there are any wcs keywords in the header, remove them
    hdr, _ = _generate_wcs_and_update_header(ccd.header)

    nccd = ccd.copy()
    nccd.data = area_ratio * projected_image_raw
    nccd.mask = output_mask
    if nccd.uncertainty:
        if nccd.uncertainty.uncertainty_type == 'std':
            nccd.uncertainty.array = area_ratio * reprojected_uncert
        if nccd.uncertainty.uncertainty_type == 'var':
            nccd.uncertainty.array = area_ratio**2 * reprojected_uncert
    nccd.meta = hdr
    nccd.wcs = target_wcs
    #nccd = CCDData(area_ratio * projected_image_raw, wcs=target_wcs,
    #               mask=output_mask,
    #               header=hdr, unit=ccd.unit)

    return nccd

def rot_to(ccd_in,
           rot_to_angle=None,
           rot_angle_from_key=None,
           rot_angle_key_unit=u.degree,
           **kwargs):
    """Reproject image such that "up" is celestial coords plus an
    additional rotation measured CCW from N.  Another way to think of
    it: situates to celestial N and rotates CW by the given angle(s)

    Parameters
    ----------
    ccd_in : astropy.nddata.CCDData
        Input `~astropy.nddata.CCDData`

    rot_to_angle : astropy.units.Quantity or None, optional
        Additional rotation angle after CCD image has been oriented N
        up, E west, CCW.  If None, no additional rotation is done
        Default is `None`
        
    rot_angle_from_key : str or list of str None
        FITS header key(s) from which rotation angle is constructed.
        Values in keys are added to each other, including the initial
        value of rot_to_angle
        Default is `None`

    rot_angle_key_unit : astropy.units.Unit
        Unit applied to rot_angle_from_key
        Default is `astropy.units.deg`

    **kwargs : required as bigmultipipe post-processing routine

    """
    ccd = ccd_in.copy()
    rot_to_angle = rot_to_angle or 0*u.degree
    docstring = ''
    if rot_angle_from_key is not None:
        rot_angle_from_key = assure_list(rot_angle_from_key)
        for k in rot_angle_from_key:
            rot_to_angle += ccd.meta[k] * rot_angle_key_unit
            docstring += f'{k} '
    # North up, east left.  auto_rotate=True prefers rotation with
    # minimal image area, so force it to be False, though that is the
    # current default
    ne_wcs, _ = find_optimal_celestial_wcs([(ccd.data, ccd.wcs)],
                                           auto_rotate=False)
    # Clockwise rotation
    r_wcs = rot_wcs(ne_wcs, -rot_to_angle)
    ccd = wcs_project(ccd, r_wcs)
    if docstring:
        ccd.meta['HIERARCH ROT_FROM_KEYS'] = docstring
    return ccd
    
def flip_wcs(ccd_in, axis):
    """Flip ccd's WCS along axis 0 = up/down, 1 = left/right"""
    # https://en.wikipedia.org/wiki/Rotations_and_reflections_in_two_dimensions
    if axis == 0:
        mat = np.asarray(((1, 0), (0, -1)))
        fits_axis = 1
    elif axis == 1:
        mat = np.asarray(((-1, 0), (0, 1)))
        fits_axis = 0
    else:
        raise ValueError(f'{axis} is not 0 or 1')
    # Keeping in mind we are in FITS space, which is transposed
    mat = np.transpose(mat)
    ccd = ccd_in.copy()
    # wcs._naxis is not always set, so we have to do this from the CCD shape
    naxis = ccd.shape[::-1]
    crpix = ccd.wcs.wcs.crpix
    delta_cr_from_center = ccd.wcs.wcs.crpix[fits_axis] - naxis[fits_axis]/2
    crpix[fits_axis] = naxis[fits_axis]/2 - delta_cr_from_center
    ccd.wcs.wcs.crpix = crpix
    
    if ccd.wcs.wcs.has_cd():
        cd = np.matmul(ccd.wcs.wcs.cd, mat)
        ccd.wcs.wcs.cd = cd
    elif ccd.wcs.wcs.has_pc():
        pc = np.matmul(ccd.wcs.wcs.pc, mat)
        ccd.wcs.wcs.pc = pc
    else:
        raise ValueError('ccd.wcs.wcs does not have expected cd or pc '
                         'transformation matrix')
    return ccd

def flip_along(ccd_in,
               flip_along_axis=None,
               flip_along_angle=None,
               flip_angle_from_key=None,
               flip_angle_key_unit=u.degree,
               **kwargs):
    """Flip ccd image along desired axis so that flip_along_angle heads
    generally to the right (axis 0) or up (axis 1).  Usually, rot_to
    has already been applied

    Parameters
    ----------
    ccd_in : astropy.nddata.CCDData
        Input `~astropy.nddata.CCDData`

    flip_along_axis : int
      Axis about which to flip (required) 0 = up/down, 1 = left/right

    flip_along_angle : astropy.units.Quantity or None, optional
        Reference angle, N up, E west, CCW.  If None, 0 is used.
        Default is `None`
        
    flip_angle_from_key : str or list of str None
        FITS header key(s) from which flip angle is constructed.
        Values in keys are added to each other, including the initial
        value of rot_to_angle
        Default is `None`

    flip_angle_key_unit : astropy.units.Unit
        Unit applied to flip_angle_from_key
        Default is `astropy.units.deg`

    **kwargs : required as bigmultipipe post-processing routine

    """
    if flip_along_axis is None:
        raise ValueError('flip_along_axis must be specified')
    flip_along_angle = flip_along_angle or 0*u.deg
    ccd = ccd_in.copy()
    docstring = ''
    if flip_angle_from_key is not None:
        flip_angle_from_key = assure_list(flip_angle_from_key)
        for k in flip_angle_from_key:
            flip_along_angle += ccd.meta[k] * flip_angle_key_unit
            docstring += f'{k} '
    # Our flip_along_angle is referenced to celestial N, but there is
    # a good chance we have rotated relative to that with rot_to.  We
    # want to rotate our angle accordingly, but we have a routine
    # handy that rotates the wcs, from which we can read the angle
    flip_along_angle = rot_angle_by_wcs(flip_along_angle, ccd)
    flip_along_angle = Angle(flip_along_angle)
    abounds = np.asarray((90, 270))
    if flip_along_axis == 1:
        abounds -= 90
    abounds *= u.deg
    if flip_along_angle.is_within_bounds(*list(abounds)):
        # We have to flip the wcs separately and carefully because of
        # the underlying C structure.
        wcs = ccd.wcs
        ccd.wcs = None
        ccd = np.flip(ccd, flip_along_axis)
        ccd.wcs = wcs
        ccd = flip_wcs(ccd, flip_along_axis)
        ccd.meta['HIERARCH FLIP_AXIS'] = (flip_along_axis, 'flipped along C axis')
        if docstring:
            ccd.meta['HIERARCH FLIP_FROM_KEYS'] = docstring
    return ccd    

def is_flux(unit):
    """Returns ``True`` if unit is reducable to flux unit"""
    unit = unit.decompose()
    if isinstance(unit, u.IrreducibleUnit):
        return False
    # numpy tests don't work with these objects, so do it by hand
    spower = [p for un, p in zip(unit.bases, unit.powers)
              if un == u.s]
    if len(spower) == 0 or spower[0] != -1:
        return False
    return True

class Photometry:
    def __init__(self,
                 ccd=None, # Passed by reference since nothing operates ON it
                 seeing=5, # pixels
                 n_connected_pixels=5,
                 source_mask_dilate=11, # pixels
                 source_mask_nsigma=2, # sigma above background
                 n_back_boxes=10, # number of boxes in each dimension used to calculate background
                 back_rms_scale=5.0, # Scale background rms for threshold calculation
                 remove_masked_sources=True,
                 no_deblend=False,
                 deblend_nlevels=32,
                 deblend_contrast=0.001,
                 exptime_key=EXPTIME_KEY,
                 exptime_unit=EXPTIME_UNIT,
                 join_tolerance=JOIN_TOLERANCE*JOIN_TOLERANCE_UNIT,
                 keys_to_source_table=None,
                 source_table_cols=SOURCE_TABLE_COLS,
                 votable_fields=VOTABLE_FIELDS,
                 photoobj_fields=PHOTOOBJ_FIELDS,
                 **kwargs):
        self.ccd = ccd
        self.seeing = seeing
        self.n_connected_pixels = n_connected_pixels
        self.source_table_cols = source_table_cols
        self.source_mask_dilate = source_mask_dilate
        self.source_mask_nsigma = source_mask_nsigma
        self.n_back_boxes = n_back_boxes
        self.back_rms_scale = back_rms_scale
        self.remove_masked_sources = remove_masked_sources
        self._no_deblend = no_deblend
        self.deblend_nlevels = deblend_nlevels
        self.deblend_contrast = deblend_contrast
        self.exptime_key = exptime_key
        self.exptime_unit = exptime_unit
        self.join_tolerance = join_tolerance
        self.keys_to_source_table = keys_to_source_table
        self.votable_fields = votable_fields
        self.photoobj_fields = photoobj_fields
        self.init_calc()

    def init_calc(self):
        # Kernel ends up getting reset when it doesn't necessarily
        # need to be, but more pain than it is worth to address that
        self._kernel = None
        self._convolved_data = None
        self._source_mask = None
        self._back_obj = None
        self._background = None
        self._sig_clipped_stats = None
        self._back_rms = None
        self._threshold = None
        self._segm_image = None
        self.segm_failed = None
        self._segm_deblend = None
        self._source_catalog = None
        self._source_table = None
        self._solved = None
        self._wcs = None

    def precalc(self):
        if self.ccd is None:
            # Can't do too much, but at least put the kernel in.  This
            # is guaranteed to work because kernel dependencies
            # (seeing) has a default value
            self.kernel
            return
        self.source_catalog
        
    @control_property
    def seeing(self):
        pass

    @control_property
    def source_mask_dilate(self):
        pass
        
    @control_property
    def source_mask_nsigma(self):
        pass
    
    @control_property
    def n_connected_pixels(self):
        pass

    @control_property
    def n_back_boxes(self):
        pass

    @control_property
    def back_rms_scale(self):
        pass
    
    @property
    def no_deblend(self):
        return self._no_deblend

    @no_deblend.setter
    def no_deblend(self, value):
        if value != self._no_deblend:
            if value:
                log.warning('Preparing to recalculate source_catalog with deblended image')
            else:
                log.warning('Preparing to recalculate source_catalog with non-deblended image')
            self._segm_deblend = None
            self._source_catalog = None
        self._no_deblend = value
    
    @control_property
    def deblend_nlevels(self):
        pass
    
    @control_property
    def deblend_contrast(self):
        pass
    
    @control_property
    def ccd(self):
        pass

    @property
    def coverage_mask(self):
        if self.ccd.mask is None:
            return None
        return self.ccd.mask        

    @property
    def kernel(self):
        """Generic "round" kernel from a characterisitic seeing in FWHM.  This
        can get much more complicated in a subclass, a la GALEX
        grism-mode streaks, if necessary

        """
        if self._kernel is not None:
            return self._kernel
        sigma = self.seeing * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma)
        kernel.normalize()
        self._kernel = kernel
        return self._kernel

    @property
    def convolved_data(self):
        if self._convolved_data is not None:
            return self._convolved_data
        if self.ccd is None:
            raise ValueError('Oops, Photometry(ccd) was not set to anything')
        return convolve(self.ccd.data, self.kernel)

    @property
    def source_mask(self):
        """Make a source mask to enable optimal background estimation"""
        if self._source_mask is not None:
            return self._source_mask
        try:
            source_mask = make_source_mask(
                self.convolved_data,
                nsigma=self.source_mask_nsigma,
                npixels=self.n_connected_pixels,
                mask=self.coverage_mask,
                dilate_size=self.source_mask_dilate)
        except Exception as e:
            log.warning('Received the following error: ' + str(e))
            log.warning('source_mask will not be used')
            source_mask = np.zeros_like(self.ccd.mask, dtype=bool)
            
        self._source_mask = source_mask
        return self._source_mask

    def show_source_mask(self):
        impl = plt.imshow(self.source_mask, origin='lower',
                          cmap=plt.cm.gray,
                          filternorm=0, interpolation='none')
        plt.show()

    @property
    def back_obj(self):
        if self._back_obj is not None:
            return self._back_obj
        try: 
            box_size = int(np.mean(self.ccd.shape) / self.n_back_boxes)
            back = Background2D(self.ccd, box_size, mask=self.source_mask,
                                coverage_mask=self.coverage_mask)
            self._back_obj = back
            return self._back_obj
        except Exception as e:
            log.warning('Received the following error: ' + str(e))
            return None

    @property
    def sig_clipped_stats(self):
        """returns ()"""
        if self._sig_clipped_stats is not None:
            return self._sig_clipped_stats
        mean, median, std = sigma_clipped_stats(
            self.ccd.data, sigma=3.0, mask=self.ccd.mask)
        self._sig_clipped_stats = (
            mean*self.ccd.unit,
            median*self.ccd.unit,
            std*self.ccd.unit)
        return self._sig_clipped_stats
        
    @property
    def background(self):
        if self._background is not None:
            return self._background
        if self.back_obj is not None:
            self._background = self.back_obj.background
            return self._background

        mean, median, std = self.sig_clipped_stats
        log.warning(f'Setting background using median {median}')
        back = np.full_like(self.ccd, median)
        # np doesn't propagate units
        self._background = back*self.ccd.unit
        return self._background

    def show_background(self):
        impl = plt.imshow(self.background.value, origin='lower',
                          cmap=plt.cm.gray,
                          filternorm=0, interpolation='none')
        self.back_obj.plot_meshes()
        plt.show()

    @property
    def back_rms(self):
        if self._back_rms is not None:
            return self._back_rms
        if self.back_obj is not None:
            self._back_rms = self.back_obj.background_rms
            return self._back_rms
        
        mean, median, std = self.sig_clipped_stats
        log.warning(f'Setting background rms using image std {std}')
        rms = np.full_like(self.ccd, std)
        # np doesn't propagate units
        self._back_rms = rms*self.ccd.unit
        return self._back_rms

    @property
    def threshold(self):
        """Returns image of threshold values

        """
        if self._threshold is not None:
            return self._threshold
        self._threshold = (self.background
                           + self.back_rms_scale
                           * self.back_rms)
        return self._threshold
            
    @property
    def segm_image(self):
        if self.segm_failed:
            return None
        if self._segm_image is not None:
            return self._segm_image
        segm = detect_sources(self.convolved_data,
                              self.threshold.value,
                              npixels=self.n_connected_pixels,
                              mask=self.ccd.mask)
        if segm is None:
            self.segm_failed = True
            return None
        if (self.remove_masked_sources
            and self.ccd.mask is not None):
            segm.remove_masked_labels(self.ccd.mask, partial_overlap=True)
        self._segm_image = segm
        return self._segm_image

    @property
    def segm_deblend(self):
        # It does save a little time and a factor ~1.3 in memory if we
        # don't deblend
        if self.no_deblend:
            return self.segm_image
        if self._segm_deblend is not None:
            return self._segm_deblend
        if self.segm_image is None:
            # Pathological case of no sources found
            return None
        segm_deblend = deblend_sources(self.convolved_data,
                                       self.segm_image, 
                                       npixels=self.n_connected_pixels,
                                       kernel=self.kernel,
                                       nlevels=self.deblend_nlevels,
                                       contrast=self.deblend_contrast)
        self._segm_deblend = segm_deblend
        return self._segm_deblend

    def show_segm(self):
        """Show the segmentation image *that will be used for source catalog*.
This will be the *deblended* version by default.  Set
`Photometry.no_deblend`=True to display the non-deblended segmentation
image

        """
        impl = plt.imshow(self.segm_deblend, origin='lower',
                          cmap=plt.cm.gray,
                          filternorm=0, interpolation='none')
        plt.show()

    @property
    def source_catalog(self):
        if self._source_catalog is not None:
            return self._source_catalog
        if self.segm_deblend is None:
            # Pathological case of no sources found
            return None

        # As per calc_total_error documentation, effective_gain converts
        # ccd.data into count-based units, so it is exptime when we have
        # flux units
        # --> Eventually get units into this properly with the
        # CardQuantity stuff I have been working on, or however that turns
        # out, for now assume everything is in seconds
        if is_flux(self.ccd.unit):
            effective_gain = self.ccd.meta[self.exptime_key]*self.exptime_unit
        else:
            effective_gain = 1*self.exptime_unit

        if self.ccd.uncertainty is None:
            log.warning(f'photometry being conducted on ccd data with no uncertainty.  Is this file being properly reduced?  Soldiering on....')
            if self.ccd.unit == u.adu:
                log.warning(f'File is still in adu, cannot calculate proper '
                            f'Poisson error for sources')
                total_error = np.zeros_like(self.ccd)*u.adu
            else:
                total_error = \
                    calc_total_error(self.ccd,
                                     self.back_rms,
                                     effective_gain) 
        else:
            # BEWARE!  Units don't like being sqrted -- it is not a
            # reversable process from **2 even though they repr the same!
            assert self.ccd.unit == self.back_rms.unit
            uncert = self.ccd.uncertainty.array
            if self.ccd.uncertainty.uncertainty_type == 'std':
                total_error = np.sqrt(self.back_rms.value**2 + uncert**2)
            elif self.ccd.uncertainty.uncertainty_type == 'var':
                total_error = np.sqrt(self.back_rms.value**2 + var)
            else:
                raise ValueError(f'Unsupported uncertainty type {self.ccd.uncertainty.uncertainty_type} for {in_name}')
            total_error *= self.back_rms.unit

        # Call to SourceCatalog is a little awkward because it expects
        # a Quantity.  Don't forget to subtract background!  Note that
        # if _back is set, that means we have a somewhat pathological
        # case
        sc = SourceCatalog(
            self.ccd.data*self.ccd.unit - self.background,
            self.segm_deblend,
            convolved_data=self.convolved_data*self.ccd.unit,
            error=total_error,
            mask=self.ccd.mask,
            background=self.background)
        self._source_catalog = sc
        return self._source_catalog

    @property
    def source_table(self):
        if self._source_table is not None:
            return self._source_table
        # https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
        if self.source_catalog is None:
            return None
        self._source_table = self.source_catalog.to_table(
            self.source_table_cols)
        self._source_table.sort('segment_flux', reverse=True)
        return self._source_table
        
    @property
    def solved(self):
        if self._solved is None:
            self.wcs
        return self._solved

    @property
    def wcs(self):
        """Returns WCS using the astrometry solution of source_table 

        MUST BE SUBCLASSESED with specific code to extract pixel
        scale, FOV radius, etc. needed for astroquery.astrometry or
        astrometry.net code call

        """
        if self._wcs is not None:
            return self._wcs
        if self.source_catalog is None:
            self._solved = False
            return None
        # Let base class be used quietly in the case astrometry is not needed
        self._solved = False

    @property
    def source_table_has_coord(self):
        if self.source_table is None or not self.solved:
            return False
        if 'coord' in self.source_table.colnames:
            return True
        skies = self.wcs.pixel_to_world(
            self.source_table['xcentroid'],
            self.source_table['ycentroid'])
        self.source_table['coord'] = skies
        return True

    @property
    def source_table_has_key_cols(self):
        if self.source_table is None:
            return False
        if self.keys_to_source_table is None:
            return True
        for ku in self.keys_to_source_table:
            if isinstance(ku, tuple):
                k = ku[0]
                unit = ku[1]
            else:
                k = ku
                unit = None
            if k in self.source_table.colnames:
                continue
            val = self.ccd.meta.get(k)
            if val is None:
                log.warning(f'key {k} not found in FITS header')
                return False
            if unit is None:
                self.source_table[k] = val
            else:
                self.source_table[k] = val*unit
            self.source_table.meta[k] = self.ccd.meta.comments[k]
        return True

    @property
    def wide_source_table(self):
        if self.source_table_has_coord:
            return self.source_table
        return None    

    def plot_object(self, outname=None, expand_bbox=10, show=False, **kwargs):
        """This will probably need to be overridden to put in desirable title,
        any additional color bar axes, etc.

        """ 
        # Don't gunk up the source_table ecsv with bounding box
        # stuff, but keep in mind we sorted it, so key off of
        # label
        bbox_table = self.source_catalog.to_table(
            ['label', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'])
        mask = (self.wide_source_table['OBJECT']
                == self.ccd.meta['OBJECT'])
        label = self.wide_source_table[mask]['label']
        bbts = bbox_table[bbox_table['label'] == label]
        xmin = bbts['bbox_xmin'][0] - expand_bbox
        xmax = bbts['bbox_xmax'][0] + expand_bbox
        ymin = bbts['bbox_ymin'][0] - expand_bbox
        ymax = bbts['bbox_ymax'][0] + expand_bbox
        source_ccd = self.ccd[ymin:ymax, xmin:xmax]
        threshold = self.threshold[ymin:ymax, xmin:xmax]
        segm = self.segm_image.data[ymin:ymax, xmin:xmax]
        ax = plt.subplot(projection=source_ccd.wcs)
        ims = plt.imshow(source_ccd)
        cbar = plt.colorbar(ims, fraction=0.03, pad=0.02)
        pos = cbar.ax.get_position()
        cax1 = cbar.ax
        cax1.set_aspect('auto')
        cax1.set_ylabel(source_ccd.unit)

        ax.contour(segm, levels=0, colors='white')
        ax.contour(source_ccd - threshold,
                   levels=0, colors='gray')
        ax.set_title(f'{self.ccd.meta["DATE-OBS"]}')
        if show:
            plt.show()
        # Note. savefig does not overwrite
        plt.savefig(outname)

    @property
    def av_coord(self):
        """Returns tuple:        

        `~astropy.coordinates.SkyCoord` : average RA and DEC of
        current entries in
        `:prop:IoIO.photometry.Photometry.source_table '

        `~astropy.coordinates.Angle`: distance of each source to
        average RA and DEC

        """
        ra = self.wide_source_table['coord'].ra
        dec = self.wide_source_table['coord'].dec
        ccoord = SkyCoord(np.average(ra), np.average(dec))
        dists = ccoord.separation(tab['coord'])
        return (SkyCoord(ra, dec), dists)        

    @property
    def simbad_table(self):
        """Returns `~astropy.table.QTable` with Simbad query results at
        average RA and DEC of
        `:prop:IoIO.photometry.Photometry.wide_source_table'

        """
        ccoord, dists = self.av_coord
        sim = Simbad()
        sim.add_votable_fields('flux(U)', 'flux(B)', 'flux(V)', 'flux(R)',
                               'flux(I)')
        return sim.query_region(ccoord, radius=np.max(dists/2))

    @property
    def sdss_table(self):
        """Returns `~astropy.table.Table` with Simbad query results at
        average RA and DEC of
        `:prop:IoIO.photometry.Photometry.wide_source_table'

        """
        sdss = SDSS()

        #ccoord = SkyCoord(143.50993, 55.239775, unit="deg")
        #ccoord = SkyCoord(12, 30, unit=("hour", "deg"))

        photoobj_fields = ['ra', 'dec', 'probPSF', 'aperFlux7_u',
                           'aperFlux7_g', 'aperFlux7_r',
                           'aperFlux7_i', 'aperFlux7_z']
        sdss_results = sdss.query_region(ccoord, radius=np.max(dists/2),
                                         photoobj_fields=photoobj_fields)
        sdss_star_mask = sdss_results['probPSF'] == 1

    @property
    def cat_table(self):
        """Table of catalog query  table including all tagalongs of locally
        downloaded data

        """
        if self._cat_table is not None:
            return self._cat_table
        if self.solved:
            return self._cat_table
        return None

class PhotometryArgparseMixin:
    def add_join_tolerance(self, 
                 default=JOIN_TOLERANCE,
                 help=None,
                 **kwargs):
        option = 'join_tolerance'
        if help is None:
            help = (f'catalog join matching tolerance in join_tolerance_unit '
                    f'(default: {default})')
        self.parser.add_argument('--' + option, type=float,
                                 default=default, help=help, **kwargs)

    def add_join_tolerance_unit(self, 
                 default=JOIN_TOLERANCE_UNIT.to_string(),
                 help=None,
                 **kwargs):
        option = 'join_tolerance_unit'
        if help is None:
            help = (f'unit of catalog join matching tolerance '
                    f'(default: {default})')
        self.parser.add_argument('--' + option, 
                                 default=default, help=help, **kwargs)

if __name__ == '__main__':
    import glob
    import ccdproc as ccdp
    from IoIO.cordata import CorData
    from IoIO.cor_process import cor_process
    from IoIO.calibration import Calibration
    ccd = CorData.read('/data/Mercury/2021-10-28/Mercury-0001_Na_on-back-sub_Mercury_N_up.fits')
    north = ccd.meta['TARGET_NPole_ang']*u.deg
    antisun = ccd.meta['TARGET_sunTargetPA']*u.deg
    print(f'original north {north}')
    north = rot_angle_by_wcs(north, ccd)
    print(f'rotated by wcs {north}\n')
    print(f'original antisun {antisun}')
    antisun = rot_angle_by_wcs(antisun, ccd)
    print(f'rotated by wcs {antisun}')





    #directory = '/data/IoIO/raw/2021-10-28/'
    #collection = ccdp.ImageFileCollection(
    #    directory, glob_include='Mercury*',
    #    glob_exclude='*_moving_to*')
    #flist = collection.files_filtered(include_path=True)
    #c = Calibration(reduce=True)
    #photometry = Photometry()
    #for fname in flist:
    #    log.info(f'trying {fname}')
    #    rccd = CorData.read(fname)
    #    ccd = cor_process(rccd, calibration=c, auto=True)
    #    photometry.ccd = ccd
    #    #photometry.show_segm()
    #    source_table = photometry.source_table
    #    source_table.show_in_browser()
