"""IoIO Photometry object"""

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats

from photutils import make_source_mask, detect_sources, deblend_sources
from photutils import SourceCatalog
from photutils.background import Background2D
from photutils.utils import calc_total_error

from precisionguide import pgproperty

from IoIO.utils import is_flux

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

class Photometry:
    def __init__(self,
                 ccd=None,
                 seeing=5, # pixels
                 n_connected_pixels=5,
                 source_mask_dilate=11, # pixels
                 source_mask_nsigma=2, # pixels
                 n_back_boxes=10, # number of boxes in each dimension used to calculate background
                 back_rms_scale=2.0, # Scale background rms for threshold calculation
                 no_deblend=False,
                 deblend_nlevels=32,
                 deblend_contrast=0.001,
                 keys_to_source_table=None,
                 **kwargs):
        self.ccd = ccd
        self.seeing = seeing
        self.n_connected_pixels = n_connected_pixels
        self.source_mask_dilate = source_mask_dilate
        self.source_mask_nsigma = source_mask_nsigma
        self.n_back_boxes = n_back_boxes
        self.back_rms_scale = back_rms_scale
        self._no_deblend = no_deblend
        self.deblend_nlevels = deblend_nlevels
        self.deblend_contrast = deblend_contrast
        self.keys_to_source_table = keys_to_source_table
        self.init_calc()

    def init_calc(self):
        # Kernel ends up getting reset when it doesn't necessarily
        # need to be, but more pain than it is worth to address that
        self._kernel = None
        self._source_mask = None
        self._back_obj = None
        self._threshold = None
        self._segm_image = None
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
    def source_mask(self):
        """Make a source mask to enable optimal background estimation"""
        if self._source_mask is not None:
            return self._source_mask
        source_mask = make_source_mask(self.ccd.data,
                                       nsigma=self.source_mask_nsigma,
                                       npixels=self.n_connected_pixels,
                                       filter_kernel=self.kernel,
                                       mask=self.coverage_mask,
                                       dilate_size=self.source_mask_dilate)
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
        box_size = int(np.mean(self.ccd.shape) / self.n_back_boxes)
        back = Background2D(self.ccd, box_size, mask=self.source_mask,
                            coverage_mask=self.coverage_mask)
        self._back_obj = back
        return self._back_obj

    def show_background(self):
        impl = plt.imshow(self.back_obj.background.value, origin='lower',
                          cmap=plt.cm.gray,
                          filternorm=0, interpolation='none')
        self.back_obj.plot_meshes()
        plt.show()

    @property
    def threshold(self):
        """Returns image of threshold values

        """
        if self._threshold is not None:
            return self._threshold
        self._threshold = (self.back_obj.background
                           + self.back_rms_scale
                           * self.back_obj.background_rms)
        return self._threshold

    @property
    def segm_image(self):
        if self._segm_image is not None:
            return self._segm_image
        segm = detect_sources(self.ccd.data,
                              self.threshold.value,
                              npixels=self.n_connected_pixels,
                              filter_kernel=self.kernel, mask=self.ccd.mask)
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
        segm_deblend = deblend_sources(self.ccd.data,
                                       self.segm_image, 
                                       npixels=self.n_connected_pixels,
                                       filter_kernel=self.kernel,
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
            effective_gain = self.ccd.meta['EXPTIME']*u.s
        else:
            effective_gain = 1*u.s

        if self.ccd.uncertainty is None:
            log.warning(f'photometry being conducted on ccd data with no uncertainty.  Is this file being properly reduced?  Soldiering on....')
            if self.ccd.unit == u.adu:
                log.warning(f'File is still in adu, cannot calculate proper '
                            f'Poisson error for sources')
                total_error = np.zeros_like(self.ccd)*u.adu
            else:
                total_error = \
                    calc_total_error(self.ccd,
                                     self.back_obj.background_rms,
                                     effective_gain) 
        else:
            uncert = self.ccd.uncertainty.array*self.ccd.unit
            if self.ccd.uncertainty.uncertainty_type == 'std':
                total_error = np.sqrt(self.back_obj.background_rms**2
                                      + uncert**2)
            elif self.ccd.uncertainty.uncertainty_type == 'var':
                # --> if I work in units make var units ccd.unit**2
                var  = uncert*self.ccd.unit
                total_error = np.sqrt(self.back_obj.background_rms**2 + var)
            else:
                raise ValueError(f'Unsupported uncertainty type {self.ccd.uncertainty.uncertainty_type} for {in_name}')

        # Call to SourceCatalog is a little awkward because it expects
        # a Quantity.  Don't forget to subtract background!
        back = self.back_obj.background
        sc = SourceCatalog(self.ccd.data*self.ccd.unit - back,
                           self.segm_deblend, 
                           error=total_error,
                           mask=self.ccd.mask,
                           kernel=self.kernel,
                           background=back)
        self._source_catalog = sc
        return self._source_catalog

    @property
    def source_table(self):
        if self._source_table is not None:
            return self._source_table
        if self.source_catalog is None:
            return None
        self._source_table = self.source_catalog.to_table()
        self._source_table.sort('segment_flux', reverse=True)
        self.source_table_postprocess()
        return self._source_table

    @property
    def solved(self):
        if self._solved is None:
            self.astrometry
        return self._solved

    @property
    def wcs(self):
        """Returns WCS using the source_table

        """
        if self._wcs is not None:
            return self._wcs
        if self.source_catalog is None:
            self._solved = False
            return None
        assert False, 'Code not written to use astroquery.astrometry_net'

    @property
    def source_table_has_coord(self):
        if 'coord' in self.source_table.colnames:
            return True
        if not self.solved:
            return False
        skies = self.wcs.pixel_to_world(
            self.source_table['xcentroid'],
            self.source_table['ycentroid'])
        self.source_table['coord'] = skies
        return True

    @property
    def source_table_has_object(self):
        assert False, 'Code needs to be customized of identify object and add a column named OBJECT'

    @property
    def source_table_has_key_cols(self):
        if self.source_table is None:
            return False
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
        if (self.source_table_has_coord
            and self.source_table_has_object
            and self.source_table_has_key_cols):
            return self.source_table
        return None    

    def plot_object(self, outname=None, expand_bbox=10, show=False, **kwargs):
        # Don't gunk up the source_table ecsv with bounding box
        # stuff, but keep in mind we sorted it, so key off of
        # label
        bbox_table = photometry.source_catalog.to_table(
            ['label', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'])
        mask = (photometry.wide_source_table['OBJECT']
                == ccd.meta['OBJECT'])
        label = self.wide_source_table[mask]['label']
        bbts = bbox_table[bbox_table['label'] == label]
        xmin = bbts['bbox_xmin'][0] - expand_bbox
        xmax = bbts['bbox_xmax'][0] + expand_bbox
        ymin = bbts['bbox_ymin'][0] - expand_bbox
        ymax = bbts['bbox_ymax'][0] + expand_bbox
        source_ccd = self.ccd[ymin:ymax, xmin:xmax]
        threshold = self.threshold[ymin:ymax, xmin:xmax]
        segm = self.segm_image.data[ymin:ymax, xmin:xmax]
        #https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
        ax = plt.subplot(projection=source_ccd.wcs)
        ims = plt.imshow(source_ccd)
        cbar = plt.colorbar(ims, fraction=0.03, pad=0.11)
        pos = cbar.ax.get_position()
        cax1 = cbar.ax
        cax1.set_aspect('auto')
        cax2 = cax1.twinx()
        ylim = np.asarray(cax1.get_ylim())
        nonlin = source_ccd.meta['NONLIN']
        cax1.set_ylabel(source_ccd.unit)
        cax2.set_ylim(ylim/nonlin*100)
        cax1.yaxis.set_label_position('left')
        cax1.tick_params(labelrotation=90)
        cax2.set_ylabel('% nonlin')

        ax.contour(segm, levels=0, colors='white')
        ax.contour(source_ccd - threshold,
                   levels=0, colors='gray')
        ax.set_title(f'{ccd.meta["OBJECT"]} {ccd.meta["DATE-AVG"]}')
        savefig_overwrite(outroot + '.png')

    @property
    def source_with_simbad(self):
        """Source catalog with simbad query results"""
        assert False, 'Code not written to use, astroquery.simbad'

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

