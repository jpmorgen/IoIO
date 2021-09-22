"""I think this is obsolete
"""

from scipy import signal, ndimage
import numpy as np

from astropy import log
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats

from photutils import make_source_mask
from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties
from photutils.background import Background2D

import sx694
from IoIO import CorObsData

class RedCorObsData(CorObsData):
    """Object helpful for reducing coronagraph observations
    """

    def __init__(self,
                 *args, **kwargs):
        self._ND_mask = None
        self.seeing = 5 # pixels FWHM
        super().__init__(*args, **kwargs)

    def cleanup(self):
        pass

    @property
    def ND_mask(self):
        """Masks ND filter"""
        if self._ND_mask is not None:
            return self._ND_mask
        im = self.HDUList[0].data
        mask = np.zeros(im.shape, bool)
        mask[self.ND_coords] = True
        self._ND_mask = mask
        return self._ND_mask

    @property
    def ND_only_mask(self):
        """Masks entire image but ND filter"""
        # This should be fast enough
        return ~self.ND_mask

    @property
    def obj_center(self):
        """Returns center pixel coords of Jupiter whether or not Jupiter is on ND filter.  Unbinned pixel coords are returned.  Use [Cor]Obs_Data.binned() to convert to binned pixels.
        """
        # Returns stored center for object, None for flats
        if self._obj_center is not None or self.isflat:
            return self._obj_center

        # Work with unbinned image
        im = self.HDU_unbinned
        back_level = self.back_level / (np.prod(self._binning))

        satlevel = self.header.get('SATLEVEL')
        if satlevel is None:
            satlevel = sx694.satlevel

        # Establish some metrics to see if Jupiter is on or off the ND
        # filter.  Easiest one is number of saturated pixels
        # /data/io/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit gives
        # 4090 of these.  Calculation below suggests 1000 should be a
        # good minimum number of saturated pixels (assuming no
        # additional scattered light).  A star off the ND filter
        # /data/io/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit
        # gives 124 num_sat
        satc = np.where(im >= satlevel)
        num_sat = len(satc[0])
        #log.debug('Number of saturated pixels in image: ' + str(num_sat))


        # --> this is from photometry_process to see if Jupiter is on
        # --> the ND filter.  It would be great if we could use that
        # --> code generically

        sigma = self.seeing * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma)
        kernel.normalize()
        # Make a source mask to enable optimal background estimation
        mask = make_source_mask(self.HDUList[0].data, nsigma=2, npixels=5,
                                filter_kernel=kernel, mask=self.ND_only_mask,
                                dilate_size=11)
        #impl = plt.imshow(mask, origin='lower',
        #                  cmap=plt.cm.gray,
        #                  filternorm=0, interpolation='none')
        #plt.show()
        
        mean, median, std = sigma_clipped_stats(self.HDUList[0].data,
                                                sigma=3.0,
                                                mask=self.ND_only_mask)
        threshold = median + (2.0 * std)
        
        ### This seems too fancy for narrow ND filter
        ##box_size = int(np.mean(self.HDUList[0].data.shape) / 10)
        ##back = Background2D(self.HDUList[0].data, box_size,
        ##                    mask=mask, coverage_mask=self.ND_only_mask)
        ##threshold = back.background + (2.0* back.background_rms)
        ##
        ##print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')
        
        #impl = plt.imshow(back.background, origin='lower',
        #                  cmap=plt.cm.gray,
        #                  filternorm=0, interpolation='none')
        #back.plot_meshes()
        #plt.show()
        
        npixels = 5
        segm = detect_sources(self.HDUList[0].data, threshold, npixels=npixels,
                              filter_kernel=kernel,  mask=self.ND_only_mask)
        
        if segm is not None:
            # Some object found on the ND filter

            # It does save a little time and a factor ~1.3 in memory if we
            # don't deblend
            segm_deblend = deblend_sources(self.HDUList[0].data,
                                           segm, npixels=npixels,
                                           filter_kernel=kernel, nlevels=32,
                                           contrast=0.001)

            #impl = plt.imshow(segm, origin='lower',
            #                  cmap=plt.cm.gray,
            #                  filternorm=0, interpolation='none')
            #plt.show()

            cat = source_properties(self.HDUList[0].data,
                                    segm_deblend,
                                    mask=self.ND_only_mask)
            tbl = cat.to_table(('source_sum',
                                'moments',
                                'moments_central',
                                'inertia_tensor',
                                'centroid'))
            tbl.sort('source_sum', reverse=True)

            #xcentrd = tbl['xcentroid'][0].value
            #ycentrd = tbl['ycentroid'][0].value
            print(tbl['moments'][0])
            print(tbl['moments_central'][0])
            print(tbl['inertia_tensor'][0])
            print(tbl['centroid'][0])
            centroid = tbl['centroid'][0]

            self._obj_center = centroid

            #self._obj_center = np.asarray((ycentrd, xcentrd))
            log.debug('Object center (X, Y; binned) = '
                      + str(self.binned(self._obj_center)[::-1]))
            self.quality = 6
        else:
            log.warning('No object found on ND filter')
            # Outside the ND filter, Jupiter should be saturating.  To
            # make the center of mass calc more accurate, just set
            # everything that is not getting toward saturation to 0
            # --> Might want to fine-tune or remove this so bright
            im[np.where(im < satlevel*0.7)] = 0
            
            #log.debug('Approx number of saturating pixels ' + str(np.sum(im)/65000))

            # 25 worked for a star, 250 should be conservative for
            # Jupiter (see above calcs)
            # if np.sum(im) < satlevel * 25:
            if np.sum(im) < satlevel * 250:
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

        self.header['OBJ_CR0'] = (self._obj_center[1], 'Object center X')
        self.header['OBJ_CR1'] = (self._obj_center[0], 'Object center Y')
        self.header['QUALITY'] = (self.quality, 'Quality on 0-10 scale of center determination')
        return self._obj_center

#R_off_ND = '/data/io/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit'
#o = RedCorObsData(R_off_ND)
##sum_on_ND_filter =  161858633.36258712
##sum_on_ND_filter =  160641649.00000006

# Jupiter lost (offset)!
jup = '/data/io/IoIO/raw/20200507/Jupiter-S004-R001-C001-Na_on_dupe-1.fts'
o = CorObsData(jup)
o = RedCorObsData(jup)
# Pre-improvement
#sum_on_ND_filter =  227854.2544777951
# Post-improvement
#sum_on_ND_filter =  253682.00000000003
# RedCorObs should be the same
#sum_on_ND_filter =  253681.9999999999
#sum_on_ND_filter =  253681.9999999999
# <identical after porting back_level>

# Jupiter found
jup = '/data/io/IoIO/raw/20200507/Na_on-band_002.fits'
o = CorObsData(jup)
print(o.obj_center)
o = RedCorObsData(jup)
print(o.obj_center)
# Pre-improvement
#sum_on_ND_filter =  10136130.15718806
# Post-improvement
#sum_on_ND_filter =  11169575.000000002
# RedCorObs should be the same
#sum_on_ND_filter =  10922457.999999998
#sum_on_ND_filter =  11169575.0
# <identical after porting back_level>

#fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
#fname2 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na_off.fit'
#o = RedCorObsData(fname1)
##sum_on_ND_filter =  1851602.995563588
##sum_on_ND_filter =  1853482.9999999998
#o = RedCorObsData(fname2)
##sum_on_ND_filter =  239150.53658536595
##sum_on_ND_filter =  388940.0

o.ND_mask
