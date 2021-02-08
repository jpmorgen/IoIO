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
        super().__init__(*args, **kwargs)
        self._ND_mask = None
        self.seeing = 5 # pixels FWHM

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
        return ~self.NDmask

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


        # --> this is from photometry_process.  It would be great if
        # --> we could use that code generically

        sigma = seeing * gaussian_fwhm_to_sigma
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
        
        ##mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
        #
        
        box_size = int(np.mean(self.HDUList[0].data.shape) / 10)
        back = Background2D(self.HDUList[0].data, box_size,
                            mask=mask, coverage_mask=self.ND_only_mask)
        threshold = back.background + (2.0* back.background_rms)
    
        print(f'background_median = {back.background_median}, background_rms_median = {back.background_rms_median}')
        
        #impl = plt.imshow(back.background, origin='lower',
        #                  cmap=plt.cm.gray,
        #                  filternorm=0, interpolation='none')
        #back.plot_meshes()
        #plt.show()
        
        npixels = 5
        segm = detect_sources(self.HDUList[0].data, threshold, npixels=npixels,
                              filter_kernel=kernel, mask=ccd.mask)
        
        if segm is not None:
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
            tbl = cat.to_table()
            tbl.sort('source_sum', reverse=True)

            
        # It does save a little time and a factor ~1.3 in memory if we
        # don't deblend
        segm_deblend = deblend_sources(self.HDUList[0].data, segm,
                                       npixels=npixels,
                                       filter_kernel=kernel, nlevels=32,
                                       contrast=0.001)
            

        #### Work another way to see if the ND filter has a low flux
        #### Note, this assignment dereferences im from HDUList[0].data
        ###im  = im - back_level
        ###satlevel -= back_level
        ###
        #### Get the coordinates of the ND filter
        ###NDc = self.ND_coords
        ###
        #### Filter ND coords for ones that are at least 5 std of the
        #### bias noise above the median.  Calculate a fresh median for
        #### the ND filter just in case it is different than the median
        #### of the image as a whole (which is now 0 -- see above).  We
        #### can't use the std of the ND filter, since it is too biased
        #### by Jupiter when it is there.
        ###NDmed = np.median(im[NDc])
        ###boostc = np.where(im[NDc] > (NDmed + 5*self.biasnoise))
        ###boost_NDc0 = np.asarray(NDc[0])[boostc]
        ###boost_NDc1 = np.asarray(NDc[1])[boostc]
        ###
        #### Come up with a metric for when Jupiter is in the ND filter.
        #### Below is my scratch work        
        #### Rj = np.asarray((50.1, 29.8))/2. # arcsec
        #### plate = 1.59/2 # main "/pix
        #### 
        #### Rj/plate # Jupiter pixel radius
        #### array([ 31.50943396,  18.74213836])
        #### 
        #### np.pi * (Rj/plate)**2 # Jupiter area in pix**2
        #### array([ 3119.11276312,  1103.54018437])
        ####
        #### Jupiter is generally better than 1000
        #### 
        #### np.pi * (Rj/plate)**2 * 1000 
        #### array([ 3119112.76311733,  1103540.18436529])
        ###
        ###sum_on_ND_filter = np.sum(im[boost_NDc0, boost_NDc1])
        #### Adjust for the case where ND filter may have a fairly
        #### high sky background.  We just want Jupiter
        ###sum_on_ND_filter -= NDmed * len(boost_NDc0)
        
        #log.debug('sum of significant pixels on ND filter = ' + str(sum_on_ND_filter))
        print('sum_on_ND_filter = ', sum_on_ND_filter)
        #if num_sat > 1000 or sum_on_ND_filter < 1E6:
        # Vega is 950,000
        if num_sat > 1000 or sum_on_ND_filter < 0.75E6:
            log.warning('Jupiter outside of ND filter?')
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
        else:
            # Here is where we boost what is sure to be Jupiter, if Jupiter is
            # in the ND filter
            # --> this has trouble when there is bright skys
            im[boost_NDc0, boost_NDc1] *= 1000
            # Clean up any signal from clouds off the ND filter, which can
            # mess up the center of mass calculation
            im[np.where(im < satlevel)] = 0
            y_x = ndimage.measurements.center_of_mass(im)
    
            #print(y_x[::-1])
            #plt.imshow(im)
            #plt.show()
            #return (y_x[::-1], ND_center)
    
            # Stay in Pythonic y, x coords
            self._obj_center = np.asarray(y_x)
            log.debug('Object center (X, Y; binned) = '
                      + str(self.binned(self._obj_center)[::-1]))
            self.quality = 6
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
o = RedCorObsData(jup)
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
