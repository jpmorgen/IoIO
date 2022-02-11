"""This can be incorporated into cordata_base"""

from precisionguide.utils import hist_of_im, iter_linfit

# Unbinned coords --> Note, what with the poor filter wheel centering
# after fall 2020, this might need to change into a function that
# floats around, though for now it is just used to speed finding hot
# pixels and so is OK
SMALL_FILT_CROP = ((350, 550), (1900, 2100))

class CorDataNDparams(CorDataBase):
    def __init__(self, data,
                 n_y_steps=8, # was 15 (see adjustment in flat code)
                 x_filt_width=25,
                 cwt_width_arange_flat=None, # flat ND_params edge find
                 cwt_width_arange=None, # normal ND_params edge find
                 cwt_min_snr=1, # Their default seems to work well
                 search_margin=50, # on either side of nominal ND filter
                 max_fit_delta_pix=25, # Thowing out point in 1 line fit
                 max_parallel_delta_pix=50, # Find 2 lines inconsistent
                 max_ND_width_range=(80,400), # jump-starting flats & sanity check others
                 small_filt_crop=np.asarray(SMALL_FILT_CROP),
                 plot_prof=False,
                 plot_dprof=False,
                 plot_ND_edges=False,
                 show=False, # Show images
                 copy=False,
                 **kwargs):
        # Pattern after NDData init but skip all the tests
        if isinstance(data, CorDataNDparams):
            n_y_steps = data.n_y_steps
            x_filt_width = data.x_filt_width
            cwt_width_arange_flat = data.cwt_width_arange_flat
            cwt_width_arange = data.cwt_width_arange
            cwt_min_snr = data.cwt_min_snr
            search_margin = data.search_margin
            max_fit_delta_pix = data.max_fit_delta_pix
            max_parallel_delta_pix = data.max_parallel_delta_pix
            max_ND_width_range = data.max_ND_width_range
            small_filt_crop = data.small_filt_crop
            plot_prof = data.plot_prof
            plot_dprof = data.plot_dprof
            plot_ND_edges = data.plot_ND_edges
            show = data.show
        if copy:
            n_y_steps = deepcopy(n_y_steps)
            x_filt_width = deepcopy(x_filt_width)
            cwt_width_arange_flat = deepcopy(cwt_width_arange_flat)
            cwt_width_arange = deepcopy(cwt_width_arange)
            cwt_min_snr = deepcopy(cwt_min_snr)
            search_margin = deepcopy(search_margin)
            max_fit_delta_pix = deepcopy(max_fit_delta_pix)
            max_parallel_delta_pix = deepcopy(max_parallel_delta_pix)
            max_ND_width_range = deepcopy(max_ND_width_range)
            small_filt_crop = deepcopy(small_filt_crop)
            plot_prof = deepcopy(plot_prof)
            plot_dprof = deepcopy(plot_dprof)
            plot_ND_edges = deepcopy(plot_ND_edges)
            show = deepcopy(show)

        super().__init__(data, copy=copy, **kwargs)
        if cwt_width_arange_flat is None:
            cwt_width_arange_flat 	= np.arange(2, 60)
        self.cwt_width_arange_flat  	= cwt_width_arange_flat
        if cwt_width_arange is None:
            cwt_width_arange        	= np.arange(8, 80)
        self.cwt_width_arange           = cwt_width_arange       
        self.n_y_steps              	= n_y_steps              
        self.x_filt_width           	= x_filt_width
        self.cwt_min_snr            	= cwt_min_snr            
        self.search_margin          	= search_margin           
        self.max_fit_delta_pix      	= max_fit_delta_pix      
        self.max_parallel_delta_pix 	= max_parallel_delta_pix
        self.max_ND_width_range		= max_ND_width_range
        self.small_filt_crop        	= np.asarray(small_filt_crop)
        self.plot_prof			= plot_prof 
        self.plot_dprof             	= plot_dprof
        self.plot_ND_edges	    	= plot_ND_edges
        self.show 			= show

        self.arithmetic_keylist = ['satlevel', 'nonlin']
        self.handle_image = keyword_arithmetic_image_handler

    def _init_args_copy(self, kwargs):
        kwargs = super()._init_args_copy(kwargs)
        obj_dict = self.__dict__
        kwargs['n_y_steps'] = self.n_y_steps
        kwargs['x_filt_width'] = self.x_filt_width
        kwargs['cwt_width_arange_flat'] = self.cwt_width_arange_flat
        kwargs['cwt_width_arange'] = self.cwt_width_arange
        kwargs['cwt_min_snr'] = self.cwt_min_snr
        kwargs['search_margin'] = self.search_margin
        kwargs['max_fit_delta_pix'] = self.max_fit_delta_pix
        kwargs['max_parallel_delta_pix'] = self.max_parallel_delta_pix
        kwargs['max_ND_width_range'] = self.max_ND_width_range
        kwargs['small_filt_crop'] = self.small_filt_crop
        kwargs['plot_prof'] = self.plot_prof
        kwargs['plot_dprof'] = self.plot_dprof
        kwargs['plot_ND_edges'] = self.plot_ND_edges
        kwargs['show'] = self.show
        return kwargs
        
    @pgproperty
    def ND_params(self):
        """Returns parameters which characterize the coronagraph ND filter.
        Parameters are relative to *unbinned image.* It is important
        this be calculated on a per-image bases, since flexure in the
        instrument can shift it a bit side-to-side.  Unforunately,
        contrast is not sufficient to accurately spot the ND_filter in
        normal dark sky exposures without some sort of close starting
        point.  This is addressed by using the flats, which do have
        sufficient contrast, to provide RUN_LEVEL_DEFAULT_ND_PARAMS.

        """
        # Biases and darks don't have signal to spot ND filter
        if (self.imagetyp == 'bias'
            or self.imagetyp == 'dark'):
            return super().ND_params

        # Transform everything to our potentially binned and subframed
        # image to find the ND filter, but always return ND_params in
        # unbinned
        default_ND_params = self.ND_params_binned(self.default_ND_params)
        ND_ref_y = self.y_binned(self.ND_ref_y)
        _, ND_ref_x = self.ND_edges(ND_ref_y, default_ND_params, ND_ref_y)

        # Sanity check
        ND_ref_pt = np.asarray((ND_ref_y, ND_ref_x))
        im_ref_pt = np.asarray(self.shape) / 2
        ND_ref_to_im_cent = np.linalg.norm(ND_ref_pt - im_ref_pt)
        im_cent_to_edge = np.linalg.norm(im_ref_pt)
        if ND_ref_to_im_cent > im_cent_to_edge*0.80:
            log.warning('Subarray does not include enough of the ND filter to determine ND_params')
            return self.default_ND_params

        small_filt_crop = self.coord_binned(self.small_filt_crop,
                                            limit_edges=True)
        ytop = small_filt_crop[0,0]
        ybot = small_filt_crop[1,0]
        # x_filt_width has to be an odd integer
        x_filt_width = self.x_filt_width/self.binning[1]
        x_filt_width /= 2
        x_filt_width = 2 * np.round(x_filt_width)
        x_filt_width = np.int(x_filt_width + 1)
        search_margin = self.search_margin / self.binning[1]
        max_ND_width_range = self.max_ND_width_range / self.binning[1]
        
        # We don't need error or mask stuff, so just work with the
        # data array, which we will call "im"

        if self.imagetyp == 'flat':
            # Flats have high contrast and low sensitivity to hot
            # pixels, so we can work with the whole image.  It is OK
            # that this is not a copy of self.im since we are not
            # going to muck with it.  Since flats are high enough
            # quality, we use them to independently measure the
            # ND_params, so there is no need for the default (in fact
            # that is how we derive it!).  Finally, The flats produce
            # very narrow peaks in the ND_param algorithm when
            # processed without a default_ND_param and there is a
            # significant filter rotation.  Once things are morphed by
            # the default_ND_params (assuming they match the image),
            # the peaks are much broader.  So our cwt arange needs to
            # be a little different.
            im = self.data
            default_ND_params = None
            cwt_width_arange = self.cwt_width_arange_flat/self.binning[1]
            n_y_steps = 25/self.binning[0]
        else:
            # Non-flat case
            cwt_width_arange = self.cwt_width_arange/self.binning[1]
            # Increased S/N when binned
            n_y_steps = self.n_y_steps*self.binning[0]
            # Do a quick filter to get rid of hot pixels in awkward
            # places.  Do this only for stuff inside small_filter_crop
            # since it is our most time-consuming step.  Also work in
            # our original, potentially binned image, so hot pixels
            # don't get blown up by unbinning.  This is also a natural
            # place to check that we have default_ND_params in the
            # non-flat case and warn accordingly.

            # Obsolete code I only used in the beginning
            #if default_ND_params is None:
            #    log.warning('No default_ND_params specified in '
            #                'non-flat case.  This is likely to result '
            #                'in a poor ND_coords calculation.')
            #    # For filtering hot pixels, this doesn't need to be
            #    # super precise
            #    tb_ND_params = self.ND_params_binned(self.ND_params)
            #else:
            #    tb_ND_params = default_ND_params
            
            # Make a copy so we don't mess up the primary data array
            im = self.data.copy()
            es_top, _ = self.ND_edges(ytop, default_ND_params, ND_ref_y)
            es_bot, _ = self.ND_edges(ybot, default_ND_params, ND_ref_y)
            # Get the far left and right coords, keeping in mind ND
            # filter might be oriented CW or CCW of vertical
            x0 = int(np.min((es_bot, es_top))
                     - search_margin / self.binning[1])
            x1 = int(np.max((es_bot, es_top))
                     + search_margin / self.binning[1])
            x0 = np.max((0, x0))
            x1 = np.min((x1, im.shape[1]))
            # This is the operation that messes with the array in place
            im[ytop:ybot, x0:x1] \
                = signal.medfilt(im[ytop:ybot, x0:x1], 
                                 kernel_size=3)
            
        # At this point, im may or may not be a copy of our primary
        # data.  But that is OK, we won't muck with it from now on
        # (promise)

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
        y_bin = np.int((ybot-ytop)/n_y_steps)
        yrange = np.arange(ytop, ybot, y_bin)
        if yrange[-1] + y_bin > ybot:
            yrange = yrange[0:-1]
            # picturing the image in C fashion, indexed from the top down,
            # ypt_top is the top point from which we bin y_bin rows together

        for ypt_top in yrange:
            # We will be referencing the measured points to the center
            # of the bin
            ycent = ypt_top+y_bin/2

            if default_ND_params is None:
                # We have already made sure we are a flat at this
                # point, so just run with it.  Flats are high
                # contrast, low noise.  When we run this the first
                # time around, features are rounded and shifted by the
                # ND angle, but still detectable.

                # We can chop off the edges of the smaller SII
                # filters to prevent problems with detection of
                # edges of those filters
                bounds = small_filt_crop[:,1]
                profile = np.sum(im[ypt_top:ypt_top+y_bin,
                                    bounds[0]:bounds[1]],
                                 0)
                #plt.plot(bounds[0]+np.arange(bounds[1]-bounds[0]), profile)
                #plt.show()
                # Just doing d2 gets two peaks, so multiply
                # by the original profile to kill the inner peaks
                smoothed_profile \
                    = signal.savgol_filter(profile, x_filt_width, 3)
                d = np.gradient(smoothed_profile, 10)
                d2 = np.gradient(d, 10)
                s = np.abs(d2) * profile
            else:
                # Non-flat case.  We want to morph the image by
                # shifting each row by by the amount predicted by the
                # default_ND_params.  This lines the edges of the ND
                # filter up for easy spotting.  We will morph the
                # image directly into a subim of just the right size
                default_ND_width = (default_ND_params[1,1]
                                    - default_ND_params[1,0])
                subim_hw = int(default_ND_width/2 + search_margin)
                subim = np.empty((y_bin, 2*subim_hw))

                # rowpt is each row in the ypt_top y_bin, which we need to
                # shift to accumulate into a subim that is the morphed
                # image.
                for rowpt in np.arange(y_bin):
                    # determine how many columns we will shift each row by
                    # using the default_ND_params
                    thisy = rowpt+ypt_top
                    es, mid = self.ND_edges(thisy, default_ND_params, ND_ref_y)
                    this_ND_center = np.round(mid).astype(int)
                    left = max((0, this_ND_center-subim_hw))
                    right = min((this_ND_center+subim_hw,
                                 this_ND_center+subim.shape[1]-1))
                    #print('(left, right): ', (left, right))
                    subim[rowpt, :] \
                        = im[ypt_top+rowpt, left:right]
                
                profile = np.sum(subim, 0)
                # This spots the sharp edge of the filter surprisingly
                # well, though the resulting peaks are a little fat
                # (see signal.find_peaks_cwt arguments, below)
                smoothed_profile \
                    = signal.savgol_filter(profile, x_filt_width, 0)
                d = np.gradient(smoothed_profile, 10)
                s = np.abs(d)
                # To match the logic in the flat case, calculate
                # bounds of the subim picturing that it is floating
                # inside of the full image
                bounds = ND_ref_x + np.asarray((-subim_hw, subim_hw))
                bounds = bounds.astype(int)

            # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
            # points out same problem I had with with cwt.  It is too
            # sensitive to little peaks.  However, I can find the peaks
            # and just take the two largest ones
            #peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
            #peak_idx = signal.find_peaks_cwt(s, np.arange(2, 80), min_snr=2)
            peak_idx = signal.find_peaks_cwt(s,
                                             cwt_width_arange,
                                             min_snr=self.cwt_min_snr)
            # Need to change peak_idx into an array instead of a list for
            # indexing
            peak_idx = np.array(peak_idx)

            # Give up if we don't find two clear edges
            if peak_idx.size < 2:
                log.debug('No clear two peaks inside bounds ' + str(bounds))
                #plt.plot(s)
                #plt.show()
                continue

            if default_ND_params is None:
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
                if (de < max_ND_width_range[0]
                    or de > max_ND_width_range[1]):
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
                        diff_arr.append((ip,
                                         iop, peak_idx[iop] - peak_idx[ip]))
                diff_arr = np.asarray(diff_arr)
                closest = np.abs(diff_arr[:,2] - default_ND_width)
                sorted_idx = np.argsort(closest)
                edge_idx = peak_idx[diff_arr[sorted_idx[0], 0:2]]
                # Sanity check
                de = edge_idx[1] - edge_idx[0]
                if (de < max_ND_width_range[0]
                    or de > max_ND_width_range[1]):
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
            if default_ND_params is None:
                raise ValueError('Not able to find ND filter position')
            log.warning('Unable to improve filter position over initial guess')
            return self.default_ND_params
        
        ND_edges = np.asarray(ND_edges)
        ypts = np.asarray(ypts)
        
        if self.plot_ND_edges:
            plt.plot(ypts, ND_edges)
            plt.show()

        if default_ND_params is not None:
            # Unmorph our measured ND_edges so they are in the
            # reference frame of the original ref_ND_centers.  note,
            # they were measured in a subim with x origin subim_hw
            # away from the ref_ND_centers
            _, ref_ND_centers = self.ND_edges(ypts, default_ND_params, ND_ref_y)
            ref_ND_centers -= subim_hw
            for iy in np.arange(len(ypts)):
                ND_edges[iy, :] = ND_edges[iy, :] + ref_ND_centers[iy]
        if self.plot_ND_edges:
            plt.plot(ypts, ND_edges)
            plt.show()
            

        # Try an iterative approach to fitting lines to the ND_edges
        ND_edges = np.asarray(ND_edges)
        ND_params0 = iter_linfit(ypts-ND_ref_y, ND_edges[:,0],
                                 self.max_fit_delta_pix)
        ND_params1 = iter_linfit(ypts-ND_ref_y, ND_edges[:,1],
                                 self.max_fit_delta_pix)
        # Note when np.polyfit is given 2 vectors, the coefs
        # come out in columns, one per vector, as expected in C.
        ND_params = np.transpose(np.asarray((ND_params0, ND_params1)))
        
        # DEBUGGING
        #plt.plot(ypts, self.ND_edges(ypts, ND_params))
        #plt.show()

        # Calculate difference between bottom edges of filter in current FOV
        dp = abs((ND_params[0,1] - ND_params[0,0]) * im.shape[0]/2)
        if dp > self.max_parallel_delta_pix:
            txt = f'ND filter edges are not parallel.  Edges are off by {dp:.0f} pixels.'
            #print(txt)
            #plt.plot(ypts, ND_edges)
            #plt.show()
            
            if default_ND_params is None:
                raise ValueError(txt + '  No initial try available, raising error.')
            log.warning(txt + ' Returning initial try.')
            ND_params = default_ND_params

        return self.ND_params_unbinned(ND_params)

if __name__ == "__main__":
    log.setLevel('DEBUG')
    ccd = CorDataNDparams.read('/data/IoIO/raw/2017-03-14/Bias-0001_1x1.fit')
    print(f'background = {ccd.background}')
    ccd = CorDataNDparams.read('/data/IoIO/raw/2017-03-14/Bias-0001_2x2.fit')
    print(f'background = {ccd.background}')
    ccd = CorDataNDparams.read('/data/IoIO/raw/2017-03-14/Bias-0001_4x4.fit')
    print(f'background = {ccd.background}')

    print(f'ND_params = {ccd.ND_params}')

    ccd = CorDataNDparams.read('/data/IoIO/raw/2017-03-14/IPT-0001_off-band.fit')
    print(f'background = {ccd.background}')
    print(f'ND_params = {ccd.ND_params}')

    print(overscan_estimate(ccd, show=True))

    # Binned 2x2
    fname = '/data/IoIO/raw/2021-04_Astrometry/Main_Astrometry_East_of_Pier.fit'
    ccd = CorDataNDparams.read(fname)

    print(overscan_estimate(ccd, show=True))
    print(f'ND_params = {ccd.ND_params}')

