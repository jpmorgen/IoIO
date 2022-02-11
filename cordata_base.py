class CorDataBase(FitsKeyArithmeticMixin, NoCenterPGD, MaxImPGD):
    def __init__(self, data,
                 default_ND_params=None,
                 ND_params=None, # for _slice
                 ND_ref_y=ND_REF_Y,
                 edge_mask=(5, -15), # Absolute coords, ragged on right edge.  If one value, assumed equal from *each* edge (e.g., (5, -5)
                 y_center_offset=0, # *Unbinned* Was 70 for a while See desired_center
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
                 profile_peak_threshold=PROFILE_PEAK_THRESHOLD,
                 bright_sat_threshold=BRIGHT_SAT_THRESHOLD,
                 min_source_threshold=MIN_SOURCE_THRESHOLD,
                 copy=False,
                 **kwargs):
        # Pattern after NDData init but skip all the tests
        if isinstance(data, CorDataBase):
            # Sigh.  We have to undo the convenience of our pgproperty
            # lest we trigger the calculation of property, which leads
            # to recursion problems
            obj_dict = data.__dict__
            ND_params = obj_dict.get('ND_params')
            ND_ref_y = data.ND_ref_y
            edge_mask = data.edge_mask
            y_center_offset = data.y_center_offset
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
            profile_peak_threshold = data.profile_peak_threshold
            bright_sat_threshold = data.bright_sat_threshold
            min_source_threshold = data.min_source_threshold
        if copy:
            ND_params = deepcopy(ND_params)
            ND_ref_y = deepcopy(ND_ref_y)
            edge_mask = deepcopy(edge_mask)
            y_center_offset = deepcopy(y_center_offset)
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
            profile_peak_threshold = deepcopy(profile_peak_threshold)
            bright_sat_threshold = deepcopy(bright_sat_threshold)
            min_source_threshold = deepcopy(min_source_threshold)

        super().__init__(data, copy=copy, **kwargs)
        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.
        if ND_params is not None:
            default_ND_params 		= ND_params
        self.default_ND_params 		= default_ND_params
        self.ND_params = ND_params
        self.ND_ref_y        		= ND_ref_y
        self.edge_mask              	= edge_mask
        self.y_center_offset        	= y_center_offset
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

        self.no_obj_center          = no_obj_center
        self.profile_peak_threshold = profile_peak_threshold
        self.bright_sat_threshold = bright_sat_threshold
        self.min_source_threshold = min_source_threshold

        self.arithmetic_keylist = ['satlevel', 'nonlin']
        self.handle_image = keyword_arithmetic_image_handler

    def _init_args_copy(self, kwargs):
        kwargs = super()._init_args_copy(kwargs)
        obj_dict = self.__dict__
        kwargs['ND_params'] = obj_dict.get('ND_params')
        kwargs['default_ND_params'] = self.default_ND_params
        kwargs['ND_ref_y'] = self.ND_ref_y
        kwargs['edge_mask'] = self.edge_mask
        kwargs['y_center_offset'] = self.y_center_offset
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
        kwargs['no_obj_center'] = self.no_obj_center
        kwargs['profile_peak_threshold'] = self.profile_peak_threshold
        kwargs['bright_sat_threshold'] = self.bright_sat_threshold
        kwargs['min_source_threshold'] = self.min_source_threshold
        return kwargs
        
    @pgproperty
    def default_ND_params(self):
        """Returns default ND_params and set Y reference point, self.ND_ref_y

        Values are queried from FITS header, with globals used as fall-back
        """
        
        # Define our array and default value all in one (a little
        # risky if FNPAR are partially broken in the FITS header)
        ND_params = np.asarray(RUN_LEVEL_DEFAULT_ND_PARAMS)
        # Code from flat correction part of cor_process to use the
        # master flat ND params as default
        for i in range(2):
            for j in range(2):
                fndpar = self.meta.get(f'fndpar{i}{j}')
                if fndpar is None:
                    break
                ND_params[j][i] = fndpar
        ND_ref_y = self.meta.get('ND_REF_Y')
        self.ND_ref_y = ND_ref_y or self.ND_ref_y
        return ND_params

    @pgproperty
    def imagetyp(self):
        imagetyp = self.meta.get('imagetyp')
        if imagetyp is None:
            return None
        return imagetyp.lower()

    @pgcoordproperty
    def edge_mask(self):
        """Pixels on *inside* edge of ND filter to remove when calculating
        ND_coords.  Use a negative value to return coordinates
        extending beyond the ND filter, e.g. for masking.  Stored as a
        tuple (left, right), thus if an asymmetric value is desired,
        right should the negative of left.  If just one value is
        provided, the setter automatically negates it for the right
        hand value"""

    @edge_mask.setter
    def edge_mask(self, edge_mask):
        edge_mask = np.asarray(edge_mask)
        if edge_mask.size == 1:
            edge_mask = np.append(edge_mask, -edge_mask)
        return edge_mask        

    def ND_params_unbinned(self, ND_params_in):
        ND_params = ND_params_in.copy()
        ND_params[1, :] = self.x_unbinned(ND_params[1, :])
        return ND_params

    def ND_params_binned(self, ND_params_in):
        ND_params = ND_params_in.copy()
        ND_params[1, :] = self.x_binned(ND_params[1, :])
        return ND_params

    def ND_edges(self, y, ND_params, ND_ref_y):
        """Returns x coords of ND filter edges and center at given y(s)

        Parameters
        ---------
        y : int or numpy.ndarray
            Input y values referenced to the current self.data view
            (e.g. may be binned and subframed)

        ND_params : numpy.ndarray
            Proper ND_params for self.data view

        ND_ref_y : int
            Proper Y reference point for self.data view

        Returns
        -------
        edges, midpoint : tuple
            Edges is a 2-element numpy.ndarray float, 
            midpoint is a numpy scalar float

        """
        #print(f'in ND_edges, ND_params = {ND_params[1,:]}')
        if np.isscalar(y):
            es = ND_params[1,:] + ND_params[0,:]*(y - ND_ref_y)
            mid = np.mean(es)
            return es, mid
        es = []
        mid = []
        for ty in y:
            tes = ND_params[1,:] + ND_params[0,:]*(ty - ND_ref_y)
            mid.append(np.mean(tes))
            es.append(tes)            
        return np.asarray(es), np.asarray(mid)
                         
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
        # Biaes and darks don't have signal to spot ND filter and if
        # we don't want an obj_center, it is probably because the
        # image doesn't contain a bright enough central object to spot
        # the ND filter.  In that case, self.default_ND_params should
        # be close enough
        if (self.imagetyp == 'bias'
            or self.imagetyp == 'dark'):
            return self.default_ND_params
            #or self. no_obj_center):

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

    @pgproperty
    def ND_coords(self):
        """Returns tuple of coordinates of ND filter referenced to the potentially binned and subframed image and including the edge mask.  Change the edge-maks property, set this to None and it will recalculate the next time you ask for it"""
        xs = np.asarray((), dtype=int) ; ys = np.asarray((), dtype=int)
        ND_params = self.ND_params_binned(self.ND_params)
        ND_ref_y = self.y_binned(self.ND_ref_y)
        edge_mask = self.edge_mask/self.binning[1]
        for iy in np.arange(self.shape[0]):
            ND_es, _ = self.ND_edges(iy, ND_params, ND_ref_y)
            bounds = ND_es + edge_mask
            if (np.all(bounds < 0)
                or np.all(bounds > self.shape[1])):
                continue
            bounds = bounds.astype(int)
            bounds[0] = np.max((0, bounds[0]))
            bounds[1] = np.min((bounds[1], self.shape[1]))
            more_xs = np.arange(bounds[0], bounds[1])
            xs = np.append(xs, more_xs)
            ys = np.append(ys, np.full_like(more_xs, iy))
        ND_coords = np.asarray((ys, xs))
        array_of_tuples = map(tuple, ND_coords)
        tuple_of_tuples = tuple(array_of_tuples)
        return tuple_of_tuples

    def ND_coords_above(self, level):
        """Returns tuple of coordinates of pixels with decent signal in ND filter"""
        # Get the coordinates of the ND filter
        NDc = self.ND_coords
        im = self.data
        abovec = np.where(im[NDc] > level)
        if abovec is None:
            return None
        above_NDc0 = np.asarray(NDc[0])[abovec]
        above_NDc1 = np.asarray(NDc[1])[abovec]
        return (above_NDc0, above_NDc1)
    
        #print(f'abovec = {abovec}')
        ##return(abovec)
        ## Unwrap
        #abovec = NDc[abovec]
        #print(f'abovec = {abovec}')
        #return abovec

    # Turn ND_angle into a "getter"
    @pgproperty
    def ND_angle(self):
        """Calculate ND angle from vertical.  Note this assumes square pixels
        """
        ND_angle = np.degrees(np.arctan(np.average(self.ND_params[0,:])))
        return ND_angle

    @pgproperty
    def background(self):
        """This might eventually get better, but for now just returns overscan"""
        return overscan_estimate(self)

    def get_metadata(self):
        # Add our camera metadata.  Note, because there are many ways
        # this object is instantiated (e.g. during arithmetic), makes
        # sure we only do the FITS header manipulations when we are
        # reasonably sure we have our camera.
        instrument = self.meta.get('instrume')
        if (instrument == 'SX-H694'
            or instrument == 'IoIO Coronagraph'):
            self.meta = sx694.metadata(self.meta)
        # Other cameras would be added here...

    @property
    def readnoise(self):
        """Returns CCD readnoise as value in same unit as primary array"""
        self.get_metadata()
        # --> This gets better with FITS header units
        readnoise = self.meta.get('RDNOISE')
        if self.unit == u.adu:
            gain = self.meta.get('GAIN')
            readnoise /= gain
        return readnoise
        
    @property
    def nonlin(self):
        """Returns nonlinearity level in same unit as primary array"""
        self.get_metadata()
        return self.meta.get('NONLIN')# * self.unit
        
    @pgcoordproperty
    def obj_center(self):
        """Returns center pixel coords of Jupiter whether or not Jupiter is on ND filter.  Unbinned pixel coords are returned.  Use [Cor]ObsData.binned() to convert to binned pixels.
        """

        if self.show:
            simple_show(self, norm=LogNorm())
        # Check to see if we really want to calculate the center
        imagetyp = self.meta.get('IMAGETYP')
        if imagetyp.lower() in ['bias', 'dark', 'flat']:
            self.no_obj_center = True
        if self.no_obj_center:
            return NoCenterPGD(self).obj_center           


        # NOTE: for maximum efficiency, calculate our ND_params up
        # front, since we are going to be copying them on to other objects
        
        # --> HACK We have to calculate ND_params in self before we
        # propagate them on to copies, slices, etc.
        self.ND_params


        # --> Work with an unbinned version of ourselves to handle
        # specifics of flux (this may change to work always in binned)
        # Make sure we copy since we are going to muck with it and
        # self_unbinned does not return a copy of unbinned data.  Make
        # sure it is not raw int and flux normalize it
        ccd = self.self_unbinned.copy()
        
        ccd = ccd.divide(np.prod(self.binning)*u.dimensionless_unscaled,
                         handle_meta='first_found')
        #im = np.double(self.self_unbinned.data.copy()) / (np.prod(self.binning))
        back_level = self.background / (np.prod(self.binning))

        # Establish some metrics to see if Jupiter is on or off the ND
        # filter.  Easiest one is number of saturated pixels
        # /data/IoIO/raw/2018-01-28/R-band_off_ND_filter.fit gives
        # 4090 of these.  Calculation below suggests 1000 should be a
        # good minimum number of saturated pixels (assuming no
        # additional scattered light).  A star off the ND filter
        # /data/IoIO/raw/2017-05-28/Sky_Flat-0001_SII_on-band.fit
        # gives 124 num_sat
        log.debug(f'back_level = {back_level}, nonlin: {ccd.nonlin}, max im: {np.max(ccd)}')
        num_sat = (ccd.data >= ccd.nonlin).sum()

        # Make a 1D profile along the ND filter to search for a source there
        us = ccd.shape
        ND_profile = np.empty(us[0])
        for iy in np.arange(us[0]):
            es, _ = ccd.ND_edges(iy, self.ND_params, self.ND_ref_y)
            es = es.astype(int)
            row = ccd.data[iy, es[0]:es[1]]
            # Get rid of cosmic rays --> Eventually just leave binned
            row = signal.medfilt(row, 3)
            ND_profile[iy] = np.mean(row)

        diffs2 = (ND_profile[1:] - ND_profile[0:-1])**2
        profile_variance = np.sqrt(np.median(diffs2))

        ND_width = (ccd.ND_params[1,1]
                    - ccd.ND_params[1,0])
        prof_peak_idx = signal.find_peaks_cwt(ND_profile,
                                              np.linspace(4, ND_width))
        ymax_idx = np.argmax(ND_profile[prof_peak_idx])
        # unwrap
        ymax_idx = prof_peak_idx[ymax_idx]
        ymax = ND_profile[ymax_idx]
        med = np.median(ND_profile)
        std = np.std(ND_profile)
        peak_contrast = (ymax - med)/profile_variance
        log.debug(f'profile peak_contrast = {peak_contrast}, threshold = {self.profile_peak_threshold}, peak y = {ymax_idx}')

        #plt.plot(ND_profile)
        #plt.show()

        source_on_ND_filter = (peak_contrast > self.profile_peak_threshold)

        # Work another way to see if the ND filter has a low flux
        ccd  = ccd.subtract(back_level*ccd.unit, handle_meta='first_found')
        nonlin = ccd.nonlin

        # Come up with a metric for when Jupiter is off the ND filter.
        # Below is my scratch work        
        # Rj = np.asarray((50.1, 29.8))/2. # arcsec
        # plate = 1.59/2 # main "/pix
        # 
        # Rj/plate # Jupiter pixel radius
        # array([ 31.50943396,  18.74213836])
        # 
        # np.pi * (Rj/plate)**2 # Jupiter area in pix**2
        # array([ 3119.11276312,  1103.54018437])
        #
        # Jupiter is generally better than 1000
        # 
        # np.pi * (Rj/plate)**2 * 1000 
        # array([ 3119112.76311733,  1103540.18436529])

        log.debug(f'Number of saturated pixels in image = {num_sat}; bright source threshold = {self.bright_sat_threshold}, minimum source {self.min_source_threshold}')

        bright_source_off_ND = num_sat > self.bright_sat_threshold
        if bright_source_off_ND or not source_on_ND_filter:
            # Outside the ND filter, Jupiter should be saturating.  To
            # make the center of mass calc more accurate, just set
            # everything that is not getting toward saturation to 0
            # --> Might want to fine-tune or remove this so bright
            ccd.data[ccd.data < nonlin*0.7] = 0
            
            # Catch Jupiter at it's minimum 
            # --> This logic doesn't work well to rule out case where
            # there are many bright stars, but to do that would
            # require a lot of extra work like segmentation, which is
            # not worth it for this object.  Looking at stars and
            # getting a good center is the job of PGAstromData or
            # whatever.  If I really wanted to rule out these cases
            # without this effort, I would set min_source_threshold to
            # 1000 or change the logic above
            sum_bright_pixels = np.sum(ccd.data)
            if sum_bright_pixels < nonlin * self.min_source_threshold:
                log.debug(f'No bright source found: number of bright pixels ~ {sum_bright_pixels/nonlin} < {self.min_source_threshold} self.min_source_threshold')
                return NoCenterPGD(self).obj_center           

            # If we made it here, Jupiter is outside the ND filter,
            # but shining bright enough to be found
            if bright_source_off_ND and source_on_ND_filter:
                # Both on and off - prefer off
                log.debug('Bright source near ND filter')
            elif not source_on_ND_filter:
                log.debug('No source on ND filter')
            else:
                log.debug('Bright source off of ND filter')
            
            # Use iterative approach
            ny, nx = ccd.shape
            y_x = np.asarray(center_of_mass(ccd.data))
            log.debug(f'First iteration COM (X, Y; binned) = {self.coord_binned(y_x)[::-1]}')
            y = np.arange(ny) - y_x[0]
            x = np.arange(nx) - y_x[1]
            # input/output Cartesian direction by default
            xx, yy = np.meshgrid(x, y)
            rr = np.sqrt(xx**2 + yy**2)
            # --> Make this property of object
            ccd.data[rr > 200] = 0
            y_x = np.asarray(center_of_mass(ccd.data))
            log.debug(f'Second iteration COM (X, Y; binned) = {self.coord_binned(y_x)[::-1]}')
            self.quality = 6
            return y_x

        # If we made it here, we are reasonably sure Jupiter or a
        # suitably bright source is on the ND filter
        log.debug(f'Bright source on ND filter')

        # Use the peak on the ND filter to extract a patch over
        # which we calculate the COM
        _, xmax = ccd.ND_edges(ymax_idx, ccd.ND_params, ccd.ND_ref_y)
        iy_x = np.asarray((ymax_idx, xmax)).astype(int)
        patch_half_width = ND_width
        patch_half_width = patch_half_width.astype(int)
        ll = iy_x - patch_half_width
        ur = iy_x + patch_half_width
        patch = ccd[ll[0]:ur[0], ll[1]:ur[1]]

        if self.show:
            simple_show(patch, norm=LogNorm())

        boost_factor = ccd.nonlin*1000

        # Zero out all pixels that (1) are not on the ND filter and
        # (2) do not have decent signal
        NDmed = np.median(patch.data[patch.ND_coords])
        print(f'self.ND_params {self.ND_params}')
        print(f'patch ND_params = {patch.ND_params}')
        #print(f'patch ND_coords = {patch.ND_coords}')
        NDstd = np.std(patch.data[patch.ND_coords])
        log.debug(f'ND median, std {NDmed}, {NDstd}, 6*self.readnoise= {6*self.readnoise}')
        boost_ND_coords = patch.ND_coords_above(NDmed + 6*self.readnoise)
        patch.data[boost_ND_coords] *= boost_factor
        patch.data[patch.data < boost_factor] = 0
        patch = patch.divide(boost_factor*u.dimensionless_unscaled,
                             handle_meta='first_found')

        if self.show:
            simple_show(patch, norm=LogNorm())

        # Check for the case when Jupiter is near the edge of the ND
        # filter.  Optical effects result in bright pixels on the ND
        # filter that confuse the COM.
        bad_ND_coords = patch.ND_coords_above(nonlin + back_level)
        nbad = len(bad_ND_coords[0])
        bright_on_ND_threshold = 50
        log.debug(f'Number of bright pixels on ND filter = {nbad}; threshold = {bright_on_ND_threshold}')
        if nbad > bright_on_ND_threshold: 
            # As per calculations above, this is ~5% of Jupiter's area
            # Experiments with
            # /data/IoIO/raw/2021-04_Astrometry/Jupiter_near_ND_edge_S7.fit
            # and friends show there is an edge of ~5 pixels around
            # the plateau.  Use gaussian filter to fuzz out our saturated pixels
            log.debug(f'Setting excessive bright pixels to ND median value')
            bad_patch = np.zeros_like(patch.data)
            print(f'nonlin = {nonlin}')
            bad_patch[bad_ND_coords] = nonlin
            if self.show:
                simple_show(bad_patch)
            bad_patch = gaussian_filter(bad_patch, sigma=5)
            if self.show:
                simple_show(bad_patch)
            patch.data[bad_patch > 0.1*np.max(bad_patch)] = NDmed
            if self.show:
                simple_show(patch, norm=LogNorm())
            self.quality = 6
        else:
            log.debug(f'ND filter is clean')
            self.quality = 8
        pcenter = np.asarray(center_of_mass(patch.data))
        y_x = pcenter + ll
        log.debug(f'Object COM from clean patch (X, Y; binned) = {self.coord_binned(y_x)[::-1]}, quality = {self.quality}')
        #return y_x

        # --> Experiment with a really big hammer.  Does offer some improvement
        photometry = Photometry(ccd=patch)
        if self.show:
            photometry.show_source_mask()
            photometry.show_background()
            photometry.show_segm()
        sc = photometry.source_catalog
        if sc is None:
            log.warning('No sources found in patch.  Bright conditions?  Consider using OffCorData')
            return NoCenterPGD(self).obj_center            
            
        tbl = sc.to_table()
        tbl.sort('segment_flux', reverse=True)
        tbl.show_in_browser()
        xpcentrd = tbl['xcentroid'][0]
        ypcentrd = tbl['ycentroid'][0]
        #print(ll[::-1])
        #print(pcenter[::-1])
        #print(xpcentrd, ypcentrd)
        photometry_y_x = np.asarray((ypcentrd, xpcentrd)) + ll
        log.debug(f'Patch COM = {self.coord_binned(y_x)[::-1]} (X, Y; binned); Photometry brightest centroid {self.coord_binned(photometry_y_x)[::-1]}; quality = {self.quality}')        
        return photometry_y_x

        ## First iteration COM
        #pcenter = np.asarray(center_of_mass(patch))
        #y_x = pcenter + ll
        #log.debug(f'First iteration COM (X, Y; binned) = {self.coord_binned(y_x)[::-1]}')        
        #
        ##y_x = np.asarray(center_of_mass(im))
        ##log.debug(f'First iteration COM (X, Y; binned) = {self.coord_binned(y_x)[::-1]}')        
        ### Work with a patch of our (background-subtracted) image
        ### centered on the first iteration COM.  
        ##ND_width = self.ND_params[1,1] - self.ND_params[1,0]
        ##patch_half_width = ND_width / 2
        ##patch_half_width = patch_half_width.astype(int)
        ##iy_x = y_x.astype(int)
        ##ll = iy_x - patch_half_width
        ##ur = iy_x + patch_half_width
        ##patch = im[ll[0]:ur[0], ll[1]:ur[1]]
        ##patch /= boost_factor
        #
        #log.debug(f'First iteration cpatch (X, Y) = {cpatch[::-1]}')        
        #plt.imshow(patch)
        #plt.show()
        #
        #
        #
        ### --> Experiment with a really big hammer.  Wasn't any
        ### slower, but didn't give an different answer
        ##patch = CorData(patch, meta=self.meta)
        ##photometry = Photometry(ccd=patch)
        ##sc = photometry.source_catalog
        ##tbl = sc.to_table()
        ##tbl.sort('segment_flux', reverse=True)
        ##tbl.show_in_browser()


    @pgproperty
    def obj_to_ND(self):
        """Returns perpendicular distance of obj center to center of ND filter in binned coordinates
        """
        if self.quality == 0:
            # This quantity doesn't make sense if there is an invalid center
            return None
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # has a better factor
        imshape = self.coord_unbinned(self.shape)
        ND_params = self.ND_params_binned(self.ND_params)
        ND_ref_y = self.y_binned(self.ND_ref_y)
        m = np.average(ND_params[0,:])
        b = np.average(ND_params[1,:])
        # Random Xs (really Y, as per below) with which to calculate our line
        x1 = 1100; x2 = 1200
        # The line is actually going vertically, so X in is the C
        # convention of along a column.  Also remember our X coordinate
        # is relative to the center of the image
        y1 = m * (x1 - ND_ref_y)  + b
        y2 = m * (x2 - ND_ref_y)  + b
        x0 = self.obj_center[0]
        y0 = self.obj_center[1]
        d = (np.abs((x2 - x1) * (y1 - y0)
                    - (x1 - x0) * (y2 - y1))
             / ((x2 - x1)**2 + (y2 - y1)**2)**0.5)
        return d

    @pgcoordproperty
    def desired_center(self):
        """Returns Y, X center of ND filter at Y position determined by
        self.y_center_offset.  

        """
        # in 2019, moved this down a little to get around a piece of
        # dust and what looks like a fold in the ND filter unbinned,
        # 70 pixels down in MaxIm was about the right amount (yes,
        # MaxIm 0,0 is at the top left and Python goes Y, X.  And yes,
        # we need to work in unbinned coords, since hot pixels and the
        # ND filter are referenced to the actual CCD
        offset = np.asarray((self.y_center_offset, 0))
        unbinned_desired_center = (self.coord_unbinned(super().desired_center)
                                   + offset)
        y_center = unbinned_desired_center[0]
        x_center = np.average(self.ND_edges(y_center, self.ND_params))
        desired_center = np.asarray((y_center, x_center))
        # Check to make sure desired center is close to the center of the image
        ims = np.asarray(self.shape)
        bdc = self.coord_binned(desired_center)
        low = bdc < ims*0.25
        high = bdc > ims*0.75
        if np.any(np.asarray((low, high))):
            raise ValueError('Desired center is too far from center of image.  In original image coordinates:' + str(self.coord_binned(desired_center)))
        return desired_center

    def _card_write(self):
        """Write FITS card unique to CorData"""
        # Priorities ND_params as first-written, since they, together
        # with the boundaries set up by the flats (which initialize
        # the ND_params) provide critical context for all other views
        # of the data
        self.meta['NDPAR00'] = (self.ND_params[0,0],
                                'ND filt left side slope')
        self.meta['NDPAR01'] = (self.ND_params[1,0],
                                'Full frame X dist of ND filt left side at ND_REF_Y')
        self.meta['NDPAR10'] = (self.ND_params[0,1],
                                'ND filt right side slope')
        self.meta['NDPAR11'] = (self.ND_params[1,1],
                                'Full frame X dist of ND filt right side at ND_REF_Y')
        self.meta['ND_REF_Y'] = (self.ND_ref_y,
                                'Full-frame Y reference point of ND_params')
        super()._card_write()
        if self.quality > 5:
            self.meta['HIERARCH OBJ_TO_ND_CENTER'] \
                = (self.obj_to_ND,
                   'Obj perp dist to ND filt (pix)')
