class CorData(CorDataNDparams):
    def __init__(self, data,
                 y_center_offset=0, # *Unbinned* Was 70 for a while See desired_center
                 show=False, # Show images
                 profile_peak_threshold=PROFILE_PEAK_THRESHOLD,
                 bright_sat_threshold=BRIGHT_SAT_THRESHOLD,
                 min_source_threshold=MIN_SOURCE_THRESHOLD,
                 copy=False,
                 **kwargs):
        # Pattern after NDData init but skip all the tests
        if isinstance(data, CorData):
            y_center_offset = data.y_center_offset
            show = data.show
            profile_peak_threshold = data.profile_peak_threshold
            bright_sat_threshold = data.bright_sat_threshold
            min_source_threshold = data.min_source_threshold
        if copy:
            y_center_offset = deepcopy(y_center_offset)
            show = deepcopy(show)
            profile_peak_threshold = deepcopy(profile_peak_threshold)
            bright_sat_threshold = deepcopy(bright_sat_threshold)
            min_source_threshold = deepcopy(min_source_threshold)

        super().__init__(data, copy=copy, **kwargs)
        # Define y pixel value along ND filter where we want our
        # center --> This may change if we are able to track ND filter
        # sag in Y.
        y_center_offset = data.y_center_offset
        self.show 			= show
        self.profile_peak_threshold = profile_peak_threshold
        self.bright_sat_threshold = bright_sat_threshold
        self.min_source_threshold = min_source_threshold

    def _init_args_copy(self, kwargs):
        kwargs = super()._init_args_copy(kwargs)
        kwargs['y_center_offset'] = self.y_center_offset
        kwargs['show'] = self.show
        kwargs['profile_peak_threshold'] = self.profile_peak_threshold
        kwargs['bright_sat_threshold'] = self.bright_sat_threshold
        kwargs['min_source_threshold'] = self.min_source_threshold
        return kwargs
        
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