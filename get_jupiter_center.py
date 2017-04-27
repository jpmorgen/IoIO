#!/usr/bin/python3

# get_jupiter-center.py

# Debugging
import matplotlib.pyplot as plt
from jpm_fns import display

from astropy import log
import numpy as np
from astropy.io import fits
from scipy import signal, ndimage, misc

# ND filter width in pixels (comes in right at 1.5 mm =~ 3 Rj at opposition)
ndfilter_width = 100
# Number of steps to move the model ND filter across an image
nd_num_steps = 20

#def rot_ang(im):
#    """Find rotation angle of flat image"""
#    for ang in 
#    tim = ndimage.interpolation.rotate(im, ang)

def nd_filt_pos(im, ny=15, x_filt=25, initial_try=None, max_movement=50, max_delta_pix=10):
    """Find the position of an ND filter"""

    nd_edges = [] ; ypts = []
    y_bin = int(im.shape[0]/ny)
    yrange = np.arange(0, im.shape[0], y_bin)
    for ypt in yrange:
        subim = im[ypt:ypt+y_bin, :]
        profile = np.sum(subim, 0)
        smoothed_profile = signal.savgol_filter(profile, x_filt, 3)
        d = np.gradient(smoothed_profile, 3)
        s = np.abs(d)

        # https://blog.ytotech.com/2015/11/01/findpeaks-in-python/
        # points out same problem I had with with cwt.  It is too
        # sensitive to little peaks.  However, I can find the peaks
        # and just take the two largest ones
        peak_idx = signal.find_peaks_cwt(s, np.arange(5, 20), min_snr=2)
        # Need to change peak_idx into an array instead of a list for
        # indexing
        peak_idx = np.array(peak_idx)
        # If available, limit our search to the region max_movement
        # around initial_try.
        bounds = (0,s.size)
        if initial_try != None:
            bounds = initial_try[1,:] + initial_try[0,:]*(ypt - im.shape[0]/2) + np.asarray((-max_movement, max_movement))
            bounds = bounds.astype(int)
            goodc = np.where(np.logical_and(bounds[0] < peak_idx, peak_idx < bounds[1]))
            peak_idx = peak_idx[goodc]
            #print(peak_idx)
            #print(s[peak_idx])
            #plt.plot(s)
            #plt.show()
            if peak_idx.size < 2:
                continue

        # Sort on value
        sorted_idx = np.argsort(s[peak_idx])
        # Unwrap
        peak_idx = peak_idx[sorted_idx]

        # Thow out if lower peak is too weak.  Use Carey Woodward's
        # trick of estimating the noise on the continuum To avoid
        # contamination, do this calc just over our desired interval
        ss = s[bounds[0]:bounds[1]]

        noise = np.std(ss[1:-1] - ss[0:-2])
        #print(noise)
        if s[peak_idx[-2]] < 3 * noise:
            #print("Rejected")
            continue
        #
        ##fs = signal.medfilt(ss, 101) + np.median(ss)
        ##chop_idx = np.where(ss - fs > 0)
        #chop_idx = np.where(ss > (3 * np.std(ss)))
        #ss[chop_idx] = 0
        #plt.plot(ss)
        #plt.show()
        ##print(np.median(fs), np.std(fs))
        #print(np.median(ss), np.std(ss))
        #if s[peak_idx[-2]] < 3 * np.std(ss):
        #    print("Rejected")
        #    continue

        # Find top two and put back in index order
        top_two = np.sort(peak_idx[-2:])
        # Accumulate in tuples
        nd_edges.append(top_two)
        ypts.append(ypt)

    nd_edges = np.asarray(nd_edges)
    ypts = np.asarray(ypts)
    if nd_edges.size < 2:
        if initial_try == None:
            raise ValueError('Not able to find ND filter position')
        log.warning('Unable to improve filter position over initial guess')
        return(initial_try)
    
    plt.plot(ypts, nd_edges)
    plt.show()

    # Fit lines to our points, making the origin the center of the image in Y
    params = np.polyfit(ypts-im.shape[0]/2, nd_edges, 1)
    # Check parallelism by calculating shift of ends relative to each other
    dp = abs((params[0,1] - params[0,0]) * im.shape[0]/2)
    if dp > max_delta_pix:
        txt = 'ND filter edges are not parallel.  Edges are off by ' + str(dp) + ' pixels.'
        if initial_try == None:
            raise ValueError(txt)
        log.warning(txt + ' Returning initial try.')
        return(initial_try)
    
    #print(params)
    return(params)

        #d = profile[1:-1] - profile[0:-2]
        #d = misc.derivative(profile)


def nd_coords(im, params, edge_mask=5):
    """Returns coordinates of ND filter in im given parameters of fit from nd_filt_pos"""

    xs = [] ; ys = []
    for iy in np.arange(0, im.shape[0]):
        bounds = params[1,:] + params[0,:]*(iy - im.shape[0]/2) + np.asarray((edge_mask, -edge_mask))
        bounds = bounds.astype(int)
        for ix in np.arange(bounds[0], bounds[1]):
            xs.append(ix)
            ys.append(iy)

    # NOTE C order!
    return((ys, xs))

def hist_of_im(im):
    """Returns histogram of image and index into centers of bins"""
    
    # Code from west_aux.py, maskgen.

    # Histogram bin size should be related to readnoise
    readnoise = 5
    hrange = (im.min(), im.max())
    nbins = int((hrange[1] - hrange[0]) / readnoise)
    hist, edges = np.histogram(im, bins=nbins,
                               range=hrange, density=True)
    # Convert edges of histogram bins to centers
    centers = (edges[0:-1] + edges[1:])/2
    #plt.plot(centers, hist)
    #plt.show()

    return(hist, centers)

    
#def get_jupiter_center(fname, flat='/data/io/IoIO/raw/2017-04-20/Sky_Flat-0005_SII_off-band.fit'):
def get_jupiter_center(fname, flat='/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit'):
    """Return the center of an image, correcting for the ND filter, if Jupiter is behind it"""


    # Read our image
    H = fits.open(fname)
    im = np.asfarray(H[0].data)
    H.close()

    # Prepare to zero saturated pixels, since they confuse the region
    # around the ND filter if Jupiter or its scattered light is poking
    # out.  But don't zero them out until after we bias subtract
    satc = np.where(im > 60000)

    # Use the histogram technique to spot the bias level of the image.
    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.  The ND filter provides a
    # similar peak after bias subutraction (or, rather, it is the
    # second such peak)
    im_hist, im_hist_centers = hist_of_im(im)
    im_peak_idx = signal.find_peaks_cwt(im_hist, np.arange(10, 50))
    im -= im_hist_centers[im_peak_idx[0]]

    # Check to see if Jupiter is sticking out significantly from
    # behind the ND filter, in which case we are better off just using
    # the center of mass of the image and calling that good enough
    print(np.sum(im))
    if np.sum(im) > 1E9: 
        y_x = ndimage.measurements.center_of_mass(im)
        return(y_x[::-1])

    # If we made it here, Jupiter is at least mostly behind the ND filter.  

    # --> Note that this may not be general for the flat and that I
    # --> have seen the ND filter on the flats show up in a different
    # --> place than on the images through the night.  Consider not
    # --> passing a flat image, but NDc that I have carefully prepared by hand
    H = fits.open(flat)
    flat = np.asfarray(H[0].data)
    H.close()
    # Match shape of flat, which is full-frame, to image, which may not be
    flat = flat[0:im.shape[0], 0:im.shape[1]]
    
    # Find edges of ND filter in flat.  This is pretty much sure to
    # succeed without any additional help
    flat_nd_pos = nd_filt_pos(flat)

    # Use that to boot-strap finding accurate edges of ND filter in image
    im_nd_pos =  nd_filt_pos(im, initial_try=flat_nd_pos)
    # Boost the ND filter by 1000
    oim = im.copy()
    NDc = nd_coords(im, im_nd_pos)
    boostc = np.where(im[NDc] > (np.median(im[NDc]) + np.std(im[NDc])))
    boost_NDc0 = np.asarray(NDc[0])[boostc]
    boost_NDc1 = np.asarray(NDc[1])[boostc]
    im[boost_NDc0, boost_NDc1] *= 1000
    y_x = ndimage.measurements.center_of_mass(im)
    print(y_x[::-1])
    negc = np.where(im < 0)
    im[negc] = 0
    y_x = ndimage.measurements.center_of_mass(im)
    print(y_x[::-1])
    plt.imshow(im)
    plt.show()
    return(y_x[::-1])



    #####################
    # Make a histogram and find the peaks for spotting the bias and ND
    # filter in the flat image
    flat_hist, flat_hist_centers = hist_of_im(flat)
    flat_peak_idx = signal.find_peaks_cwt(flat_hist, np.arange(10, 50), min_snr=2)
    ND_left_idx = np.argmin(flat_hist[flat_peak_idx[0]:flat_peak_idx[1]])
    # The peaks past the bias and ND filter tend to be tiny.  Use one
    # of those to define the right side of the ND filter histogram peak
    ND_right_idx = flat_peak_idx[2]
        
    # Coordinates in flat of ND filter
    NDc = np.where(np.logical_and(flat_hist_centers[ND_left_idx] < flat, flat < flat_hist_centers[ND_right_idx]))

    # ND filter in image may be shifted.  It's a little hard to
    # determine if it is shifted up-down, so just stick with
    # left-right.  In cases where the filter is tipped to match
    # Jupiter's rotation axis, go plenty far in either direction to
    # make sure we pass over the filter
    nd_steps = np.arange(-int(ndfilter_width), int(ndfilter_width+1), int(ndfilter_width*2/nd_num_steps))

    # Move a blanked-out version of the flat's ND filter across the
    # image.  When they line up, there will be a minimum in the total
    # image count
    # Now zero the saturated pixels in the image
    # --> Do we need to do this anymore?
    im[satc] = 0
    masses = []
    for dx in nd_steps:
        # Shift X (second coord) and make sure it is in bounds
        sNDc1 = NDc[1] + dx
        goodc = np.where(np.logical_and(sNDc1 >= 0, sNDc1 < im.shape[1]))
        blank_im = im.copy()    
        blank_im[NDc[0][goodc], sNDc1[goodc]] = 0
        masses.append(np.sum(blank_im))

        #plt.imshow(blank_im)
        #plt.show()

    plt.plot(nd_steps, masses)
    plt.show()
    return(0)
                      


    
    # Correct for ND filter density, but avoid inducing noise with
    # negative values by boosting only points 1 std above median
    boostc = np.where(im[NDc] > (np.median(im[NDc]) + np.std(im[NDc])))
    
    # Because the ND filter moves during the night, it doesn't work to
    # just boost the flat's version of the ND filter region and get an
    # accurate center of mass of Jupiter, at least when Jupiter is
    # near the edge of the ND filter.  First get a measurement of the
    # X and Y locations of Jupiter's scattered light by moving the ND
    # filter from one extreme to the other.  Note, X is the second
    # coordinate here.  --> Note that this doesn't handle an ND filter
    # perfectly aligned on X or Y (which we don't actually want for
    # other reasons -- rather line up to Jupiter's axis)
    nd_steps = np.arange(-int(ndfilter_width/2), int(ndfilter_width/2+1), int(ndfilter_width/nd_num_steps))
    print(nd_steps)
    mass_im = np.zeros((nd_steps.size, nd_steps.size, 2))
    #mass_im = np.zeros((nd_steps.size, nd_steps.size))
    #mass_im2 = np.zeros((nd_steps.size, nd_steps.size))
    iy = 0
    for dy in nd_steps:
        ix = 0
        for dx in nd_steps:
            dNDc = (NDc[0]+dy, NDc[1]+dx)
            # --> Need to limit indices here
            boost_NDc0 = dNDc[0][boostc]
            boost_NDc1 = dNDc[1][boostc]
            # In the Y direction, the ND filter can move off the edge
            # of the FOV, causing an array out of bounds error.  Catch
            # and just skip this point if that happens.
            boost_im = im.copy()
            blank_im = im.copy()
            try:
                boost_im[boost_NDc0, boost_NDc1] *= 1000.
                blank_im[dNDc] = 0
            except Exception as e:
                print(str(e) + ' happened for ', ix, '  ', iy)
                continue
            #y_x = ndimage.measurements.center_of_mass(boost_im)
            #dxs.append(dx)
            #dys.append(dy)
            #mass_im[iy, ix] = np.sum(boost_im)
            #mass_im2[iy, ix] = np.sum(blank_im)
            mass_im[iy, ix, 0] = np.sum(boost_im)
            mass_im[iy, ix, 1] = np.sum(blank_im)
            #print(dx, y_x[::-1], masses[-1])
            #print(dx, dy, mass_im[iy, ix])
            ix += 1
        iy += 1

        #plt.imshow(boost_im)
        #plt.show()
        #display(boost_im, scale=None)

    # See where the max in our masses is.
    #plt.imshow(mass_im)
    #plt.show()
    #plt.imshow(mass_im2)
    #plt.show()
    plt.imshow(mass_im[:,:,0])
    plt.show()
    plt.imshow(mass_im[:,:,1])
    plt.show()
    #display(im, scale=None)

    # Inspect our scattered light measurement to see if we are too far
    # off to one side.  If so, we are better off using this
    # measurement.
    # y_x = ndimage.measurements.center_of_mass(mass_im)
    # y_x = np.asarray(y_x)
    y_x = np.unravel_index(np.argmax(-mass_im[:, :]), mass_im.shape)
    y_x = np.asarray(y_x)
    print(y_x[::-1])
    print(y_x[::-1] - nd_steps.size/2)
    print(mass_im[:, :].min(), mass_im[:, :].max())
    #
    #y_x = np.unravel_index(np.argmax(mass_im[:, :, 0]), mass_im.shape[0:2])
    #y_x = np.asarray(y_x)
    #print(y_x[::-1])
    #print(y_x[::-1] - nd_steps.size/2)
    #print(mass_im[:, :, 0].max())
    #
    #y_x = np.unravel_index(np.argmax(mass_im[:, :, 1]), mass_im.shape[0:2])
    #y_x = np.asarray(y_x)
    #print(y_x[::-1])
    #print(y_x[::-1] - nd_steps.size/2)
    #print(mass_im[:, :, 1].max())

    
    return((y_x[::-1] - nd_steps.size/2) * nd_num_steps)

    if np.linalg.norm(y_x - nd_steps.size/2) > ndfilter_width/nd_num_steps*0.3:
        pass
        
    # flat_peak_idx = signal.find_peaks_cwt(flat_hist, np.arange(10, 50), min_snr=2)    
    # plt.plot(dxes, masses)
    # plt.show()
    # y_x = ndimage.measurements.center_of_mass(im)
    return(y_x[::-1])


# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# 
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))


#H = fits.open('/data/io/IoIO/raw/2017-04-20/Sky_Flat-0007_Na_off-band.fit')
#flat = np.asfarray(H[0].data)
#H.close()
#ND_filt_pos(flat)

#H = fits.open('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit')
#flat = np.asfarray(H[0].data)
#H.close()
#ND_filt_pos(flat)
