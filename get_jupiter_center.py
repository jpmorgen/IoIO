#!/usr/bin/python3

# get_jupiter-center.py

# Debugging
import matplotlib.pyplot as plt
from jpm_fns import display

import numpy as np
from astropy.io import fits
from scipy import signal, ndimage

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
    """Return the center of an image, correcting for the ND filter.  Note X and Y are reversed"""

    # The coronagraph creates a margin of un-illuminated pixels on the
    # CCD.  These are great for estimating the bias and scattered
    # light for spontanous subtraction.  The ND filter provides a
    # similar peak after bias subutraction (or, rather, it is the
    # second such peak)

    H = fits.open(fname)
    im = np.asfarray(H[0].data)
    H.close()
    H = fits.open(flat)
    flat = np.asfarray(H[0].data)
    H.close()
    flat = flat[0:im.shape[0], 0:im.shape[1]]

    # --> Note that this may not be general and that I have seen the
    # --> ND filter on the flats show up in a different place than on
    # --> the images through the night.  Consider not passing a flat
    # --> image, but NDc that I have carefully prepared
    flat_hist, flat_hist_centers = hist_of_im(flat)
    im_hist, im_hist_centers = hist_of_im(im)

    flat_peak_idx = signal.find_peaks_cwt(flat_hist, np.arange(10, 50), min_snr=2)
    ND_left_idx = np.argmin(flat_hist[flat_peak_idx[0]:flat_peak_idx[1]])
    # The peaks past the bias and ND filter tend to be tiny.  Use one
    # of those to define the right side of the ND filter histogram peak
    ND_right_idx = flat_peak_idx[2]
        
    # Coordinates in image of ND filter
    NDc = np.where(np.logical_and(flat_hist_centers[ND_left_idx] < flat, flat < flat_hist_centers[ND_right_idx]))
    
    # Move ND filter to match this particular image (note Y, X)
    NDc = (NDc[0], NDc[1]-10)

    # Prepare to zero saturated pixels, since they confuse the region
    # around the ND filter if Jupiter or its scattered light is poking
    # out
    satc = np.where(im > 60000)

    # Subtract the bias of the image
    im_peak_idx = signal.find_peaks_cwt(im_hist, np.arange(10, 50))
    im -= im_hist_centers[im_peak_idx[0]]

    # Now zero scattered light, to avoid large negative
    im[satc] = 0

    # Correct for ND filter density, but avoid inducing noise by
    # looking only at points 1 std above median
    boostc = np.where(im[NDc] > (np.median(im[NDc]) + np.std(im[NDc])))
    NDc0 = NDc[0][boostc]
    NDc1 = NDc[1][boostc]
    NDc = (NDc0, NDc1)
    im[NDc] *= 1000

    plt.imshow(im)
    plt.show()
    #display(im, scale=None)
    

    y_x = ndimage.measurements.center_of_mass(im)
    return(y_x[::-1])


# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0032_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0033_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0034_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0035_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0036_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0037_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0038_off-band.fit'))
# 
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0042_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0043_off-band.fit'))
#print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0044_off-band.fit'))
print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0045_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0046_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0047_off-band.fit'))
# print(get_jupiter_center('/data/io/IoIO/raw/2017-04-20/IPT-0048_off-band.fit'))
