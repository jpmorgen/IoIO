#!/usr/bin/python3
import os
import re
import time
from multiprocessing import Pool
import csv
import argparse

import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from skimage import exposure
from astropy import log
from astropy import units as u
from astropy.time import Time, TimeDelta
# Using source for now
from astroquery.jplhorizons import Horizons
#from jplhorizons import Horizons
#from photutils import CircularAperture, aperture_photometry

import ccdproc
import matplotlib.pyplot as plt
import moviepy.editor as mpy

from precisionguide import get_HDUList
import IoIO
#from IoIO import CorObsData, run_level_default_ND_params
import define as D

# Constants for use in code
rversion = 0.2 # reduction version 
data_root = '/data/io/IoIO'
plate_scale = 1.56/2 # arcsec/pix
# For greping filter names out of headers, since I may change them or
# expand the number
# Na 5890A 10A FWHM
# [SII] 6731A 10A FWHM
# Na continuum 50A FWHM
# [SII] continuum 40A FWHM
line_associations = [{'line': '[SII]', 'cwl': '6731'},
                     {'line': 'Na',    'cwl': '5890'}]
# These are the equavalent widths of the filters in angstroms
SII_eq_width = 9.95
Na_eq_width = 11.22
# There is a noticeable scattered light loss in the Na on-band filter.
# Aperture photometry didn't fix it, so this is my guess at the
# magnitude of the problem
SII_on_loss = 0.95
Na_on_loss = 0.8
# Tue Jul 24 17:37:12 2018 EDT  jpmorgen@snipe
# See notes in ioio.notebk of this day
global_bias = 1633 # ADU
global_dark = 0.021 # ADU/s
reduce_edge_mask = -10 # Block out beyond ND filter
ap_sum_fname = 'ap_sum.csv'

# 80 would be perfect match.  Lets go a little short of that
#global_frame_rate = 80
global_frame_rate = 40
default_movie_speedup = 24000
movie_edge_mask = -8
#movie_edge_mask = 0
# Max perpendicular distance from center of ND filter
max_ND_dist = 20
# Hyperthreading is a little optimistic reporting two full processes
# per core.  Just stick with one process per core
threads_per_core = 2
# Raised this from 100 since I am doing a better job with bias & dark
background_light_threshold = 250 # ADU
movie_background_light_threshold = 250 # rayleighs
# Astrometry.  Just keep track of angle for now
astrometry = [('2017-03-03', 57),
              ('2017-03-04', 175-180),
              ('2017-03-05', 358),
              ('2017-03-07', 177-180),
              ('2017-03-12', 181-180),
              ('2017-03-13', 0),
              ('2017-04-14', 0),
              ('2017-04-17', 4),
              ('2017-05-18', 178-180),
              ('2018-01-15', 358),
              ('2018-01-24', 359)]

def get_astrometry_angle(date_obs):
    Tin = Time(date_obs, format='fits')
    alast = astrometry[0]
    T = Time(alast[0] + 'T00:00:00', format='fits')
    if Tin < T:
        raise ValueError(date_obs + ' before first IoIO observation!')
    for a in astrometry:
        T = Time(a[0] + 'T00:00:00', format='fits')
        if T > Tin:
            return(alast[1])
        alast = a
    return(alast[1])

def get_filt_band(header):
    f = header['FILTER'] 
    if 'cont' in f or 'off' in f:
        return 'off'
    # On-band is harder, since I don't labeled it as such...
    if 'on' in header['FILTER']:
        # ...but may in the future
        return 'on'
    line_assoc = [la for la in line_associations
                  if la['cwl'] in f]
    # Make it just one element
    line_assoc = line_assoc[0]
    if (line_assoc['line'] in f
        and (line_assoc['cwl'] in f
             or ("on" in f
                 and not "cont" in f))):
        return 'on'
    raise ValueError("Cannot identify band in " + f)

def get_filt_name(collection, line=None, band='None'):
    assert line is not None
    assert band is not None
    band = band.lower()  
    # unique has problems when biases are in the directory because
    # their missing FILTER keywords results in a masked value being
    # inserted into the array, which throws a error: TypeError:
    # unhashable type: 'MaskedConstant'.  So do it by hand
    if not 'filter' in collection.keywords:
        log.warning('FILTER keyword not present in any FITS headers')
        return None 
    filt_names = collection.values('filter')#, unique=True)
    filt_names = [f for f in filt_names
                  if isinstance(f, str)]
    # https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python
    filt_names = list(set(filt_names))
    line_assoc = [la for la in line_associations
                  if line in la['line']]
    # make it one item
    line_assoc = line_assoc[0]
    if band == 'on':
        filt = [f for f in filt_names
                if (line_assoc['line'] in f
                    and (line_assoc['cwl'] in f
                         or ("on" in f
                             and not "cont" in f)))]
    elif band == 'off':
        filt = [f for f in filt_names
                    if (line_assoc['line'] in f
                        and ("cont" in f
                             or "off" in f))]
    else:
        raise ValueError("Unknown band type " + band + ".  Expecting 'on' or 'off'")
    if len(filt) == 0:
        log.warning('Requested filter ' + line + ' ' + band + '-band not found')
        return None
    return filt[0]

        
def get_ND_params_1flat(fname):
    iter_ND_params = None
    try:
        # Iterate to get independent default_ND_params for each flat
        default_ND_params = None
        for i in np.arange(3):
            F = IoIO.CorObsData(fname, default_ND_params=iter_ND_params)
            iter_ND_params = F.ND_params
    except ValueError as e:
        log.error('Skipping: ' + fname + '. ' + str(e))
    return iter_ND_params

def get_default_ND_params(directory='.',
                          collection=None,
                          maxcount=None,
                          num_processes=None):
    """Derive default_ND_params from up to maxcount flats in directory.  Returns None if no (good) flats are found
    """
    assert os.path.isdir(directory), "Specify directory to search for flats and derive default_ND_params"
    if maxcount is None:
        maxcount = 10
    if num_processes is None:
        num_processes=int(os.cpu_count()/threads_per_core)
    if collection is None:
        collection = ccdproc.ImageFileCollection(directory)
    if not 'imagetyp' in collection.keywords:
        log.warning('IMAGETYP keyword not found in any files: ' + directory)
        return None 

    flats = collection.files_filtered(imagetyp='FLAT', include_path=True)
    if len(flats) == 0:
        return None
    maxcount = np.min((len(flats), maxcount))
    # Do this in parallel for speed :-)
    with Pool(processes=num_processes) as p:
        ND_params_list = p.map(get_ND_params_1flat, flats[0:maxcount])
    ND_params_list = [p for p in ND_params_list
                      if p is not None]
    if len(ND_params_list) == 0:
        return None
    ND_params_array = np.asarray(ND_params_list)
    default_ND_params \
        = ((np.median(ND_params_array[:, 0, 0]),
            np.median(ND_params_array[:, 0, 1])),
           (np.median(ND_params_array[:, 1, 0]),
            np.median(ND_params_array[:, 1, 1])))
    return np.asarray(default_ND_params)

def cmd_get_default_ND_params(args):
    print(get_default_ND_params(args.directory, args.maxcount))

def ND_params_dir(directory=None, default_ND_params=None):
    """Calculate ND_params for all observations in a directory
    """
    if default_ND_params is None:
        try:
            default_ND_params = get_default_ND_params(directory)
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            raise ValueError('Problem with flats in ' + directory + ': '  + str(e))
            
            if persistent_default_ND_params is not None:
                default_ND_params = persistent_default_ND_params

    # Collect file names
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    objs = []
    for f in sorted(files):
        if 'flat' in f.lower():
            pass
        elif 'bias' in f.lower():
            pass 
        elif 'dark' in f.lower():
            pass
        else:
            objs.append(os.path.join(directory, f))

    start = time.time()

    count = 0
    torus_count = 0
    Na_count = 0
    for count, f in enumerate(objs):
        D.say(f)
        try:
            O = IoIO.CorObsData(f, default_ND_params=default_ND_params)
            if O.header["EXPTIME"] == 300:
                if O.header["FILTER"] == "[SII] 6731A 10A FWHM":
                    torus_count += 1
                if O.header["FILTER"] == "Na 5890A 10A FWHM":
                    Na_count += 1
                
            D.say(O.obj_center)
            if O.obj_to_ND > 30:
                log.warning('Large dist: ' + str(int(O.obj_to_ND)))
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            log.error('Skipping: ' + str(e)) 

    elapsed = time.time() - start

    return((count, torus_count, Na_count, elapsed, count/elapsed))

def ND_params_tree(args):
    """Calculate ND_params for all observations in a directory tree
    """
    start = time.time()
    # We have some files recorded before there were flats, so get ready to
    # loop back for them
    skipped_dirs = []
    
    # list date directory one level deep
    totalcount = 0
    totaltorus = 0
    totalNa = 0
    totaltime = 0
    dirs = [os.path.join(args.directory, d)
            for d in os.listdir(args.directory) if os.path.isdir(os.path.join(args.directory, d))]
    persistent_default_ND_params = None
    for d in sorted(dirs):
        D.say(d)
        try:
            default_ND_params = get_default_ND_params(d)
            persistent_default_ND_params = default_ND_params
        except KeyboardInterrupt:
            # Allow C-C to interrupt
            raise
        except Exception as e:
            log.warning('Problem with flats: ' + str(e))
            if persistent_default_ND_params is None:
                log.error('No flats have been processed yet!  Skipping directory')
                continue
            log.warning('Using previous value for default_ND_params')
            default_ND_params = persistent_default_ND_params
        
        (count, torus_count, Na_count, elapsed, T_per_file) \
            = ND_params_dir(d, default_ND_params=default_ND_params)
        print(count, torus_count, Na_count, elapsed)
        totalcount += count
        totaltorus += torus_count
        totalNa += Na_count
        totaltime += elapsed
    end = time.time()
    D.say('Total elapsed time: ' + str(end - start) + 's')
    D.say(str(totalcount) + ' obj files took ' + str(totaltime) + 's')
    D.say('Average time per file: ' + str(totalcount / totaltime) + 's')
    D.say('Total torus: ' + str(totaltorus))
    D.say('Total Na: ' + str(totalNa))

def TiltImage(image, ImageNorth, JupiterNorth):
    # Assuming that "0" deg is to the right and ImageNorth is "90" deg.
    # Assuming JupiterNorth is relative to ImageNorth.
    DesiredChange = ImageNorth - JupiterNorth
    return ndimage.interpolation.rotate(image, DesiredChange)

    # Code above was provided by Daniel R. Morgenthaler, June 2018

#def IPTProfile(image):
def surf_bright(im, coord, minrad=3.):
    """Find average surface brightness using curve of growth"""
    r = minrad
    slist = []
    while r < 80:
        aperture = CircularAperture(coord, r=r)
        phot_tab = aperture_photometry(im, aperture)
        s = phot_tab['aperture_sum']
        sb = s/(np.pi * r**2)
        print(sb)
        r += 1

class Background():
    """Class for measuring and providing CCD background as a function of time using [SII] on-band images"""
    def __init__(self,
                 fname_or_directory=None,
                 collection=None,
                 num_processes=None):
        if fname_or_directory is None:
            fname_or_directory = '.'
        if num_processes is None:
            num_processes=int(os.cpu_count()/threads_per_core)
        SII_on_list = []
        if os.path.isfile(fname_or_directory):
            SII_on_list = [fname_or_directory]
        elif os.path.isdir(fname_or_directory):
            if collection is None:
                collection = ccdproc.ImageFileCollection(fname_or_directory)
            if not 'imagetyp' in collection.keywords:
                raise ValueError('IMAGETYP keyword not found in any files: ' + fname_or_directory)
            
        # Make a dictionary containing two keys: on and off, with
        # the lists of files for on and off-band filters in each
        fdict = {}
        for band in ['on', 'off']:
            SII_filt = get_filt_name(collection, '[SII]', band)
            flist = [os.path.join(fname_or_directory, l['file'])
                     for l in collection.summary
                     if (l['filter'] == SII_filt
                         and l['imagetyp'].lower() == 'light'
                         and l['xbinning'] == 1
                         and l['ybinning'] == 1)]
            if len(flist) == 0:
                continue
            fdict[band] = flist
        if not fdict:
            # --> could potentially do the best I can with Na or
            # another method
            raise ValueError('Background: no [SII] images to work with')
        self.jd_b_dict = {}
        self.spl_dict = {}
        for band in ['on', 'off']:
            with Pool(int(num_processes)) as p:
                jd_b_list = p.map(self.worker_get_back_level, fdict[band])
            self.jd_b_dict[band] = jd_b_list
            if len(jd_b_list) == 1:
                log.warning("Only one " + band + "-band image found, doing the best I can with it's background")
                self.spl_dict[band] = None
                return
        # UnivariateSpline needs things in order.  Thanks to
        # https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
        self.spl_dict = {}
        for band in ['on', 'off']:
            jd_b_list = self.jd_b_dict[band]
            jd_b_list.sort(key=lambda tup: tup[0])
            (jdlist, backlist) = zip(*jd_b_list)
            spl = UnivariateSpline(jdlist, backlist)
            self.spl_dict[band] = spl
        return

    def worker_get_back_level(self, f):
        HDUL = get_HDUList(f)
        T = Time(HDUL[0].header['DATE-OBS'], format='fits')
        b = IoIO.back_level(HDUL[0].data)
        return (T.jd, b)
    
    def background(self, header):
        """Returns best estimate background for time given in FITS string format"""
        band = get_filt_band(header)
        if self.spl_dict[band] is None:
            return self.jd_b_dict[band][0][1]
        T = Time(header['DATE-OBS'], format='fits')
        return np.asscalar(self.spl_dict[band](T.jd))

def get_tmid(l):
    """Get midpoint of observation whose FITS header is stored in dictionary l (l can be a line in a collection)"""
    return Time(l['date-obs'], format='fits') + l['exptime']/2*u.s

def strip_sum(im, center, ap_height, imtype, header, row):
    """Take aperture sums -- expressed as average pixel values -- in strips on the image.  ap_height = 0 whole image, ap_height > 0 strip of that height centered on center of image, ap_height < 0 two strips excluding strip of that height centered on center of image"""
    tim = im + 0
    if ap_height > 0:
        ny, nx = tim.shape
        # Blank out pixels above and below aperture strip
        tim[0:int(center[0]-ap_height/2), :] = 0
        tim[int(center[0]+ap_height/2):ny, :] = 0
    elif ap_height < 0:
        # Blank out the strip in the center
        tim[int(center[0]+ap_height/2):int(center[0]-ap_height/2), :] = 0
    asum = np.sum(tim)
    good_idx = np.where(tim != 0)
    asum /= len(good_idx[0])
    sap_height = str(abs(ap_height))
    if ap_height > 0:
        keypm = 'p'
        comstr = 'strip ' + sap_height + ' pix in Y'
    elif ap_height < 0:
        keypm = 'm'
        comstr = 'excluding strip ' + sap_height + ' pix in Y'
    else:
        keypm = '_'
        comstr = 'entire image'
    key = imtype + keypm + sap_height
    header[key] = (asum, 'average of ' + comstr)
    row[key] = asum
    return key        

def Rj_strip_sum(ang_width, im, center, Rj_ap_height, imtype, header, row):
    """Take aperture sums -- expressed as average pixel values -- in strips on the image.  ap_height = 0 whole image, ap_height > 0 strip of that height in Rj centered on center of image, ap_height < 0 two strips excluding strip of that height in Rj centered on center of image.  ang_width is the angular diameter of Jupiter in arcsec"""
    Rj = ang_width/2/plate_scale # arcsec / (arcsec/pix)
    #D.say('Rj = ', Rj, ' pixels')
    ap_height = int(Rj_ap_height * Rj)
    if ((abs(ap_height)/2 + center[0]) >= im.shape[0]
        or (center[0] - abs(ap_height)/2 < 0 )):
        log.warning('Rj_ap_height ' + str(Rj_ap_height) + ' Rj too large, setting aperture sum to zero')
        asum = 0
    else:
        tim = im + 0
        #D.say('Total of pixels in image: ', np.sum(tim))
        #D.say(Rj_ap_height, ' Rj ', ap_height, ' pix')
        if ap_height > 0:
            ny, nx = tim.shape
            #D.say('ycenter, ny: ', center[0], ny)
            # Blank out pixels above and below aperture strip
            tim[0:int(center[0]-ap_height/2), :] = 0
            tim[int(center[0]+ap_height/2):ny, :] = 0
            #impl = plt.imshow(tim, origin='lower',
            #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
            #plt.show()

        elif ap_height < 0:
            # Blank out the strip in the center
            tim[int(center[0]+ap_height/2):int(center[0]-ap_height/2), :] = 0
            #impl = plt.imshow(tim, origin='lower',
            #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
            #plt.show()
        asum = np.sum(tim)
        #D.say(asum)
        good_idx = np.where(tim != 0)
        asum /= len(good_idx[0])
        #D.say(asum)
    sap_height = str(abs(Rj_ap_height))
    if ap_height > 0:
        keypm = 'p'
        comstr = 'strip ' + sap_height + ' Rj in Y'
    elif ap_height < 0:
        keypm = 'm'
        comstr = 'excluding strip ' + sap_height + ' Rj in Y'
    else:
        keypm = '_'
        comstr = 'entire image'
    key = imtype + 'Rj' + keypm + sap_height
    header[key] = (asum, 'average of ' + comstr)
    row[key] = asum
    return key        

def Rj_box_sum(ang_width, im, center, Rj_ap_side, imtype, header, row):
    """Take aperture sums -- expressed as average pixel values -- for a square box Rj_ap_side Rj on a side.  Rj_ap_side = 0 whole image, Rj_ap_side > 0 box of that side and width in Rj centered on center of image, Rj_ap_side < 0 area outside of box.  ang_width is the angular diameter of Jupiter in arcsec"""
    Rjpix = ang_width/2/plate_scale # arcsec / (arcsec/pix)
    #D.say('Rj = ', Rj, ' pixels')
    ap_side = int(Rj_ap_side * Rjpix)
    if ((abs(ap_side)/2 + center[0]) >= im.shape[0]
        or (center[0] - abs(ap_side)/2 < 0 )
        or (abs(ap_side)/2 + center[1]) >= im.shape[1]
        or (center[1] - abs(ap_side)/2 < 0 )):
        log.warning('Rj_ap_side ' + str(Rj_ap_side) + ' Rj too large, setting aperture sum to zero')
        asum = 0
    else:
        tim = im + 0
        #D.say('Total of pixels in image: ', np.sum(tim))
        #D.say(Rj_ap_side, ' Rj ', ap_side, ' pix')
        if ap_side > 0:
            ny, nx = tim.shape
            #D.say('ycenter, ny: ', center[0], ny)
            # Blank out pixels outside of the box
            tim[0:int(center[0]-ap_side/2), :] = 0
            tim[int(center[0]+ap_side/2):ny, :] = 0
            tim[:, 0:int(center[1]-ap_side/2)] = 0
            tim[:, int(center[1]+ap_side/2):nx] = 0
            #impl = plt.imshow(tim, origin='lower',
            #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
            #plt.show()

        elif ap_side < 0:
            # Blank out the box
            tim[int(center[0]+ap_side/2):int(center[0]-ap_side/2),
            int(center[1]+ap_side/2):int(center[1]-ap_side/2)] = 0
            #impl = plt.imshow(tim, origin='lower',
            #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
            #plt.show()
        asum = np.sum(tim)
        #D.say(asum)
        good_idx = np.where(tim != 0)
        asum /= len(good_idx[0])
        #D.say(asum)
    sap_side = str(abs(Rj_ap_side))
    if ap_side > 0:
        keypm = 'p'
        comstr = 'box ' + sap_side + ' Rj in size'
    elif ap_side < 0:
        keypm = 'm'
        comstr = 'excluding box ' + sap_side + ' Rj in size'
    else:
        keypm = '_'
        comstr = 'entire image'
    key = imtype + 'Rj' + keypm + sap_side
    header[key] = (asum, 'average of ' + comstr)
    row[key] = asum
    return key        

def aperture_sum(im, center, y, x, r, imtype, header, row):
    """Take aperture sums of im.  y, x relative to center"""
    r2 = int(r/2)
    center = center.astype(int)
    asum = np.sum(im[center[0]+y-r2:center[0]+y+r2,
                     center[1]+x-r2:center[1]+x+r2])
    asum /= r**2
    key = imtype + 'AP' + str(x) + '_' + str(y)
    header[key] = (asum, 'square aperture average at x, y, r = ' + str(r))
    row[key] = asum
    return key        

def Rj_aperture_sum(ang_width, im, center, yRj, xRj, rpix, imtype, header, row):
    """Take aperture sums of im.  y, x relative to center"""
    r2 = int(rpix/2)
    center = center.astype(int)
    Rjpix = ang_width/2/plate_scale # arcsec / (arcsec/pix)
    y = int(yRj * Rjpix)
    x = int(xRj * Rjpix)
    center = center.astype(int)
    asum = np.sum(im[center[0]+y-r2:center[0]+y+r2,
                     center[1]+x-r2:center[1]+x+r2])
    asum /= rpix**2
    key = imtype + 'AP' + str(xRj) + '_' + str(yRj)
    header[key] = (asum, str(rpix) + ' pix square aperture [-]NNN_[-]MMM Rj from Jupiter' )
    row[key] = asum
    return key        

def reduce_pair(OnBand_HDUList_im_or_fname=None,
                OffBand_HDUList_im_or_fname=None,
                back_obj=None,
                default_ND_params=None,
                NPole_ang=None,
                ang_width=None,
                outfname=None,
                recalculate=False):
    # Let these raise errors if our inputs have problems
    OnBand_HDUList = get_HDUList(OnBand_HDUList_im_or_fname)
    OffBand_HDUList = get_HDUList(OffBand_HDUList_im_or_fname)
    log.debug(OnBand_HDUList.filename() + ' ' + OffBand_HDUList.filename())
    # Check to see if we want to recalculate & overwrite.  Do this
    # in general so we can be called at any directory level
    # (though see messages for issues)
    header = OnBand_HDUList[0].header
    if outfname is None:
        rawfname = OnBand_HDUList.filename()
        if rawfname is None:
            log.warning('On-band image was not associated with any filename and outfname is not specified, writing to current directory, ReducedCorObs.fits')
            outfname = 'ReducedCorObs.fits'
        elif not os.path.isabs(rawfname):
            log.warning("Outfname not specified and on-band image fname was not an absolute path and outfname is not specified.  I can't deconstruct the raw to reduced path structure, writing to current directory, ReducedCorObs.fits")
            outfname = 'ReducedCorObs.fits'
        else:
            # Expect raw filenames are of the format
            # /data/io/IoIO/raw/2018-05-20/Na_on-band_001.fits
            # Create reduced of the format
            # /data/io/IoIO/reduced/2018-05-20/Na_on-band_001r.fits
            # --> Consider making the filename out of the line and on-off
            # We should be in our normal directory structure
            basename = os.path.basename(rawfname)
            # Insert "r" so no collisions are possible
            (fbase, ext) = os.path.splitext(basename)
            redbasename = fbase + 'r' + ext
            # --! This is an assumtion
            rawdatepath = os.path.dirname(rawfname)
            datedir = os.path.split(rawdatepath)[1]
            red_data_root = os.path.join(data_root, 'reduced')
            reddir = os.path.join(data_root, 'reduced', datedir)
            outfname = os.path.join(reddir, redbasename)

    # Return if we have nothing to do.
    if (not recalculate
        and os.path.isfile(outfname)):
        log.debug('skipping -- output file exists and recalculate=False: '
                  + outfname)
        return

    # Use IoIO.CorObsData to get basic properties like background level
    # and center.
    OnBandObsData = IoIO.CorObsData(OnBand_HDUList,
                                    default_ND_params=default_ND_params,
                                    edge_mask=reduce_edge_mask)
    if OnBandObsData.quality < 5:
        log.warning('Skipping: poor quality center determination for '
                    + OnBand_HDUList.filename())
        return
    OffBandObsData = IoIO.CorObsData(OffBand_HDUList,
                                     default_ND_params=default_ND_params,
                                     edge_mask=reduce_edge_mask)
    if OffBandObsData.quality < 5:
        log.warning('Skipping: poor quality center determination for '
                    + OffBand_HDUList.filename())
        return
    if back_obj is None:
        rawfname = OnBand_HDUList.filename()
        if rawfname is None:
            back_obj = Background(OnBand_HDUList)
        else:
            back_obj = Background(os.path.dirname(rawfname))
    bias_dark = back_obj.background(OnBandObsData.header)
    on_im = OnBand_HDUList[0].data -  bias_dark
    header['ONBSUB'] = (bias_dark,
                         'on-band back (bias, dark) value subtracted')
    header['DONBSUB'] = (bias_dark - OnBandObsData.back_level,
                         'on-band back - ind. est. back via histogram')

    on_back = np.mean(on_im)
    if on_back > background_light_threshold:
        log.warning('On-band background level too high: ' + str(on_back)
                    + ' for ' + OnBand_HDUList.filename())
        return
    bias_dark = back_obj.background(OffBandObsData.header)
    off_im = OffBand_HDUList[0].data - bias_dark
    #D.say('Off-band difference between bias+dark and first hist peak: ' ,
    #      bias_dark - OffBandObsData.back_level)
    off_back = np.mean(off_im)
    if off_back > background_light_threshold:
        log.warning('Off-band background level too high: ' + str(off_back)
                    + ' for ' + OffBand_HDUList.filename())
        return
    if abs(on_back - off_back) > background_light_threshold:
        log.warning('Off-band minus off-band background level too high for '
                    + OnBand_HDUList.filename() + ' '
                    + OffBand_HDUList.filename())
        return
    header['OFFBSUB'] = (bias_dark,
                         'off-band back (bias, dark) value subtracted')
    header['DOFFBSUB'] = (bias_dark - OffBandObsData.back_level,
                         'off-band back - ind. est. back via histogram')
    # --> Worry about flat-fielding later
    # --> Make all these things FITS keywords
    # Get ready to shift off-band image to match on-band image
    # --> consider making a better center finder with correlation with
    # 90 degree flip method that Carl suggested
    on_center = OnBandObsData.obj_center
    off_center = OffBandObsData.obj_center
    if OnBandObsData.header['OBJ2NDC'] > max_ND_dist:
        log.error('on-band image: obj too far off center of ND filter')
        return
    if OffBandObsData.header['OBJ2NDC'] > max_ND_dist:
        log.error('off-band image: obj too far off center of ND filter')
        return
    shift_off = on_center - off_center
    d_on_off = np.linalg.norm(shift_off)
    OnBandObsData.header['D_ON-OFF'] = (d_on_off, 'dist in pix between on and off centers')
    if d_on_off > 5:
        log.warning('On- and off-band image centers are > 5 pixels apart')
            
    # on_jup and off_jup are the average brightness over 1 pixel.
    # Call that a pixel-averaged surface brightness.
    # --> Tried fancy surf_bright method and that didn't seem to
    # help Na over-subtraction problem  
    #surf_bright(on_im, on_center)
    #surf_bright(off_im, off_center)

    # Wed Oct 31 13:19:25 2018 EDT  jpmorgen@snipe
    # --> Email discussion of yesterday and today suggests better
    # on/off calibration is done using ratio of identically exposed
    # on-and off-band images of sources away from the ND filter.  I
    # think I have stars that can oblige, maybe even day sky.
    on_center = np.round(on_center).astype(int)
    off_center = np.round(off_center).astype(int)
    on_jup = np.average(on_im[on_center[0]-5:on_center[0]+5,
                              on_center[1]-5:on_center[1]+5])
    off_jup = np.average(off_im[off_center[0]-5:off_center[0]+5,
                                off_center[1]-5:off_center[1]+5])
    off_im = ndimage.interpolation.shift(off_im, shift_off)
    # Note transpose for FITS/FORTRAN from C world
    header['OFFS0'] = (shift_off[1], 'off-band axis 0 shift to align w/on-band')
    header['OFFS1'] = (shift_off[0], 'off-band axis 1 shift to align w/on-band')
    if '[SII]' in OnBandObsData.header['FILTER']:
        line = '[SII]'
        eq_width = SII_eq_width
        on_loss = SII_on_loss
    elif 'Na' in OnBandObsData.header['FILTER']:
        line = 'Na'
        eq_width = Na_eq_width
        on_loss = Na_on_loss
    else:
        raise ValueError('Improper filter ' +
                         OnBandObsData.header['FILTER'])
    header['ON_LOSS'] \
        = (on_loss, 'on-band scat. light loss for discrete sources')
    off_im[OffBandObsData.ND_coords] = 0
    offscale = on_jup/off_jup
    off_im = off_im * offscale * on_loss
    scat_sub_im = on_im - off_im
    ## DEBUGGING
    #bad_idx = np.where(on_im > 100)
    #on_im[bad_idx] = 0
    #impl = plt.imshow(on_im, origin='lower',
    #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
    #plt.show()
    #bad_idx = np.where(off_im > 100)
    #off_im[bad_idx] = 0
    #impl = plt.imshow(on_im, origin='lower',
    #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
    #plt.show()
    # Get an on-band ObsData that has the ND_coords inside the edge of
    # the ND filter and use that to define the good ND filter pixels
    O = IoIO.CorObsData(OnBand_HDUList, default_ND_params=default_ND_params)
    good_ndpix = scat_sub_im[O.ND_coords]
    # Blank out our ND filter with the reduce_edge_mask
    scat_sub_im[OnBandObsData.ND_coords] = 0
    # Poke back in good pixels
    scat_sub_im[O.ND_coords] = good_ndpix
    header['OFFFNAME'] = (OffBand_HDUList.filename(),
                          'off-band file')
    header['OFFSCALE'] = (offscale, 'scale factor applied to off-band im')
    # Establish calibration in Rayleighs.  Brown & Schneider 1981
    # Jupiter is 5.6 MR/A
    # --> This is between the Na lines, so it is not quite right.  The
    # Na lines will knock it down by 10-20%

    # Tue Nov 13 11:30:03 2018 EST  jpmorgen@snipe

    # Carl's calculations in an email of yesterday with code delivered
    # last week (~/pro/IoIO/Ioio_flux_cal.pro) incorporate the
    # spectra, albedo, etc. suggest that for Na, the effective
    # continuum is 52.6 MR to 54.0 MR.  Since the bandpass is 11.22,
    # the calculations below end up using 62.832E6 or 1.17 too high.
    MR = 5.6E6 * eq_width
    ADU2R = on_jup * 1000 / MR
    # 1000 is ND filter
    # Tue Nov 13 11:38:04 2018 EST  jpmorgen@snipe
    # See notes of today in ~/IoIO_reduction.notebk which measure ND
    # filter to be more like 734 instead of 1000
    scat_sub_im = scat_sub_im / ADU2R
    header['BUNIT'] = ('rayleighs', 'pixel unit')
    header['ADU2R'] = (ADU2R, 'conversion factor from ADU to R')
    # --> will want to check this earlier for proper pairing of
    # on-off band images
    pier_side = OnBandObsData.header.get('PIERSIDE')
    if pier_side is not None and pier_side == 'EAST':
        gem_flip = 180
    else:
        gem_flip = 0
        
    # --> consider just storing this in the header and letting
    # --> subsequent reduction & analysis do things in native
    # --> coordinates
    
    # Calculate NPole_ang if we weren't passed it, but store it in
    # property so it can be used for the next file in this day.
    # Also truncate to the integer so that astroquery caching can
    # work --> might want to loosen this for nth degree
    # calculations when I am ready for those
    if NPole_ang is None:
        # V09 is the Moca observatory at Benson, which looks like
        # the San Pedro Valley observatory
        T = Time(header['DATE-OBS'], format='fits')
        # --> eventually I might want to be general with the object
        jup = Horizons(id=599,
                       location='V09',
                       epochs=np.round(T.jd),
                       id_type='majorbody')
        NPole_ang = jup.ephemerides()['NPole_ang'].quantity.value[0]
        ang_width = jup.ephemerides()['ang_width'].quantity.value[0]
    # Save off original center of image for NDparams update, below
    o_center = np.asarray(scat_sub_im.shape)/2
    on_shift = o_center - OnBandObsData.obj_center
    aangle = get_astrometry_angle(header['DATE-OBS'])
    on_angle = aangle - NPole_ang + gem_flip
    # interpolation.rotate rotates CW for positive angle
    scat_sub_im = ndimage.interpolation.shift(scat_sub_im, on_shift)
    scat_sub_im = ndimage.interpolation.rotate(scat_sub_im, on_angle)

    # Update centers and NDparams
    center = np.asarray(scat_sub_im.shape)/2
    header['OBJ_CR0'] = (center[1], 'Object center X')
    header['OBJ_CR1'] = (center[0], 'Object center Y')
    header['DES_CR0'] = (center[1], 'Desired center X')
    header['DES_CR1'] = (center[0], 'Desired center Y')
    # Tried to do this in the general case but I got confused by
    # the geometry or a rolling cube.  Plus I am not set up to
    # deal with the ND filter in the horizontal position
    on_angle -= gem_flip
    ron_angle = np.radians(on_angle)
    # Offsets
    ND01 = header['NDPAR01']
    ND11 = header['NDPAR11']
    # Note on_shift is y,x
    xshift = np.dot(on_shift, np.asarray(((0,1))))
    ND01 += xshift
    ND11 += xshift
    ND01 = (o_center[0] * abs(np.sin(ron_angle))
            + ND01 * np.cos(ron_angle))
    ND11 = (o_center[0] * abs(np.sin(ron_angle))
            + ND11 * np.cos(ron_angle))
    header['NDPAR01'] = ND01
    header['NDPAR11'] = ND11
    # Angles
    ND00 = header['NDPAR00']
    ND10 = header['NDPAR10']
    
    # Tricky!  Swapped image so north is up
    header['NDPAR00'] = -np.tan(np.arctan(ND00) + ron_angle)
    header['NDPAR10'] = -np.tan(np.arctan(ND10) + ron_angle)

    # Coronagraph flips images N/S.  Transpose alert
    scat_sub_im =np.flipud(scat_sub_im)

    # Get ready to write some output to the reduced directory
    if not os.path.exists(red_data_root):
        os.mkdir(red_data_root)
    if not os.path.exists(reddir):
        os.mkdir(reddir)

    # Do some quick-and-dirty aperture sums
    # --> improve on this
    #fieldnames = ['TMID', 'ANGDIAM', 'EXPTIME', 'FNAME', 'LINE', 'ONBSUB', 'OFFBSUB', 'DONBSUB', 'DOFFBSUB']
    this_ap_sum_fname = os.path.join(reddir, ap_sum_fname)
    tmid = (get_tmid(header)).fits
    header['TMID'] = (tmid, 'Midpoint of observation')
    header['ANGDIAM'] = (ang_width, 'Angular diameter of Jupiter (arcsec)')
    row = {'FNAME': outfname,
           'LINE': line,
           'TMID': tmid,
           'EXPTIME': header['EXPTIME'],
           'ONBSUB': header['ONBSUB'],
           'DONBSUB': header['DONBSUB'],
           'OFFBSUB': header['OFFBSUB'],
           'DOFFBSUB': header['DOFFBSUB'],
           'OFFSCALE': header['OFFSCALE'],
           'ADU2R': ADU2R,
           'ANGDIAM': ang_width}
    fieldnames = list(row.keys())
    # Remember to shift, rotate, and calibrate On and Off images
    for imtype in ['AP', 'On', 'Off']:
        if imtype == 'AP':
            im = scat_sub_im
        elif imtype == 'On':
            on_im = ndimage.interpolation.shift(on_im, on_shift)
            on_im = ndimage.interpolation.rotate(on_im, on_angle)
            im = on_im / ADU2R
        elif imtype == 'Off':
            off_im = ndimage.interpolation.shift(off_im, on_shift)
            off_im = ndimage.interpolation.rotate(off_im, on_angle)
            im = off_im / ADU2R
        else:
            raise ValueError('Unknown imtype ' + imtype)
        center = np.asarray(im.shape)/2
        #for ap_height in [0, 1200, 600, 300, -300, -600, -1200]:
        #    key = strip_sum(im, center, ap_height, imtype, header, row)
        #    fieldnames.append(key)
        for ap_box in [0, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 15, 10, 5]:
            key = Rj_box_sum(ang_width, im, center, ap_box, imtype, header, row)
            fieldnames.append(key)
        #for y in [600, 150, 0, -150, -600]:
        #    for x in [-600, -500, -400, -300, -200, 200, 300, 400, 500, 600]:
        #        key = aperture_sum(im, center, y, x, 60, imtype, header, row)
        #        fieldnames.append(key)
        #for y in [30, 20, 10, 5, 0, -5, -10, -20, -30]:
        #    for x in [30, 20, 10, 5, 0, -5, -10, -20, -30]:
        #        key = Rj_aperture_sum(ang_width, im, center, y, x, 60, imtype, header, row)
        #        fieldnames.append(key)
    rdate = (Time.now()).fits
    header['RDATE'] = (rdate, 'UT time of reduction')
    header['RVERSION'] = (rversion, 'Reduction version')
    row['RDATE'] = rdate
    row['RVERSION'] = rversion
    fieldnames.extend(['RDATE', 'RVERSION'])
    if not os.path.exists(this_ap_sum_fname):
        with open(this_ap_sum_fname, 'w', newline='') as csvfile:
            csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                   quoting=csv.QUOTE_NONNUMERIC)
            csvdw.writeheader()
    with open(this_ap_sum_fname, 'a', newline='') as csvfile:
        csvdw = csv.DictWriter(csvfile, fieldnames=fieldnames,
                               quoting=csv.QUOTE_NONNUMERIC)
        csvdw.writerow(row)

    # Get ready to write
    OnBand_HDUList[0].data = scat_sub_im
    OnBand_HDUList.writeto(outfname, overwrite=recalculate)

class ReduceDir():
    def __init__(self,
                 directory=None,
                 collection=None,
                 recalculate=False,
                 back_obj=None,
                 default_ND_params=None,
                 NPole_ang=None,
                 ang_width=None,
                 num_processes=None,
                 movie=None):
        assert directory is not None
        self.directory = directory
        self._collection = collection
        self.recalculate = recalculate
        self._back_obj = back_obj
        self._default_ND_params = default_ND_params
        # --> This will eventually be a more involved set of ephemerides outputs
        self.NPole_ang = NPole_ang
        self.ang_width = ang_width
        self.num_processes = num_processes
        self.movie = movie
        self.reduce_dir()

    @property
    def collection(self):
        if self._collection is not None:
            return self._collection
        self._collection = ccdproc.ImageFileCollection(self.directory)
        return self._collection

    @property
    def back_obj(self):
        if self._back_obj is not None:
            return self._back_obj
        self._back_obj = Background(self.directory, self.collection)

    @property
    def default_ND_params(self):
        if self._default_ND_params is not None:
            return self._default_ND_params
        self._default_ND_params \
            = get_default_ND_params(self.directory, self.collection)
        if self._default_ND_params is None:
            log.warning('No (good) flats found in directory '
                        + self.directory + ' using run_level_default_ND_params')
            self._default_ND_params = IoIO.run_level_default_ND_params
        return self._default_ND_params

    def worker_reduce_pair(self, pair):
        try:
            reduce_pair(pair[0],
                        pair[1],
                        back_obj=self.back_obj,
                        default_ND_params=self.default_ND_params,
                        NPole_ang=self.NPole_ang,
                        ang_width=self.ang_width,
                        recalculate=self.recalculate)
        except Exception as e:
            log.error(str(e) + ' skipping ' + pair[0] + ' ' + pair[1])

        
    def reduce_dir(self):
        if not 'filter' in self.collection.keywords:
            log.warning('FILTER keyword not present in any FITS headers, no usable files in ' + self.directory)
            return None 
        summary_table = self.collection.summary
        reduced_dir = self.directory.replace('/raw/', '/reduced/')
        # --! We need to make sure the Background and
        # default_ND_params code has run once so it evaluates to a
        # value rather than a method when used with multiprocessing
        # otherwise it raises a "daemonic processes are not allowed to
        # have children" error
        try:
            self.back_obj
        except Exception as e:
            log.error(str(e) + ' skipping ' + self.directory)
            return

        self.default_ND_params

        # Create a list of on-band off-band pairs that are closest in
        # time for each of our science lines
        line_names = ['[SII]', 'Na']
        on_off_pairs = []
        for line in line_names:
            on_filt = get_filt_name(self.collection, line, 'on')
            off_filt = get_filt_name(self.collection, line, 'off')
            on_idx = [i for i, l in enumerate(summary_table)
                      if (l['filter'] == on_filt
                          and l['imagetyp'].lower() == 'light'
                          and l['xbinning'] == 1
                          and l['ybinning'] == 1)]
            if len(on_idx) == 0:
                break
            off_idx = [i for i, l in enumerate(summary_table)
                       if (l['filter'] == off_filt
                           and l['imagetyp'].lower() == 'light'
                           and l['xbinning'] == 1
                           and l['ybinning'] == 1)]

            if len(off_idx) == 0:
                break
            for i_on in on_idx:
                tmid_on = get_tmid(summary_table[i_on])
                dts = [tmid_on - T for T in get_tmid(summary_table[off_idx])]
                # Unwrap
                i_off = off_idx[np.argmin(np.abs(dts))]
                on_fname = os.path.join(self.directory,
                                        summary_table[i_on]['file'])
                off_fname = os.path.join(self.directory,
                                         summary_table[i_off]['file'])
                pair = [os.path.join(self.directory,
                                     summary_table[i]['file'])
                                     for i in (i_on, i_off)]
                on_off_pairs.append(pair)
        if len(on_off_pairs) == 0:
            log.warning('No object files found in ' + self.directory)
            return
    
        # --> I am going to want to improve this
        if self.NPole_ang is None:
            # Reading in our first file is the easiest way to get the
            # properly formatted date for astroquery.  Use UT00:00,
            # since astroquery caches and repeat querys for the whole
            # day will therefore benefit
            HDUL = get_HDUList(on_off_pairs[0][0])
            T = Time(HDUL[0].header['DATE-OBS'], format='fits')
            jup = Horizons(id=599,
                           location='V09',
                           epochs=np.round(T.jd),
                           id_type='majorbody')
            self.NPole_ang = jup.ephemerides()['NPole_ang'].quantity.value[0]
            self.ang_width = jup.ephemerides()['ang_width'].quantity.value[0]
        
        # Get our summary file(s) ready.
        this_ap_sum_fname = os.path.join(reduced_dir, ap_sum_fname)
        if self.recalculate and os.path.isfile(this_ap_sum_fname):
            # All files will be rewritten, so we want to start with a
            # frech file
            os.remove(os.path.join(reduced_dir, ap_sum_fname))
        else:
            # If reduced files already exist, they will be skipped,
            # but we may have improved the code to add more reduced
            # files, in which case we want their records to append to
            # the summary file(s)
            pass
        start = time.time()
        with Pool(int(args.num_processes)) as p:
            p.map(self.worker_reduce_pair, on_off_pairs)

        elapsed = time.time() - start
        log.info('Elapsed time for ' + self.directory + ': ' + str(elapsed))
        log.info('Average per file: ' +
                 str(elapsed/(len(on_off_pairs))))

        if self.movie is not None:
            # We can be lazy here, since we know our directory
            # structure and OS
            try:
                make_movie(reduced_dir, recalculate=args.recalculate)
            except Exception as e:
                log.error(str(e) + ' skipping movie for ' + self.directory)
        return

def get_dirs(directory, filt_list=None):
    """get sorted list of subdirectories excluding dirs containing strings in filt_list"""
    assert os.path.isdir(directory)
    return [os.path.join(directory, d) for d in sorted(os.listdir(directory))
            if (os.path.isdir(os.path.join(directory, d))
                and (filt_list is None
                     or not np.any([filt in d for filt in filt_list])))]
    

def reduce_cmd(args):
    if args.tree is not None:
        top = args.directory
        if top is None:
            top = os.path.join(data_root, 'raw')
        persistent_default_ND_params = None
        for directory in reversed(get_dirs(top)):
            collection = ccdproc.ImageFileCollection(directory)
            log.info(collection.location)
            if args.default_ND_params is None:
                # We usually expect this, since we are going to run on
                # a wide range of directories with different ND
                # parameters
                default_ND_params \
                    = get_default_ND_params(directory, collection)
                if (default_ND_params is None
                    and persistent_default_ND_params is None):
                    # First time through no flats.  Presumably this is
                    # recent data from the current run
                    default_ND_params = IoIO.run_level_default_ND_params
                    log.warning('No default_ND_params supplied and flats in '
                                + directory)
                elif (default_ND_params is None
                      and persistent_default_ND_params is not None):
                    # No flats in current directory, use previous value
                    default_ND_params = persistent_default_ND_params
                persistent_default_ND_params = default_ND_params
            R = ReduceDir(directory,
                          collection=collection,
                          recalculate=args.recalculate,
                          default_ND_params=default_ND_params,
                          num_processes=args.num_processes,
                          movie=args.movie)
        redtop = top.replace('/raw', '/reduced')
        filt_list = ['cloudy', 'marginal', 'dew', 'bad']
        csvlist = [os.path.join(d, 'ap_sum.csv')
                   for d in get_dirs(redtop, filt_list=filt_list)
                   if os.path.isfile(os.path.join(d, 'ap_sum.csv'))]
        # https://stackoverflow.com/questions/13613336/python-concatenate-text-files
        first = True
        with open(os.path.join(redtop, 'ap_sum.csv'), 'w') as outfile:
            for fname in csvlist:
                with open(fname) as infile:
                    if not first:
                        # Read past the header
                        infile.readline()
                    first = False
                    outfile.write(infile.read())
        if args.movie is not None:
            movie_concatenate(redtop)
        return

    if args.directory is not None:
        R = ReduceDir(args.directory,
                      recalculate=args.recalculate,
                      default_ND_params=args.default_ND_params,
                      num_processes=args.num_processes,
                      movie=args.movie)
        return
    # Reduce a pair of files -- just keep it simple
    if len(args.on_band) == 2:
        on_band = args.on_band[0]
        off_band = args.on_band[1]
    else:
        on_band = args.on_band
        off_band = args.off_band
    reduce_pair(on_band,
                off_band,
                default_ND_params=args.default_ND_params,
                recalculate=args.recalculate)

class MovieCorObs():
    def __init__(self,
                 flist,
                 speedup=None,
                 frame_rate=None,
                 crop=None):
        assert isinstance(flist, list) and len(flist) > 0
        self.flist = flist
        self.speedup = speedup
        self.frame_rate = frame_rate
        self.crop = crop
        # We live in transpose space in a C language
        self.mp4shape = np.asarray((480,640))
        if self.speedup is None:
            self.speedup = default_movie_speedup
        if self.frame_rate is None:
            self.frame_rate=global_frame_rate
        if crop is None:
            # We need to make sure our image isn't too big for the mp4
            self.crop = 4*self.mp4shape
        else:
            # reading arrays stored in a Fortran memory style
            self.crop = np.asarray(crop.lower().split('x')).astype(int)
            # [::-1] does transpose, since we are in C-style language
            self.crop = self.crop[::-1]
            # Do this like an iterator
        self.fnum = None
        self.dt_cur = None
        self.HDUCur = None
        self.HDUNext = None
        self.persist_im = None
        self.last_persist_im = None
        self.next_f()
        HDULast = get_HDUList(flist[-1])
        self.Tstop = Time(HDULast[0].header['DATE-OBS'], format='fits')
        last_exp = HDULast[0].header['EXPTIME']
        # This is the calculated duration
        self.duration = ((self.Tstop - self.Tstart).sec + last_exp)/self.speedup

    def prev_f(self):
        assert self.fnum > 0
        self.persist_im = None
        self.HDUNext = self.HDULcur
        self.dt_next = self.dt_cur
        self.fnum -= 1
        self.HDULcur = get_HDUList(self.flist[self.fnum])
        T = Time(self.HDULcur[0].header['DATE-OBS'], format='fits')
        self.dt_cur = (T - self.Tstart).sec

    def next_f(self):
        if self.persist_im is not None:
            self.last_persist_im = self.persist_im
        self.persist_im = None
        if self.fnum is None:
            # Initialize here, since we have shared code
            self.fnum = 0
            self.HDULcur = get_HDUList(self.flist[self.fnum])
            if '[SII]' in self.HDULcur[0].header['FILTER']:
                self.filt = '[SII]'
            elif 'Na' in self.HDULcur[0].header['FILTER']:
                self.filt = 'Na'
            else:
                raise ValueError('Improper filter ' +
                                 OnBandObsData.header['FILTER'])
        else:
            assert self.HDUNext is not None, 'Code error? running off the end'
            self.HDULcur = self.HDUNext
            self.fnum += 1
        T = Time(self.HDULcur[0].header['DATE-OBS'], format='fits')
        if self.dt_cur is None:
            self.Tstart = T
            self.dt_cur = 0
        else:
            self.dt_cur = (T - self.Tstart).sec
        if self.fnum < len(self.flist) - 1:
            self.HDUNext = get_HDUList(self.flist[self.fnum + 1])
            T = Time(self.HDUNext[0].header['DATE-OBS'], format='fits')
            self.dt_next = (T - self.Tstart).sec
        else:
            self.HDUNext = None
            self.dt_next = self.dt_cur + self.HDULcur[0].header['EXPTIME']

    def get_good_frame(self, t):
        if self.last_persist_im is not None:
            # Some previous image was good.  Use it in place of the
            # bad one.
            self.persist_im = self.last_persist_im
            return self.persist_im
        # If we made it here, our first image was bad.  Recursively
        # read the next one and pretend it is the first until we find
        # a good one
        if self.HDUNext is None:
            raise ValueError ('No good images')
        self.next_f()
        self.dt_cur = 0
        return(self.make_frame(t))
        
    def make_frame(self, t):
        # Make a general backward-forward iterator since sometimes we
        # run the object backward to get back to the beginning of a
        # movie.  The idea is we read a frame in and store it in
        # persist_im until we advance time past the beginning of the
        # next frame.
        m_dt = t * self.speedup
        while t > 0 and m_dt < self.dt_cur:
            self.prev_f()
        while t <= self.duration and self.dt_next <= m_dt:
            self.next_f()
        if self.persist_im is not None:
            return self.persist_im
        # If we made it here, we need to create our image
        # Do some checks to see if it is crummy
        if self.HDULcur[0].header['D_ON-OFF'] > 5:
            log.warning('on & off centers too far apart '
                        + self.HDULcur.filename())
            return(self.get_good_frame(t))
        im = self.HDULcur[0].data
        if abs(np.mean(im)) > movie_background_light_threshold:
            log.warning('background light too large or small for '
                        + self.HDULcur.filename())
            return(self.get_good_frame(t))
        # --> playing with these on 2018-04-21 [seem good in general]
        if self.filt == '[SII]':
            chop = 2000
            scale_jup = 100
        else:
            chop = 8000
            scale_jup = 50
        # Might want to adjust edge_mask.  -5 was OK on 2018-04-21
        O = IoIO.CorObsData(self.HDULcur, edge_mask=movie_edge_mask)
        c = (np.asarray(im.shape)/2).astype(int)
        # Scale Jupiter down by 10 to get MR/A and 10 to get
        # it on comparable scale to torus
        im[O.ND_coords] = im[O.ND_coords] /scale_jup
        #jcrop = np.asarray((50,50))
        #ll = (c - jcrop).astype(int)
        #ur = (c + jcrop).astype(int)
        #im[ll[0]:ur[0], ll[1]:ur[1]] \
        #    = im[ll[0]:ur[0], ll[1]:ur[1]]/10/10
        if self.crop is not None:
            ll = (c - self.crop/2).astype(int)
            ur = (c + self.crop/2).astype(int)
            im = im[ll[0]:ur[0], ll[1]:ur[1]]
        # chop high pixels
        badc = np.where(np.logical_or(im < 0, im > chop))
        im[badc] = 0
        # Keep it linear for now on [SII] -- this accentuated noise
        if self.filt == 'Na':
            #--> try adjust_log here
            im = exposure.equalize_adapthist(np.asarray(im/np.max(im)))
            # Logarithmic brights out the noise no matter what
            #im = exposure.adjust_log(np.asarray(im), gain=0.25)

        # mp4 wants to be 640 x 360 or 640 × 480
        # Note transpose space for C-style language
        scale = (np.round(im.shape / self.mp4shape)).astype(int)
        if np.any(scale > 1):
            scale = np.max(scale)
            im = ndimage.zoom(im, 1/scale, order=0)
        # Scale pixel values for mp4
        im = im/np.max(im) * 255
        # MP4 thinks of pixels coordinates in the X-Y Cartesian sense,
        # but filling in from the top down
        im = np.flipud(im)

        # DEBUGGING
        #impl = plt.imshow(im, origin='lower',
        #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
        #plt.show()
        
        # Thanks to https://stackoverflow.com/questions/39463019/how-to-copy-numpy-array-value-into-higher-dimensions
        self.persist_im = np.stack((im,) * 3, axis=-1)
        return self.persist_im

def make_movie(directory,
               recalculate=False,
               SII_crop=None,
               Na_crop=None,
               frame_rate=None,
               speedup=None):
    assert not 'raw' in directory, "Not ready to make movies from raw directories" 
        
    # Return if we have nothing to do.  Eventually put all desired
    # products in here, though for now this is the only one
    if (not recalculate
        and os.path.isfile(os.path.join(directory, "Na_SII.mp4"))):
        log.debug('movie output file(s) exist and recalculate=False: '
                  + directory)
        return
    collection = ccdproc.ImageFileCollection(directory)
    collection.sort('date-obs')
    if not 'filter' in collection.keywords:
        log.warning('Directory does not contain any usable images')
        return        
    SII_filt = get_filt_name(collection, '[SII]', 'on')
    Na_filt = get_filt_name(collection, 'Na', 'on')
    SII_on_list = collection.files_filtered(filter=SII_filt,
                             include_path=True)
    Na_on_list = collection.files_filtered(filter=Na_filt,
                            include_path=True)

    if len(SII_on_list) == 0 or len(Na_on_list) == 0:
        # Just return if we don't have BOTH [SII] and Na to display
        return

    # Set our default crop here so we can handle being called with
    # only the directory from reduce
    if SII_crop is None:
        SII_crop = "600x600"
    M_SII = MovieCorObs(SII_on_list,
                        speedup,
                        frame_rate,
                        SII_crop)
    M_Na = MovieCorObs(Na_on_list,
                       speedup,
                       frame_rate,
                       Na_crop)
    duration = np.max((M_SII.duration, M_Na.duration))
    SII_movie = mpy.VideoClip(M_SII.make_frame, duration=duration)
    Na_movie = mpy.VideoClip(M_Na.make_frame, duration=duration)
    datedir = os.path.split(directory)[1]
    # Needed to get full libmagick++-dev package and edit /etc/ImageMagick-6/policy.xml to comment out <policy domain="path" rights="none" pattern="@*" />
    # https://askubuntu.com/questions/873112/imagemagick-cannot-be-detected-by-moviepy
    txt = (mpy.TextClip('Io Input/Output Facility (IoIO)\n' + datedir,
                        font="Times-Roman", fontsize=15, color='white')
           .set_position(("center","top"))
           .set_duration(duration)
           .set_fps(M_SII.frame_rate))
    Na_SII_movie = mpy.CompositeVideoClip(
        [mpy.clips_array([[Na_movie, SII_movie]]), txt])
    Na_SII_movie.write_videofile(
        os.path.join(directory, "Na_SII.mp4"),
        fps=M_SII.frame_rate)
    # Since I am going for dual-display, no need to spend time writing these
    #SII_movie.write_videofile(os.path.join(directory, 
    #                                       "SII_movie.mp4"),
    #                          fps=M_SII.frame_rate)
    # Except I want this for the volcanic eruption paper
    Na_movie = mpy.CompositeVideoClip([Na_movie, txt])
    Na_movie.write_videofile(os.path.join(directory, 
                                          "Na_movie.mp4"),
                              fps=M_SII.frame_rate)
  
def movie_concatenate(directory):
    if directory is None:
        directory = os.path.join(data_root, 'reduced')
    clips = []
    Na_clips = []
    # --> eventually I want to have the data themselves indicate this
    filt_list = ['cloudy', 'marginal', 'dew', 'bad']
    for d in get_dirs(directory, filt_list=filt_list):
        try:
            c = mpy.VideoFileClip(os.path.join(d, 'Na_SII.mp4'))
        except:
            log.warning('Bad movie in ' + d)
            continue
        clips.append(c)
        try:
            c = mpy.VideoFileClip(os.path.join(d, 'Na_movie.mp4'))
        except:
            log.warning('Bad Na movie in ' + d)
            continue
        Na_clips.append(c)
    animation = mpy.concatenate_videoclips(clips)
    animation.write_videofile(os.path.join(directory, 'Na_SII.mp4'),
                              fps=global_frame_rate)
    animation = mpy.concatenate_videoclips(Na_clips)
    animation.write_videofile(os.path.join(directory, 'Na_movie.mp4'),
                              fps=global_frame_rate)

# https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
# Trying to use
class PoolWorker():
    """Get multiprocess to work with argparse.  Function is the function to call, iterable is a string indicating the argparse namespace element that will become the iterable for multiprocess, and args is the argparse args namespace"""
    def __init__(self, function, iterable, args):
        self.function = function
        self.iterable = iterable
        self.args = args
    def worker(self, arg):
        """Method used by multiprocess"""
        # Extract the dictionary from the argparse args namespace
        d = vars(self.args)
        # Insert the multiprocess-supplied arg into argparse args for
        # the function call
        d[self.iterable] = arg
        self.function(self.args)

def movie_cmd(args):
    if args.tree:
        top = args.directory
        if top is None:
            top = os.path.join(data_root, 'reduced')
        # remove --tree so our recursive call won't loop!  This is a
        # bit of a hack.  Really should have a MakeMovie object like
        # ReduceDir
        args.tree = None
        W = PoolWorker(movie_cmd, 'directory', args)
        with Pool(int(args.num_processes)) as p:
            p.map(W.worker, get_dirs(top))
        movie_concatenate(top)
        return
    if args.concatenate:
        movie_concatenate(args.directory)
        return
    assert args.directory is not None
    try:
        make_movie(args.directory,
                   recalculate=args.recalculate,
                   SII_crop=args.SII_crop,
                   Na_crop=args.Na_crop,
                   frame_rate=args.frame_rate,
                   speedup=args.speedup)
    except Exception as e:
        log.error(str(e) + ' skipping movie for ' + args.directory)

if __name__ == "__main__":
    # --> Figure out how to do this with a command-line switch that
    # --> works for everything
    #log.setLevel('DEBUG')
    parser = argparse.ArgumentParser(
        description="IoIO-related instrument image reduction")
    # --> Update this with final list once settled
    subparsers = parser.add_subparsers(dest='one of the subcommands in {}, above', help='sub-command help')
    subparsers.required = True

    ND_params_parser = subparsers.add_parser(
        'ND_params', help='Get ND_params from flats in a directory')
    ND_params_parser.add_argument(
        'directory', nargs='?', default='.', help='directory')
    ND_params_parser.add_argument(
        'maxcount', nargs='?', default=None,
        help='maximum number of flats to process -- median of parameters returned')
    ND_params_parser.set_defaults(func=cmd_get_default_ND_params)

    tree_parser =  subparsers.add_parser(
        'ND_params_tree', help='Find ND_params for all files in a directory tree')
    raw_data_root = os.path.join(data_root, 'raw')
    tree_parser.add_argument(
        'directory', nargs='?', default=raw_data_root, help='root of directory tree')
    tree_parser.set_defaults(func=ND_params_tree)

    reduce_parser = subparsers.add_parser(
        'reduce', help='Reduce files in a directory')
    reduce_parser.add_argument(
        '--tree', action='store_const', const=True,
        help='Reduce all data files in tree rooted at directory')
    reduce_parser.add_argument(
        '--num_processes', type=int, default=os.cpu_count()/threads_per_core,
        help='number of subprocesses for parallelization')
    reduce_parser.add_argument(
        '--default_ND_params', help='Default ND filter parameters to use')
    reduce_parser.add_argument(
        '--directory', help='Directory to process')
    reduce_parser.add_argument(
        '--recalculate', action='store_const', const=True,
        help='recalculate and overwrite files in reduced directory')
    reduce_parser.add_argument(
        'on_band', nargs='?', help='on-band filename')
    reduce_parser.add_argument(
        'off_band', nargs='?', help='off-band filename')
    reduce_parser.add_argument(
        '--movie', action='store_const', const=True,
        help="Create movie when done")
    reduce_parser.set_defaults(func=reduce_cmd)

    movie_parser = subparsers.add_parser(
        'movie', help='Makes a movie of the Io plasma torus')
    movie_parser.add_argument(
        '--recalculate', action='store_const', const=True,
        help='recalculate and overwrite files in reduced directory')
    movie_parser.add_argument(
        '--tree', action='store_const', const=True,
        help='Makes a movie out of all of the files in the tree (unless there is already a movie).')
    movie_parser.add_argument(
        '--concatenate', action='store_const', const=True,
        help='concatenate all movies in directories below directory (default is top-level reduced.  --recalculate is implied')
    movie_parser.add_argument(
        '--num_processes', type=int, default=os.cpu_count()/threads_per_core,
        help='number of subprocesses for parallelization')
    movie_parser.add_argument(
        '--speedup', type=int, help='Factor to speedup time.')
    movie_parser.add_argument(
        '--frame_rate', type=int, help='Video frame rate (frames/s).')
    movie_parser.add_argument(
        '--SII_crop', help='[SII] image crop, format: 600x300 (X by Y)')
    movie_parser.add_argument(
        '--Na_crop', help='Na image crop, format: 600x300 (X by Y)')
    movie_parser.add_argument(
        'directory', nargs='?', help='Directory to process')
    movie_parser.set_defaults(func=movie_cmd)

    # Final set of commands that makes argparse work
    args = parser.parse_args()
    # This check for func is not needed if I make subparsers.required = True
    if hasattr(args, 'func'):
        args.func(args)


#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_on-band_008.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_off-band_008.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_on-band_009.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_off-band_009.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_on-band_010.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_off-band_010.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_on-band_043.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'SII_off-band_043.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_on-band_011.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_off-band_011.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_on-band_001.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_off-band_001.fits')
#on_band = os.path.join(data_root, 'raw', '2018-03-28', 'SII_on-band_010.fits')
#off_band = os.path.join(data_root, 'raw', '2018-03-28', 'SII_off-band_010.fits')
#on_band = os.path.join(data_root, 'raw', '2018-05-20', 'SII_on-band_010.fits')
#off_band = os.path.join(data_root, 'raw', '2018-05-20', 'SII_off-band_010.fits')
#on_band = 'SII_on-band_010.fits'
#off_band = 'SII_off-band_010.fits'

#reduce_pair(on_band, off_band, recalculate=True)
#R = ReduceCorObs(on_band, off_band, recalculate=True)
#HDUList = get_HDUList(on_band)
#im = TiltImage(HDUList[0].data, .933, 16)
#plt.imshow(im)
#plt.show()
#from argparse import Namespace
#args = Namespace(directory='/data/io/IoIO/raw/2018-04-21',
#                 tree=None,
#                 default_ND_params=None)
#reduce_dir(args)
#print(get_astrometry_angle('2017-01-18T00:00:00'))
