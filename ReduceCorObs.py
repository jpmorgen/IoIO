#!/usr/bin/python3
import os
import re
import time
import argparse

import numpy as np
from scipy import ndimage
from skimage import exposure
from astropy import log
from astropy.time import Time, TimeDelta
#from astroquery.jplhorizons import Horizons
from jplhorizons import Horizons
#from photutils import CircularAperture, aperture_photometry

import matplotlib.pyplot as plt
import moviepy.editor as mpy
#from moviepy.video.fx.all import loop as mpyloop
#from moviepy.editor import VideoClip
#from moviepy.video.io.bindings import mplfig_to_npimage

from precisionguide import get_HDUList
from IoIO import CorObsData, run_level_default_ND_params
import define as D

# Constants for use in code
# These are the equavalent widths of the filters in angstroms
SII_eq_width = 9.95
Na_eq_width = 11.22
# There is a noticeable scattered light loss in the Na on-band filter.
# Aperture photometry didn't fix it, so this is my guess at the
# magnitude of the problem
SII_on_loss = 1
Na_on_loss = 0.8

background_light_threshold = 100 # ADU

def get_default_ND_params(directory='.', maxcount=None):
    """Derive default_ND_params from up to maxcount flats in directory
    """
    if not os.path.isdir(directory):
        raise ValueError("Specify directory to search for flats and derive default_ND_params")
    if maxcount is None:
        maxcount = 10

    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]

    flats = []
    for f in sorted(files):
        if 'flat' in f.lower():
            flats.append(os.path.join(directory, f))

    # Create default_ND_params out of the flats in this directory.
    ND_params_list = []
    # Just do 10 flats
    for fcount, f in enumerate(flats):
        try:
            # Iterate to get independent default_ND_params for
            # each flat
            default_ND_params = None
            for i in np.arange(3):
                F = CorObsData(f, default_ND_params=default_ND_params)
                default_ND_params = F.ND_params
        except ValueError as e:
            log.error('Skipping: ' + f + '. ' + str(e))
            continue
        ND_params_list.append(default_ND_params)
        if fcount >= maxcount:
            break
    if len(ND_params_list) == 0:
        raise ValueError('No good flats found in ' + directory)

    # If we made it here, we have a decent list of ND_params.  Now
    # take the median to create a really nice default_ND_params
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
            O = CorObsData(f, default_ND_params=default_ND_params)
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

data_root = '/data/io/IoIO'

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

class ReduceCorObs():
    """Do a quick reduction of on-band off-band coronagraph image pair"""

    def __init__(self,
                 OnBand_HDUList_im_or_fname=None,
                 OffBand_HDUList_im_or_fname=None,
                 default_ND_params=None,
                 NPang=None,
                 outfname=None,
                 overwrite=False):
        # Only bother with getting NPang (Jupiter's projected north
        # pole angle relative to celestial N in deg CCW) once per
        # directory, but do that with calling program's help
        self.NPang = NPang
        OnBand_HDUList = get_HDUList(OnBand_HDUList_im_or_fname)
        header = OnBand_HDUList[0].header
        OffBand_HDUList = get_HDUList(OffBand_HDUList_im_or_fname)
        # Use CorObsData to get basic properties like background level
        # and center.
        OnBandObsData = CorObsData(OnBand_HDUList,
                                   default_ND_params=default_ND_params)
        OffBandObsData = CorObsData(OffBand_HDUList,
                                    default_ND_params=default_ND_params)
        # --> eventually do proper bias subtraction
        # --> potentially check for high levels to reject bad files
        on_im = OnBand_HDUList[0].data - OnBandObsData.back_level
        on_back = np.mean(on_im)
        if on_back > background_light_threshold:
            log.warning('On-band background level too high: ' + str(on_back))
            return
        off_im = OffBand_HDUList[0].data - OffBandObsData.back_level
        off_back = np.mean(off_im)
        if off_back > background_light_threshold:
            log.warning('Off-band background level too high: ' + str(off_back))
            return
        header['BACKSUB'] = (OnBandObsData.back_level,
                             'background value subtracted')
        # --> Worry about flat-fielding later
        # Get ready to shift off-band image to match on-band image
        on_center = OnBandObsData.obj_center
        off_center = OffBandObsData.obj_center
        shift_off = on_center - off_center
        if np.linalg.norm(shift_off) > 5:
            log.warning('On- and off-band image centers are > 5 pixels apart')
            
        # on_jup and off_jup are the average brightness over 1 pixel.
        # Call that a pixel-averaged surface brightness.
        # --> Tried fancy surf_bright method and that didn't seem to
        # help Na over-subtraction problem  
        #surf_bright(on_im, on_center)
        #surf_bright(off_im, off_center)
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
            eq_width = SII_eq_width
            on_loss = SII_on_loss
        elif 'Na' in OnBandObsData.header['FILTER']:
            eq_width = Na_eq_width
            on_loss = Na_on_loss
        else:
            raise ValueError('Improper filter ' +
                             OnBandObsData.header['FILTER'])
        header['ON_LOSS'] \
            = (on_loss, 'on-band scat. light loss for discrete sources')
        scat_sub_im = on_im - off_im * on_jup/off_jup * on_loss
        header['OFFFNAME'] = (OffBand_HDUList.filename(),
                              'off-band file')
        header['OFFSCALE'] = (on_jup/off_jup, 'scale factor applied to off-band im')
        # Establish calibration in Rayleighs.  Brown & Schneider 1981
        # Jupiter is 5.6 MR/A
        MR = 5.6E6 * eq_width
        # 1000 is ND filter
        scat_sub_im = scat_sub_im / (on_jup * 1000) * MR
        # Put Jupiter back in.  Note this is MR, not MR/A
        scat_sub_im[OnBandObsData.ND_coords] \
            = on_im[OnBandObsData.ND_coords] / (on_jup * 1000) * MR
        header['BUNIT'] = ('rayleighs', 'pixel unit')

        # --> will want to check this earlier for proper pairing of
        # on-off band images
        pier_side = OnBandObsData.header.get('PIERSIDE')
        if pier_side is not None and pier_side == 'EAST':
            gem_flip = 180
        else:
            gem_flip = 0
            
        # Calculate NPang if we weren't passed it, but store it in
        # property so it can be used for the next file in this day
        if NPang is None:
            # V09 is the Moca observatory at Benson, which looks like
            # the San Pedro Valley observatory
            T = Time(header['DATE-OBS'], format='fits')
            jup = Horizons(id=599,
                           location='V09',
                           epochs=T.jd,
                           id_type='majorbody')
            NPang = jup.ephemerides()['NPang'].quantity.value[0]
            self.NPang = NPang
            # --> handle astrometry in a more general way.  -0.933 is for 2018
            # Save off original center of image for NDparams update, below
        o_center = np.asarray(scat_sub_im.shape)/2
        on_shift = o_center - OnBandObsData.obj_center
        # interpolation.rotate rotates CW for positive angle
        on_angle = -0.933 - NPang + gem_flip
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
        on_angle = abs(on_angle - gem_flip)
        ron_angle = np.radians(on_angle)
        # Offsets
        ND01 = header['NDPAR01']
        ND11 = header['NDPAR11']
        # Note on_shift is y,x
        xshift = np.dot(on_shift, np.asarray(((0,1))))
        ND01 += xshift
        ND11 += xshift
        ND01 = (o_center[0] * np.sin(ron_angle)
                + ND01 * np.cos(ron_angle))
        ND11 = (o_center[0] * np.sin(ron_angle)
                + ND11 * np.cos(ron_angle))
        header['NDPAR01'] = ND01
        header['NDPAR11'] = ND11
        # Angles
        ND00 = header['NDPAR00']
        ND10 = header['NDPAR10']
        header['NDPAR00'] = np.tan(np.arctan(ND00) - on_angle)
        header['NDPAR10'] = np.tan(np.arctan(ND10) - on_angle)

        # Coronagraph flips images N/S.  Transpose alert
        scat_sub_im =np.flipud(scat_sub_im)

        # Get ready to write
        OnBand_HDUList[0].data = scat_sub_im
        if outfname is not None:
            OnBand_HDUList.writeto(outfname)
            return
        rawfname = OnBand_HDUList.filename()
        if rawfname is None:
            log.warning('On-band image was not associated with any filename and outfname is not specified, writing to current directory, ReducedCorObs.fits')
            OnBand_HDUList.writeto('ReducedCorObs.fits')#, overwrite=True)
            return
        if not os.path.isabs(rawfname):
            log.warning("On-band image fname was not an absolute path and outfname is not specified.  I can't deconstruct the raw to reduced path structure, writing to current directory, ReducedCorObs.fits")
            OnBand_HDUList.writeto('ReducedCorObs.fits')#, overwrite=True)
            return
        # If we made it here, we have the normal case of writing to
        # the reduced directory tree
        basename = os.path.basename(rawfname)
        # Insert "r" so no collisions are possible
        (fbase, ext) = os.path.splitext(basename)
        redbasename = fbase + 'r' + ext
        rawdatepath = os.path.dirname(rawfname)
        datedir = os.path.split(rawdatepath)[1]
        red_data_root = os.path.join(data_root, 'reduced')
        if not os.path.exists(red_data_root):
            os.mkdir(red_data_root)
        reddir = os.path.join(data_root, 'reduced', datedir)
        if not os.path.exists(reddir):
            os.mkdir(reddir)
        redfname = os.path.join(reddir, redbasename)

        header['RVERSION'] = (0.0, 'Reduction version')
        OnBand_HDUList.writeto(redfname, overwrite=overwrite)

def reduce(args):
    if args.tree is not None:
        raise ValueError('Code not written yet')

    if not args.directory is None:
        # Collect file names
        files = [f for f in os.listdir(args.directory)
                 if os.path.isfile(os.path.join(args.directory, f))]

        # For now just put things together based on file names since IPT_Na_R
        SII_on_list = []
        SII_off_list = []
        Na_on_list = []
        Na_off_list = []
    
        SII_on = re.compile('^SII_on-band')
        SII_off = re.compile('^SII_off-band')
        Na_on = re.compile('^Na_on-band')
        Na_off = re.compile('^Na_off-band')
    
        for f in sorted(files):
            if SII_on.match(f):
                SII_on_list.append(os.path.join(args.directory, f))
            if SII_off.match(f):
                SII_off_list.append(os.path.join(args.directory, f))
            if Na_on.match(f):
                Na_on_list.append(os.path.join(args.directory, f))
            if Na_off.match(f):
                Na_off_list.append(os.path.join(args.directory, f))
    
        if len(SII_on_list) == 0 and len(Na_on_list) == 0:
            # Just return if we don't have any files
            return
    
        if args.default_ND_params is None:
            try:
                default_ND_params = get_default_ND_params(args.directory)
            except KeyboardInterrupt:
                # Allow C-C to interrupt
                raise
            except Exception as e:
                log.warning('Problem with flats in ' + args.directory + ': '
                            + str(e) + '.  using run_level_default_ND_params')
                default_ND_params = run_level_default_ND_params
    
        start = time.time()
        count = 0
        torus_count = 0
        Na_count = 0
        NPang = None
        n_SII = np.min((len(SII_on_list), len(SII_off_list)))
        for ip in np.arange(n_SII):
            log.debug(SII_on_list[ip], SII_off_list[ip])
            R = ReduceCorObs(SII_on_list[ip],
                             SII_off_list[ip],
                             NPang=NPang,
                             default_ND_params=default_ND_params,
                             overwrite=True)
            # ReduceCorObs stores NPang since it is expensive to get, but
            # only use it once per day, since it does change a bit
            NPang = R.NPang
        n_Na = np.min((len(Na_on_list), len(Na_off_list)))
        for ip in np.arange(n_Na):
            log.debug(Na_on_list[ip], Na_off_list[ip])
            R = ReduceCorObs(Na_on_list[ip],
                             Na_off_list[ip],
                             NPang=NPang,
                             default_ND_params=default_ND_params,
                             overwrite=True)
        elapsed = time.time() - start
        log.info('Elapsed time for ' + args.directory + ': ' + str(elapsed))
        log.info('Average per file: ' + str(elapsed/(n_SII+n_Na)))

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
            self.speedup = 24000
        if self.frame_rate is None:
            self.frame_rate=40
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
            T = Time(self.HDULcur[0].header['DATE-OBS'], format='fits')
            self.dt_next = (T - self.Tstart).sec
        else:
            self.HDUNext = None
            self.dt_next = self.dt_cur + self.HDULcur[0].header['EXPTIME']

    def make_frame(self, t):
        # Let us stick on frames before the calculated start time and
        # after the calculated duration
        m_dt = t * self.speedup
        while t > 0 and m_dt < self.dt_cur:
            self.prev_f()
        while t <= self.duration and m_dt >= self.dt_next:
            self.next_f()
        if self.persist_im is not None:
            return self.persist_im
        # If we made it here, we need to create our image
        im = self.HDULcur[0].data
        O = CorObsData(self.HDULcur)
        c = (np.asarray(im.shape)/2).astype(int)
        # Scale Jupiter down by 10 to get MR/A and 10 to get
        # it on comparable scale to torus
        im[O.ND_coords] = 100#im[O.ND_coords] 
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
        if self.filt == '[SII]':
            chop = 2000
        else:
            chop = 5000
        badc = np.where(np.logical_or(im < 0, im > chop))
        im[badc] = 0
        # Keep it linear for now -- this accentuated noise
        #im = exposure.equalize_adapthist(np.asarray(im/np.max(im)))
        # Scale for mp4
        im = im/np.max(im) * 255

        # mp4 wants to be 640 x 360 or 640 × 480
        # Note transpose space for C-style language
        scale = (np.round(im.shape / self.mp4shape)).astype(int)
        if np.any(scale > 1):
            scale = np.max(scale)
            im = ndimage.zoom(im, 1/scale, order=0)
            
        ## DEBUGGING
        #impl = plt.imshow(im, origin='lower',
        #                  cmap=plt.cm.gray, filternorm=0, interpolation='none')
        #plt.show()
        
        # Thanks to https://stackoverflow.com/questions/39463019/how-to-copy-numpy-array-value-into-higher-dimensions
        self.persist_im = np.stack((im,) * 3, axis=-1)
        return self.persist_im
               
def make_movie(args):
    if args.tree is not None:
        raise ValueError('Code not written yet')
    # Collect file names
    files = [f for f in os.listdir(args.directory)
             if os.path.isfile(os.path.join(args.directory, f))]
    SII_on_list = []
    Na_on_list = []

    SII_on = re.compile('^SII_on-band')
    Na_on = re.compile('^Na_on-band')

    for f in sorted(files):
        if SII_on.match(f):
            SII_on_list.append(os.path.join(args.directory, f))
        if Na_on.match(f):
            Na_on_list.append(os.path.join(args.directory, f))

    M_SII = MovieCorObs(SII_on_list,
                        args.speedup,
                        args.frame_rate,
                        args.crop)
    #SII_movie.write_gif(os.path.join(args.directory, 
    #                                 "SII_movie.gif"),
    #                    fps=M_SII.frame_rate)
    M_Na = MovieCorObs(Na_on_list,
                       args.speedup,
                       args.frame_rate)
    duration = np.max((M_SII.duration, M_Na.duration))
    SII_movie = mpy.VideoClip(M_SII.make_frame, duration=duration)
    SII_movie.write_videofile(os.path.join(args.directory, 
                                           "SII_movie.mp4"),
                              fps=M_SII.frame_rate)
    Na_movie = mpy.VideoClip(M_Na.make_frame, duration=duration)
    Na_movie.write_videofile(os.path.join(args.directory, 
                                           "Na_movie.mp4"),
                              fps=M_SII.frame_rate)
    Na_clip = mpy.VideoFileClip(os.path.join(args.directory, 
                                             "Na_movie.mp4"))
    SII_clip = mpy.VideoFileClip(
        os.path.join(args.directory,
                     "SII_movie.mp4")).resize(height=Na_clip.h)
    animation = mpy.clips_array([[Na_clip, SII_clip]])
    animation.write_videofile(os.path.join(args.directory,
                                     "Na_SII.mp4"),
                        fps=M_SII.frame_rate)
    #animation = animation.fx(mpy.vfx.loop)
    #animation = animation.fx(mpy.vfx.make_loopable, 0.5)
    #animation.write_videofile(os.path.join(args.directory,
    #                                 "Na_SII_loop.mp4"),
    #                          fps=M_SII.frame_rate)
  
if __name__ == "__main__":
    log.setLevel('DEBUG')
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
        '--default_ND_params', help='Default ND filter parameters to use')
    reduce_parser.add_argument(
        '--directory', help='Directory to process')
    reduce_parser.add_argument(
        'on_band', nargs='?', help='on-band filename')
    reduce_parser.add_argument(
        'off_band', nargs='?', help='off-band filename')
    reduce_parser.add_argument(
        '--NPang', help="Target's projected north pole angle relative to celestial N in deg CCW")
    reduce_parser.set_defaults(func=reduce)

    movie_parser = subparsers.add_parser(
        'movie', help='Makes a movie of the Io plasma torus')
    movie_parser.add_argument(
        '--tree', action='store_const', const=True,
        help='Makes a movie out of all of the files in the tree (unless there is already a movie).')
    movie_parser.add_argument(
        '--speedup', type=int, help='Factor to speedup time.')
    movie_parser.add_argument(
        '--frame_rate', type=int, help='Video frame rate (frames/s).')
    movie_parser.add_argument(
        '--crop', help='Format: 600x300 (X by Y)')
    movie_parser.add_argument(
        'directory', help='Directory to process')
    movie_parser.set_defaults(func=make_movie)

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
on_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_on-band_011.fits')
off_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_off-band_011.fits')
#on_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_on-band_001.fits')
#off_band = os.path.join(data_root, 'raw', '2018-04-21', 'Na_off-band_001.fits')
#on_band = 'SII_on-band_010.fits'
#off_band = 'SII_off-band_010.fits'
    
R = ReduceCorObs(on_band, off_band, overwrite=True)
#HDUList = get_HDUList(on_band)
#im = TiltImage(HDUList[0].data, .933, 16)
#plt.imshow(im)
#plt.show()
#from argparse import Namespace
#args = Namespace(directory='/data/io/IoIO/raw/2018-04-21',
#                 tree=None,
#                 default_ND_params=None)
#reduce_dir(args)
