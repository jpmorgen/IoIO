"""Adds astrometry capability to IoIO Photometry object, including
guaranteed WCS header

"""

import os
import glob
import warnings
import argparse
from random import randint
from time import sleep
from tempfile import TemporaryDirectory
import subprocess

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits import getheader
from astropy.table import Table, join, join_skycoord
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from astroquery.simbad import Simbad

from reproject.mosaicking import find_optimal_celestial_wcs

import ccdproc as ccdp

from bigmultipipe import outname_creator

from precisionguide import PGData

import IoIO.sx694 as sx694
from IoIO.utils import FITS_GLOB_LIST, multi_glob, savefig_overwrite
from IoIO.cordata import CorData
from IoIO.photometry import (SOLVE_TIMEOUT, rot_wcs,
                             Photometry, PhotometryArgparseMixin)
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBinnedOK, nd_filter_mask)
from IoIO.calibration import Calibration
from IoIO.horizons import GALSATS

MIN_SOURCES_TO_SOLVE = 5
MAX_SOURCES_TO_SOLVE = 500
ASTROMETRY_GLOB = '*_Astrometry'
KEYS_TO_SOURCE_TABLE = ['DATE-AVG',
                        ('DATE-AVG-UNCERTAINTY', u.s),
                        ('EXPTIME', u.s),
                        'FILTER',
                        'AIRMASS']

# We are going to save all our astrometry and photometry solutions to
# a centralized location
ASTROMETRY_ROOT = os.path.join(IoIO_ROOT, 'Astrometry')
PHOTOMETRY_ROOT = os.path.join(IoIO_ROOT, 'FieldStar')

# Proxy WCS settings.  
MIN_CENTER_QUALITY = 5

# Galsat mask for determining pierside via galsats
GALSAT_MASK_SIDE = 20 * u.pixel

# MaxIm provides output as below whenever a PinPoint solution is
# successful.  
#
# Plate solve for 4x4 binnning
# RA 15h 14m 53.1s,  Dec 02' 05.2"
# Pos Angle +57 15.6', FL 3920.1 mm, 0.96"/Pixel
#
# Starting 2018-01-17, the images on which the PinPoint plate solves
# were conducted were saved to disk right after there was a major
# instrument configuration change.  This was required for
# precisionguide.  Starting 2021-10-04, there was considerable
# instability in the instrument configuration until 2022-02-17.  This
# is during the period when the old focuser was being debugged and
# eventually replaced.  Astrometric solutions from comet and/or
# exoplanet obesrvations can be used for this period.
#
# Before 2018-01-17, the only record of astrometry is the PinPoint
# solution notes provided by MaxIm (e.g., above) and recorded in
# IoIO.notebk.  Thus, provision is made here to rotate the WCS in the
# first recorded plate solve through the POSITION_ANGLES list.  These
# angles are all cast into the ASCOM standard where PIERSIDE = WEST
# (looking east) is taken as the nominal north up east left
# configuration and a German Equatorial Mount (GEM) pier flip rotation
# of 180 degrees must be applied to PIERSIDE = EAST.  Note that here,
# position angle refers to the position of "up" in the MaxIm image
# relative to north in degrees CCW.  This is in the opposite sense
# that the FITS/WCS transformations in two ways.  (1) "up" is toward
# negative Y because MaxIm plots images with 0,0 in the upper left
# corner (2) FITS/WCS start with pixel coordinates and provide the
# values necessary to get to world (e.g. Equation 1 in Calabretta and
# Greisen, 2002) or the older formalism of Wells et al. (1981) that
# used CROTAn to specify rotation of the NAXISn axes onto the CTYPEn
# axes.  (1) is important to absolute WCS calculations but doesn't
# matter to rotations (2) is what we need to deal with here.
POSITION_ANGLES = [('2017-03-03', 57),
                   ('2017-03-04', 175-180),
                   ('2017-03-05', 358),
                   ('2017-03-07', 177-180),
                   ('2017-03-12', 181-180),
                   ('2017-03-13', 0),
                   ('2017-04-14', 0),
                   ('2017-04-17', 4),
                   ('2017-05-18', 178-180)]

# Because the telescope was not connected while the early IoIO
# shellout from ACP was operating, we don't know the pierside with
# absolute certainty at all times.  Our best guesses are entered in
# with cor_process.fix_obj_and_coord, but there is a deadband ~45
# minutes past the meridian labeled as PIERSIDE = UNKNOWN.  This
# motivates us to establish pierside with our astrometry solves
# wherever possible.  In general, the IoIO coronagraph was oriented
# the same way plus or minus ~20 degrees, depending on Jupiter's north
# pole angle.  Thus the cardinal directions are consistent, modulo the
# GEM pier flip.  Provide a one-time reference to the ASCOM standard
# non pier-flipped state
PIERSIDE_WEST_FILE = os.path.join(RAW_DATA_ROOT,
                                  '2018-04_Astrometry',
                                  'PinPointSolutionWestofPier.fit')

def read_wcs(fname):
    """Avoid annoying messages for WCSs not written by wcslib"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FITSFixedWarning)
        hdr = getheader(fname)
        return WCS(hdr)

def cardinal_directions(wcs):
    """Returns matrix of -1, 0, 1s indicating general pixel to world
    coordinate transformation directions (e.g., 1 indicates world axis
    goes in same sense as coordinate axis).  Minor axes are zeroed
    out, hence the 0s

    """
    # Avoid pass by reference bug by copying cd/pc to mat
    if wcs.wcs.has_cd():
        mat = wcs.wcs.cd.copy()
    elif wcs.wcs.has_pc():
        mat = wcs.wcs.pc.copy()
    else:
        raise ValueError('wcs.wcs does not have expected cd or pc '
                         'pixel to world transformation matrix')
    # Zero out the minor terms
    for irow in np.arange(2):
        if np.abs(mat[irow,0]) < np.abs(mat[irow,1]):
            mat[irow,0] = 0
        else:
            mat[irow,1] = 0
    return np.sign(mat)

PIERSIDE_WEST = cardinal_directions(read_wcs(PIERSIDE_WEST_FILE))

def pierside(wcs):
    cdwcs = cardinal_directions(wcs)
    if np.all(cdwcs == PIERSIDE_WEST):
        return 'WEST'
    if np.all(cdwcs == -PIERSIDE_WEST):
        return 'EAST'
    log.error(f'WCS solution not consistent with '
              f'IoIO coronagraph {wcs}')
    return 'ERROR'

def pierflip(wcs):
    """Pierflip rotates the FOV by 180 deg"""
    return rot_wcs(wcs, 180*u.deg)

def mask_galsats(ccd_in, galsat_mask_side=GALSAT_MASK_SIDE, **kwargs):
    assert galsat_mask_side.unit == u.pixel
    ccd = ccd_in.copy()
    galsat_mask = np.zeros_like(ccd.mask)
    galsats = list(GALSATS.keys())
    galsats = galsats[1:]
    for g in galsats:
        ra = ccd.meta[f'{g}_RA']
        dec = ccd.meta[f'{g}_DEC']
        sc = SkyCoord(ra, dec, unit=u.deg)
        try:
            pix = ccd.wcs.world_to_array_index(sc)
            hs = int(round(galsat_mask_side.value/2))
            bot = pix[0] - hs
            top = pix[0] + hs
            left = pix[1] - hs
            right = pix[1] + hs
            galsat_mask[bot:top, left:right] = True
        except Exception as e:
            # The wcs method may fail or the array index might fail.
            # Not sure yet would the Exception would be.  Either way,
            # we just want to skip it
            log.debug(f'Problem masking galsat: {e}')
            continue
        ccd.mask = np.ma.mask_or(ccd.mask, galsat_mask)
    return ccd

def sum_ccddata(ccd):
    """Annoying that sum and np.ma.sum don't work out of the box"""
    a = np.ma.array(ccd.data, mask=ccd.mask)
    return np.ma.sum(a)

def sum_galsat_positions(ccd_in, wcs):
    ccd = ccd_in.copy()
    ccd.wcs = wcs
    ccd = mask_galsats(ccd)
    ccd.mask = ~ccd.mask
    return sum_ccddata(ccd)
    
def pierside_from_galsats(ccd_in, wcs):
    """Use alignment of Galilean satellites to determine pierflip.  NOTE:
    this does not (yet) include any check for cases where the sums may
    be close

    Parameters
    ----------
    ccd_in : ccd

    wcs : WCS
        wcs centered on ccd

    Returns
    -------
    pierside : str

    """
    sum_no_flip = sum_galsat_positions(ccd_in, wcs)
    sum_with_flip = sum_galsat_positions(ccd_in, pierflip(wcs))
    ratio = sum_no_flip / sum_with_flip
    if 0.5 < ratio and ratio < 2:
        rawfname = ccd_in.meta['RAWFNAME']
        log.warning(f'PIERSIDE not determined from galsats, '
                    f'no_flip/flip ratio = {ratio}: {rawfname}')
        return 'UNKNOWN'
    #print(f'sum_no_flip = {sum_no_flip}, sum_with_flip = {sum_with_flip}')
    if sum_with_flip > sum_no_flip:
        return pierside(pierflip(wcs))
    return pierside(wcs)

def astrometry_outname(fname, date_obs):
    """Returns full pathname to centralized location into which a
    plate-solved FITS header will be written.

    """
    bname = os.path.basename(fname)
    broot, _ = os.path.splitext(bname)
    outdir = os.path.join(ASTROMETRY_ROOT, date_obs)
    outname = os.path.join(outdir, broot+'_wcs.fits')
    return outname

def astrometry_outname_as_outname(ccd, bmp_meta=None,
                                  in_name=None,
                                  photometry=None,
                                  **kwargs):
    """Cormultipipe postprocessing routine that enables FITS file to be
    written to ASTROMETRY_ROOT system

    """
    date, _ = ccd.meta['DATE-OBS'].split('T')
    outname = astrometry_outname(in_name, date)
    bmp_meta['outname'] = outname
    return ccd

class CorPhotometry(Photometry):
    """Subclass of Photometry to deal with specifics of IoIO coronagraph
    parameters and local astrometry.net implementation

        Parameters
        ----------
        ccd : CCDData

        outname : str or None
            Presumed name of FITS file that is being plate solved.
            The file need not exist yet, since we are passing the
            already extracted star positions to solve-field.  The
            extension will be removed and the rest of the name used
            for the solve-field input and output.  If `None`, a
            temporary directory is created and files are created
            there.

        solve_timeout : float
            Plate solve time limit (seconds)

        keys_to_source_table : list
            FITS header keys to add as columns to wide_source_table

    """
    def __init__(self,
                 ccd=None,
                 outname=None,
                 rawdir=RAW_DATA_ROOT,
                 fits_glob_list=FITS_GLOB_LIST,
                 solve_timeout=SOLVE_TIMEOUT,
                 solve_by_proxy=True,
                 min_center_quality=MIN_CENTER_QUALITY,
                 keys_to_source_table=KEYS_TO_SOURCE_TABLE,
                 **kwargs):
        super().__init__(ccd=ccd,
                         keys_to_source_table=keys_to_source_table,
                         **kwargs)
        self.outname = outname
        self.rawdir = rawdir
        self.fits_glob_list = fits_glob_list
        self.solve_timeout = solve_timeout
        self.solve_by_proxy = solve_by_proxy
        self.min_center_quality = min_center_quality
        self._raw_wcs_flist = None
        # We have to do this on instantiation or else we get
        # AssertionError: daemonic processes are not allowed to have children
        # when its pipeline runs while we are in a pipeline process
        self.reduce_base_astrometry()

    def init_calc(self):
        super().init_calc()
        self._proxy_wcs_file = None
        self._pierside = None
        self._rdls_table = None
        self._gaia_table = None
        self._source_gaia_join = None

    @property
    def proxy_wcs_file(self):
        if self._proxy_wcs_file is not None:
            return self._proxy_wcs_file
        self.wcs
        return self._proxy_wcs_file

    @property
    def pierside(self):
        """Property used to get pierside out of photometry into ccd in
        add_astrometry

        """
        if self.ccd is not None:
            self._pierside = self.ccd.meta.get('pierside')
        return self._pierside

    @property
    def raw_wcs_flist(self):
        """List of main camera image files in *_Astrometry directories for
        which PinPoint was run

        """
        if self._raw_wcs_flist is not None:
            return self._raw_wcs_flist

        dirlist = glob.glob(os.path.join(self.rawdir, ASTROMETRY_GLOB))
        dirlist.sort()
        rwfl = []
        for d in dirlist:
            flist = multi_glob(d, self.fits_glob_list)
            for f in flist:
                wcs = read_wcs(f)
                # Reject images with no WCS (some CorData experiments were
                # stashed in the _Astrometry directory).  WCS makes this hard,
                # since ctype is not quite a list of str.
                if (wcs.wcs.ctype[0] == ''
                    and wcs.wcs.ctype[1] == ''):
                    continue
                # Reject guide camera images
                main_naxis = np.asarray((sx694.naxis1, sx694.naxis2))
                binning, mod = np.divmod(main_naxis, wcs._naxis)
                if np.any(mod != 0):
                    continue
                rwfl.append((f, wcs))
        self._raw_wcs_flist = rwfl
        return(rwfl)

    def list_unreduced_astrometry_fnames(self):
        to_reduce = []
        for f, wcs in self.raw_wcs_flist:
            date, _ = wcs.wcs.dateobs.split('T')
            outname = astrometry_outname(f, date)
            if os.path.exists(outname):
                continue
            to_reduce.append(f)
        return to_reduce
    
    def reduce_base_astrometry(self):
        """Reduce the astrometry files that were recorded each time the
        instrument configuration was changed

        """
        flist = self.list_unreduced_astrometry_fnames()
        if len(flist) == 0:
            return
        calibration = Calibration(reduce=True)
        # Make sure original solutions stands.  Note we have to do
        # this in the photometry we pass to cmp, since add_astrometry
        # doesn't handle it
        self.solve_by_proxy = False
        cmp = CorMultiPipeBinnedOK(
            calibration=calibration,
            auto=True,
            photometry=self,
            solve_timeout=600, # Try to solve all [didn't help]
            mask_ND_before_astrometry=True, 
            create_outdir=True,
            post_process_list=[add_astrometry,
                               astrometry_outname_as_outname])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FITSFixedWarning)
            #pout = cmp.pipeline([flist[1]], overwrite=True)#, outdir='/tmp', outname='/tmp/astrometry.fits')
            pout = cmp.pipeline(flist, overwrite=True)

    def astrometry_best_dir(self, date_obs):
        """This is going to be at least the first of our base_astrometry
        directories but may be earlier if astormetry solutions have
        filtered into ASTROMETRY_ROOT.  The best dir is the same as
        our date_obs, but may be before.  At worst, it is the earliest
        dir and we use position_angle to rotate the WCS.

        """
        dirlist = os.listdir(ASTROMETRY_ROOT)
        dirlist.sort()
        if date_obs < dirlist[0]:
            return os.path.join(ASTROMETRY_ROOT, dirlist[0])
        for id, d in enumerate(dirlist):
            if d == date_obs:
                return os.path.join(ASTROMETRY_ROOT, dirlist[id])
            if d > date_obs:
                return os.path.join(ASTROMETRY_ROOT, dirlist[id-1])
        return os.path.join(ASTROMETRY_ROOT, dirlist[-1])

    def position_angle(self, date_obs):
        """Return position angle of north on date_obs (str).  As described
        above, this is the negative of the rotation that WCS does to
        translate pixels to world

        """
        earliest, _ = POSITION_ANGLES[0]
        _, earlist_raw_wcs = self.raw_wcs_flist[0]
        #for wcs in self.wcs_list: print(wcs.wcs.dateobs)
        if date_obs < earliest or earlist_raw_wcs.dateobs < date_obs:
            raise ValueError(date_obs + ' is either before first IoIO observation or after the first WCS solution')
        for iua, d_ua in enumerate(POSITION_ANGLES):
            d, _ = d_ua
            if d > date_obs:
                _, angle= POSITION_ANGLES[iua-1]
                return angle
        _, angle= POSITION_ANGLES[-1]
        return angle

    def match_wcs_scale_and_cr_to_ccd(self, wcs_file):
        """Helper method that is called more than once.  NOTE: This does not
        handle pierflip

        Parameters
        ----------
        wcs_file : str
            full path to proxy wcs file

        Returns
        -------
        (wcs, wcs_hdr) : tuple
            wcs with scale matched to ccd and wcs FITS header

        """

        try:
            wcs_hdr = getheader(wcs_file)
        except OSError as e:
            rawfname = self.ccd.meta.get('RAWFNAME')
            log.warning(f'Likely process collision.  Trying again to read '
                        f'wcs_file {wcs_file} '
                        f'for {rawfname}: {e}')
            sleep(randint(1,5))
            return self.match_wcs_scale_and_cr_to_ccd(wcs_file)
        wcs = WCS(wcs_hdr)

        # Scale WCS transformations to match ccd pixel size.  All of
        # our Astrometry solutions have run through the astropy wcslib
        # stuff, which translates older CD matrices to PC-style.
        # https://stackoverflow.com/questions/36384760/transforming-a-row-vector-into-a-column-vector-in-numpy
        # Calabretta and Greisen (2002) define the PC
        ccd_xy_binning = self.ccd.binning[::-1]
        wcs_xy_binning = np.asarray((wcs_hdr['xbinning'],
                                     wcs_hdr['ybinning']))

        wcs_to_ccd_scale = ccd_xy_binning / wcs_xy_binning
        wcs_to_ccd_scale_col = wcs_to_ccd_scale[..., None]
        wcs.wcs.pc *= wcs_to_ccd_scale_col
        #wcs._naxis /= wcs_to_ccd_scale
        #wcs.wcs.cdelt *= wcs_to_ccd_scale

        # SIP distortion is present when astrometry.net code has
        # solved the proxy image, but I find that SIP is not
        # consistent across pier flips, so just turn it off.  Turning
        # SIP off also makes it easier to deal with crpix and crval,
        # since SIP is referenced to a particular crpix which may not
        # be obj_center+1.  That means that if we were to propagate
        # SIP to a new WCS, we would have to somehow iteratively find
        # the crval that resulted in OBJCTRA,OBJCTDEC at obj_center.
        # Shupe et al. (2005) define the SIP
        wcs.sip = None

        #if wcs.has_distortion:
        #    for iu in np.arange(wcs.sip.a_order):
        #        for iv in np.arange(wcs.sip.a_order):
        #            if iu + iv > wcs.sip.a_order:
        #                continue
        #            # This is a little dangerous for anything but an
        #            # anti-diagonal matrix and equal binning in both
        #            # axes.  But all our binning are always equal in
        #            # both axes and a spot-check of matrices suggests
        #            # they all anti-diagonal, so don't worry about it
        #            wcs_wcs.sip.a[iu, iv] *= (
        #                wcs_to_ccd_scale[0]**iu
        #                * wcs_to_ccd_scale[1]**iv)

        # Point the WCS to our object.  Note C ordering and zero
        # reference of obj_center.  CRPIX is FORTRAN order and 1
        # reference
        wcs.wcs.crval = np.asarray((self.ccd.sky_coord.ra.deg,
                                    self.ccd.sky_coord.dec.deg))
        wcs.wcs.crpix = self.ccd.obj_center[::-1] + 1
        wcs.wcs.dateobs = self.ccd.meta['DATE-OBS']
        wcs.wcs.mjdobs = self.ccd.meta['MJD-OBS']
        wcs.wcs.dateavg = self.ccd.meta['DATE-AVG']
        wcs.wcs.mjdavg = self.ccd.meta['MJD-AVG']
        return wcs, wcs_hdr

    @property
    def proxy_wcs(self):
        if (self.ccd.center_quality < self.min_center_quality
            or self.ccd.meta.get('OBJECT_TO_OBJCTRADEC') is None):
            # There is not much we can do in these cases -- we need to
            # know where our object is in both pixels and RA and DEC
            self._solved = False
            return None
        
        # First, find the closest astrometry directory before or equal
        # to our observation.  Below, we handle the case where the
        # directory is after our observation.
        date_obs = self.ccd.meta['date-obs']
        best_dir = self.astrometry_best_dir(date_obs)
        wcs_collect = ccdp.ImageFileCollection(
            best_dir, keywords=['date-obs',
                                'pierside',
                                'xbinning',
                                'ybinning'])

        # See if we can match binning, though it is not the end of the
        # world if we can't
        ccd_xy_binning = self.ccd.binning[::-1]
        try:
            # The way filter works, this is the best we can do to
            # guard against not finding our target
            wcs_collect = wcs_collect.filter(
                xbinning=ccd_xy_binning[0],
                ybinning=ccd_xy_binning[1])
        except FileNotFoundError:
            wcs_collect = wcs_collect
        except TypeError:
            rawfname = self.ccd.meta.get('RAWFNAME')
            log.warning(f'Likely process collision.  '
                        f'Trying again to read collection from '
                        f'wcs_file {best_dir} '
                        f'for {rawfname}: {e}')
            sleep(randint(1,5))
            return self.proxy_wcs

        # Fix unknown pierside issue using cute trick with Galsats
        if self.ccd.meta['pierside'] == 'UNKNOWN':
            # Read any old WCS as our first-pass WCS with which to
            # check pierside
            twcs_file = wcs_collect.files_filtered(
                include_path=True)[0]
            wcs, _ = self.match_wcs_scale_and_cr_to_ccd(twcs_file)
            self.ccd.meta['pierside'] = pierside_from_galsats(self.ccd, wcs)
            if self.ccd.meta['pierside'] == 'UNKNOWN':
                # We failed in our galsat attempt, which is hopefully
                # rare, so don't work harder at trying to succeed.
                # Log message is produce in pierside_from_galsats
                self._solved = False
                return None

        ccd_pierside = self.ccd.meta['pierside']
        # Match PIERSIDE, if possible
        try:
            wcs_collect = wcs_collect.filter(
                pierside=ccd_pierside)
        except FileNotFoundError:
            wcs_collect = wcs_collect

        # Get our best wcs and scale and center it
        wcs_tobs = Time(wcs_collect.values('date-obs'))
        tobs = Time(date_obs)
        dts = tobs - wcs_tobs
        ibest = np.argmin(np.abs(dts))
        best_wcs_file = wcs_collect.files_filtered(
            include_path=True)[ibest]
        wcs, phdr = self.match_wcs_scale_and_cr_to_ccd(best_wcs_file)

        # Handle pier flip
        _, earlist_raw_wcs = self.raw_wcs_flist[0]
        if (wcs.wcs.dateobs == earlist_raw_wcs.wcs.dateobs
            and date_obs < wcs.wcs.dateobs):
            # Observation is before earliest PinPoint solution and no
            # astrometry solutions have been recorded.  We need to
            # rotate by -position_angle.  Note, this may pierflip the
            # astrometry
            wcs, _ = find_optimal_celestial_wcs([(self.ccd.data, wcs)])
            wcs = rot_wcs(wcs, -self.position_angle(date_obs))
            wcs_pierside = pierside(wcs)
        else:
            # In the case that pierside is unknown
            wcs_pierside = phdr['pierside']
            
        if ccd_pierside != 'EAST' and ccd_pierside != 'WEST':
            log.error(f'Unexpected ccd PIERSIDE = {ccd_pierside}')
            flip = False
        elif wcs_pierside != 'EAST' and wcs_pierside != 'WEST':
            log.error(f'Unexpected WCS PIERSIDE = {wcs_pierside}')
            flip = False
        elif ccd_pierside == wcs_pierside:
            flip = False
        else:
            flip = True
        if flip:
            wcs = pierflip(wcs)

        # Stash ccd pierside in property for add_astrometry
        self._pierside = ccd_pierside

        # Reference original WCS file in our header if this is a chain
        # of proxies
        phdr = getheader(best_wcs_file)
        proxy_wcs_file = phdr.get('PROXYWCS')
        if proxy_wcs_file:
            self._proxy_wcs_file = proxy_wcs_file
        else:
            self._proxy_wcs_file = best_wcs_file
        
        self._solved = True
        return wcs        

    @property
    def wcs(self):
        """Plate solve ccd image, setting ccd's WCS object

        Returns
        -------
        wcs : `~astropy.wcs.WCS` or None
            ``None`` is returned if astrometry fails

        """
        if self._wcs is not None:
            return self._wcs
        if (self.source_catalog is None
            or len(self.source_catalog) < MIN_SOURCES_TO_SOLVE):
            return self.proxy_wcs

        # Ack.  Astropy is not making this easy.  Writing the QTable
        # fails entirely, possibly because it has mixed Quantity and
        # non-Quantity columns (has_mixin_columns).  Error has
        # something about not being able to represent an object.  When
        # I convert to Table and try to save that, I get lots of
        # WARNINGS about units not being able to round-trip, even
        # though they can round trip if that is the only type of unit
        # in the table.  Sigh.  Save off a separate Table with just
        # the quantities we are interested in, since we throw it away
        # anyway
        xyls_table = self.source_table['xcentroid', 'ycentroid']
        # This assumes that source_table is sorted by reverse segment_flux
        last_source = np.min((len(xyls_table), MAX_SOURCES_TO_SOLVE))
        xyls_table = xyls_table[0:last_source]

        ra = self.ccd.sky_coord.ra.to_string(unit=u.deg)
        dec = self.ccd.sky_coord.dec.to_string(unit=u.deg)
        ccd_shape = np.asarray(self.ccd.shape)
        pdiameter = np.average(ccd_shape) * u.pixel
        # PIXSCALE is binned
        pixscale = self.ccd.meta.get('PIXSCALE')
        if pixscale is None:
            # failsafe that should not happen if sx694.metadata has run
            # Assumes full-frame
            diameter = 1*u.deg
            pixscale = diameter/pdiameter
            pixscale = pixscale.to(u.arcsec/u.pixel)
            pixscale = pixscale.value
        else:
            diameter = pdiameter * pixscale * u.arcsec/u.pixel
        diameter = diameter.to(u.deg)
        if (self.ccd.meta.get('OBJECT_TO_OBJCTRADEC')
            and isinstance(self.ccd, PGData)
            and self.ccd.center_quality > 5):
            # Reference CRPIX* to object's PGData center pixel
            # Note obj_center is python ordering and zero centered
            c = self.ccd.obj_center + 1
            crpix_str = f'--crpix-x {c[1]} --crpix-y {c[0]}'
        else:
            crpix_str = ''

        if self.solve_timeout is not None:
            cpulimit_str = f'--cpulimit {self.solve_timeout}'
        else:
            cpulimit_str = ''

        # Search radius has to be surprisingly large for this to work
        # with the IoIO Astrometry calibration data for some reason.
        # 2 worked, but make it 4 for good measure
        search_radius = diameter.value*4

        # Trying odds-to-solve and odds-to-reject didn't help
        # f'--odds-to-solve 1e2 --odds-to-reject 1e-200 '\
        # Trying bigger diameter and more pix scale
        # f'--scale-low {pixscale*0.5:.2f} '\
        # f'--scale-high {pixscale*2:.2f} --scale-units arcsecperpix ' \

        # f'--scale-low {pixscale*0.8:.2f} '\
        # f'--scale-high {pixscale*1.2:.2f} --scale-units arcsecperpix ' \
        astrometry_command = \
            f'solve-field --x-column xcentroid --y-column ycentroid ' \
            f'--ra {ra} --dec {dec} --radius {search_radius:.2f} ' \
            f'--width {ccd_shape[1]:.0f} --height {ccd_shape[0]:.0f} ' \
            f'--scale-low {pixscale*0.8:.2f} '\
            f'--scale-high {pixscale*1.2:.2f} --scale-units arcsecperpix ' \
            f'{crpix_str} {cpulimit_str} ' \
            f'--tag-all --no-plots --overwrite '
        with TemporaryDirectory(
                prefix='IoIO_photometry_plate_solve_') as tdir:
            # We only use the tempdir when there is no filename, but
            # it does no harm to create and delete it
            outname = self.outname
            if outname is None:
                # This file does not need to exist
                outname = os.path.join(tdir, 'temp.fits')
            outroot, _ = os.path.splitext(outname)
            xyls_file = outroot + '.xyls'
            astrometry_command += xyls_file
            xyls_table.write(xyls_file, format='fits', overwrite=True)
            p = subprocess.run(astrometry_command, shell=True,
                               capture_output=True)
            self.solve_field_stdout = p.stdout
            self.solve_field_stderr = p.stderr
            if os.path.isfile(outroot+'.solved'):
                wcs = WCS(outroot + '.wcs')
                self._pierside = pierside(wcs)
                # The astrometry.net code or CCDData.read seems to set
                # RADESYS = FK5 deep in the bowls of wcs so nothing I
                # do affects it
                self._rdls_table = Table.read(outroot + '.rdls')
                #self.source_table.show_in_browser()
                self._solved = True
                log.debug(f'SOLVED {self.ccd.meta["rawfname"]}')
            elif not self.solve_by_proxy:
                # For the Astrometry base files just in case one
                # doesn't get solved.  Initial trouble I had with
                # images not getting solved has been fixed
                # PinPoint WCS does get read, so might as well put it
                # in our object, since this gives us a way to get
                # astrometry in a pinch
                wcs = self.ccd.wcs
                self._pierside = pierside(wcs)
                self._solved = False
                log.debug(f'NOT SOLVED {self.ccd.meta["rawfname"]}')
            else:
                wcs = self.proxy_wcs
                log.debug(f'NOT SOLVED {self.ccd.meta["rawfname"]}')
            self._wcs = wcs
            return self._wcs

    @property
    def rdls_table(self):
        """astrometry.net rdls table from the locally downloaded 5200 HEAVY
        index files.  This will get supplemented as needed

        """
        if self._rdls_table is not None:
            return self._rdls_table
        if self.solved:
            return self._rdls_table
        return None

    @property
    def source_table_has_object(self):
        if not self.source_table_has_coord:
            return False
        obj = self.ccd.meta.get('OBJECT')
        objctra = self.ccd.meta.get('OBJCTRA')
        objctdec = self.ccd.meta.get('OBJCTDEC')
        obj_coord = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
        dobj = obj_coord.separation(self.source_table['coord'])
        dobj = dobj.to(u.arcsec)
        idx = np.argmin(dobj)
        # String column ends up being of fixed width
        objs = np.full(len(self.source_table),
                       fill_value=' '*len(obj))
        objs[idx] = obj
        self.source_table['OBJECT'] = objs
        self.source_table.meta['OBJECT'] = 'Object name'
        self.source_table['DOBJ'] = dobj
        self.source_table.meta['DOBJ'] = \
            f'Distance to primary object {dobj.unit}'
        return True

    @property
    def wide_source_table(self):
        if (self.source_table_has_coord
            and self.source_table_has_object
            and self.source_table_has_key_cols):
            return self.source_table
        return None    

    @property
    def rdls_table_has_coord(self):
        if 'coord' in self.rdls_table.colnames:
            return True
        if not self.solved:
            return False
        skies = SkyCoord(
            self.rdls_table['RA'],
            self.rdls_table['DEC'])
        # Prefer skycoord representation
        del self.rdls_table['RA', 'DEC']
        self.rdls_table['coord'] = skies
        return True

    @property
    def wide_rdls_table(self):
        if self.rdls_table_has_coord:
            return self.rdls_table
        return None

    @property
    def source_gaia_join(self):
        if self._source_gaia_join is not None:
            return self._source_gaia_join
        if self.wide_source_table is None:
            return None
        if self.rdls_table is None:
            return None
        j = join(
            self.wide_source_table, self.wide_rdls_table,
            join_type='inner',
            keys='coord',
            table_names=['source', 'gaia'],
            uniq_col_name='{table_name}_{col_name}',
            join_funcs={'coord': join_skycoord(self.join_tolerance)})
        self._source_gaia_join = j
        return self._source_gaia_join

    def plot_object(self, outname=None, expand_bbox=10, show=False, **kwargs):
        """Plot primary object

        """
        # Don't gunk up the source_table ecsv with bounding box
        # stuff, but keep in mind we sorted it, so key off of
        # label
        bbox_table = self.source_catalog.to_table(
            ['label', 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'])
        mask = (self.wide_source_table['OBJECT']
                == self.ccd.meta['OBJECT'])
        label = self.wide_source_table[mask]['label']
        bbts = bbox_table[bbox_table['label'] == label]
        # This assume that the source is not on the edge of the ccd
        xmin = bbts['bbox_xmin'][0] - expand_bbox
        xmax = bbts['bbox_xmax'][0] + expand_bbox
        ymin = bbts['bbox_ymin'][0] - expand_bbox
        ymax = bbts['bbox_ymax'][0] + expand_bbox
        source_ccd = self.ccd[ymin:ymax, xmin:xmax]
        threshold = self.threshold[ymin:ymax, xmin:xmax]
        segm = self.segm_image.data[ymin:ymax, xmin:xmax]
        back = self.background[ymin:ymax, xmin:xmax]
        # Take median back_rms for plotting purposes
        back_rms = np.median(self.back_rms[ymin:ymax, xmin:xmax])
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
        ax.contour(source_ccd - back,
                   levels=np.arange(2,11)*back_rms, colors='gray')
        ax.contour(source_ccd - threshold,
                   levels=0, colors='green')
        ax.set_title(f'{self.ccd.meta["OBJECT"]} {self.ccd.meta["DATE-AVG"]}')
        ax.text(0.1, 0.9, f'back_rms_scale = {self.back_rms_scale}',
                color='white', transform=ax.transAxes)
        if show:
            plt.show()
        if outname is not None:
            savefig_overwrite(outname)
        plt.close()

def object_to_objctradec(ccd_in, **kwargs):
    """cormultipipe post-processing routine to query Simbad for RA and DEC

    """    
    ccd = ccd_in.copy()
    s = Simbad()
    obj = ccd.meta['OBJECT']
    simbad_results = s.query_object(obj)
    if simbad_results is None:
        # --> This is where I want to emulate simbad results for
        # Qatars or add abstraction layer for Simbad to do so

        # Don't fail, since OBJT* are within pointing errors
        log.warning(f'Simbad did not resolve: {obj}, relying on '
                    f'OBJCTRA = {ccd.meta["OBJCTRA"]} '
                    f'OBJCTDEC = {ccd.meta["OBJCTDEC"]}')
        return ccd
    obj_entry = simbad_results[0]
    ra = Angle(obj_entry['RA'], unit=u.hour)
    dec = Angle(obj_entry['DEC'], unit=u.deg)
    ccd.meta['OBJCTRA'] = (ra.to_string(),
                      '[hms J2000] Target right assention')
    ccd.meta['OBJCTDEC'] = (dec.to_string(),
                       '[dms J2000] Target declination')
    ccd.meta.insert('OBJCTDEC',
                    ('HIERARCH OBJECT_TO_OBJCTRADEC', True,
                     'OBJCT* point to OBJECT'),
                    after=True)
    # Reset ccd.sky_coord, just in case it has been used before now
    # (e.g. in cor_process)
    ccd.sky_coord = None
    return ccd

def write_astrometry(ccd_in, in_name=None, outname=None,
                     write_to_central_astrometry=True,
                     **kwargs):
    # Save WCS into centralized directory before we do any subsequent
    # rotations.  This also effectively caches our astrometry
    # solutions!  We have to copy our current ccd and send that to an
    # HDUList, otherwise the wcs won't get into the metadata.  Unlike
    # write_photometry, we don't expect a local version of this, since
    # the WCS solutions are saved with the primary FITS output
    if not write_to_central_astrometry:
        return ccd_in
    ccd = ccd_in.copy()
    date, _ = ccd.meta['DATE-OBS'].split('T')
    aoutname = astrometry_outname(in_name, date)
    ccd.mask = None
    ccd.uncertainty = None
    # ccd.to_hdu needs at least some data to work
    hdul = ccd.to_hdu()
    hdul[0].data = None
    d = os.path.dirname(aoutname)
    os.makedirs(d, exist_ok=True)
    hdul.writeto(aoutname, overwrite=True)
    return

def write_photometry(in_name=None, outname=None, photometry=None,
                     create_outdir=False,
                     write_proxy_photometry=False,
                     write_wide_source_table=False,
                     write_source_gaia_join=False,
                     **kwargs):
    # CorPhotometry.wide_source_table requires we have coordinates
    # (which we need) and a valid OBJECT and other columns, which
    # we don't technically need at this point.  But there are not
    # currently [m?]any cases we care about that would fail to
    # have OBJECT, etc.  We certainly need the coordinates, so
    # just go with this.
    if (photometry is None
        or not (photometry.solved or write_proxy_photometry)
        or photometry.wide_source_table is None
        or not (write_wide_source_table
                or write_source_gaia_join)):
        return

    photometry.wide_source_table['in_name'] = in_name
    photometry.wide_source_table['outname'] = outname
    if create_outdir:
        os.makedirs(os.path.dirname(outname), exist_ok=True)
    oroot, _ = os.path.splitext(outname)
    oname = oroot + '.ecsv'
    if write_wide_source_table:
        photometry.wide_source_table.write(
            oname, delimiter=',', overwrite=True)
    if (photometry.source_gaia_join is None
        or not write_source_gaia_join):
        return
    gname = oroot + '_gaia.ecsv'
    photometry.source_gaia_join.write(
        gname, delimiter=',', overwrite=True)

def add_astrometry(ccd_in, bmp_meta=None,
                   photometry=None,
                   mask_ND_before_astrometry=False,
                   in_name=None,
                   outdir=None,
                   create_outdir=None,
                   solve_timeout=None,
                   keep_intermediate=False,
                   write_to_central_photometry=True,
                   write_local_photometry=False,
                   **kwargs):
    """cormultipipe post-processing routine to add wcs to ccd

    Parameters
    ----------
    ccd_in

    photometry created if none proveded

    mask_ND_before_astrometry : bool
        Convenience parameter to improve photometry solves by masking
        ND filter in the case where ND filter is not otherwise masked
        Default is ``False``

    """
    bmp_meta = bmp_meta or {}
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            result =  [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=fname, outdir=outdir,
                create_outdir=create_outdir,
                solve_timeout=solve_timeout,
                keep_intermediate=keep_intermediate,
                write_to_central_photometry=write_to_central_photometry,
                write_local_photometry=write_local_photometry,
                **kwargs)
                       for ccd, fname in zip(ccd_in, in_name)]                
        else:
            result = [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=in_name, outdir=outdir,
                create_outdir=create_outdir,
                solve_timeout=solve_timeout,
                keep_intermediate=keep_intermediate,
                write_to_central_photometry=write_to_central_photometry,
                write_local_photometry=write_local_photometry
                **kwargs)
                    for ccd in ccd_in]
        if None in result:
            bmp_meta.clear()
            return None
        return result

    ccd = ccd_in.copy()
    photometry = photometry or CorPhotometry()

    #if 'SII' in ccd.meta['filter']:
    #    # Experiment with sub-image in SII filter cases Didn't seem to
    #    # help, OOPS, we need to reference everything to full CCD to
    #    # get WCS reference right
    #    s = SMALL_FILT_CROP
    #    ccd = ccd[s[0,0]:s[1,0], s[0,1]:s[1,1]]

    if mask_ND_before_astrometry:
        # Subtle order bug if obj_center isn't called before we blank
        # out the info it uses to to find itself
        if isinstance(ccd, CorData):
            ccd.obj_center
        photometry.ccd = nd_filter_mask(ccd)
    else:
        photometry.ccd = ccd

    photometry.solve_timeout = solve_timeout or photometry.solve_timeout
    if keep_intermediate:
        if outdir is None:
            outname = None
        else:
            bname = os.path.basename(in_name)
            outname = os.path.join(outdir, bname)
            if create_outdir:
                os.makedirs(outdir, exist_ok=True)
            else:
                # This is safe because the astrometry stuff does not
                # actually write to the input filename, just use it as
                # a base
                outname = in_name
    else:
        outname = None
    photometry.outname = outname or photometry.outname

    ccd.wcs = photometry.wcs
    if photometry.wcs is None:
        # Pathological case: no wcs solution and no proxy solution
        bmp_meta.clear()
        return None
    
    ccd.meta['HIERARCH ASTROMETRY_NET_SOLVED'] = (
        photometry.solved and photometry.proxy_wcs_file is None)
    ccd.meta['PROXYWCS'] = photometry.proxy_wcs_file
    if photometry.source_table is None:
        nsources = 0
    else:
        nsources = len(photometry.source_table)
        ccd.meta['HIERARCH PHOTUTILS_NSOURCES'] = nsources

    ccd_pierside = ccd.meta['PIERSIDE']
    if ccd_pierside == 'UNKNOWN':
        # Set PIERSIDE to value found in proxy_wcs.  If no pierside
        # was found, proxy_wcs fails
        ccd.meta['PIERSIDE'] = photometry.pierside
    elif ccd_pierside != photometry.pierside:
        raise ValueError(f'CCD pierside and photometry pierside do not '
                         f'agree {ccd_pierside}, {photometry.pierside}')

    # Permanently install centralized astrometry and photometry
    # writing, though user can override.  Note that astrometry is only
    # saved if astrometry.net solved, but write photometry, since that
    # should be close enough
    if ccd.meta['HIERARCH ASTROMETRY_NET_SOLVED']:
        write_astrometry(ccd, in_name=in_name, **kwargs)
    if write_to_central_photometry:
        # Slighly different logic, since write_photometry also handles
        # case where local progressing pipeline writes photometry
        bname = os.path.basename(in_name)
        date, _ = ccd.meta['DATE-OBS'].split('T')
        outname = os.path.join(PHOTOMETRY_ROOT, date, bname)
        write_photometry(in_name=in_name, outname=outname,
                         photometry=photometry,
                         create_outdir=True,
                         write_source_gaia_join=True,
                         **kwargs)
    if write_local_photometry:
        # Guess at reasonable outname
        outname = outname_creator(in_name, outdir=outdir, **kwargs)
        write_photometry(in_name=in_name, outname=outname,
                         photometry=photometry,
                         outdir=outdir,
                         create_outdir=create_outdir,
                         **kwargs)

    return ccd    

class CorPhotometryArgparseMixin(PhotometryArgparseMixin):
    def add_solve_timeout(self, 
                 default=SOLVE_TIMEOUT,
                 help=None,
                 **kwargs):
        option = 'solve_timeout'
        if help is None:
            help = (f'max plate solve time in seconds (default: {default})')
        self.parser.add_argument('--' + option, type=float,
                                 default=default, help=help, **kwargs)

    def add_keep_intermediate(self, 
                      default=False,
                      help=None,
                      **kwargs):
        option = 'keep_intermediate'
        if help is None:
            help = (f'Keep intermediate astrometry solve files')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)


# rdls_table = Table.read('/tmp/WASP-36b-S001-R013-C002-R.rdls')
# #rdls_table.sort('mag')
# coords = SkyCoord(
#     rdls_table['RA'], rdls_table['DEC'])
# del rdls_table['RA', 'DEC']
# #rdls_table.add_column(MaskedColumn(data=coords,
# #                                   name='coord', mask=False))
# #rdls_table.add_column(Column(data=coords,
# #                             name='coord'))
# rdls_table['coord'] = coords
# source_table = Table.read('/tmp/WASP-36b-S001-R013-C002-R_p.ecsv')
# #source_table_coord.add_column(MaskedColumn(data=coords,
# #                                           name='coord', mask=False))
# #source_table_coord.add_column(Column(data=coords,
# #                                     name='coord'))
# tol = 10*u.arcsec
# source_with_gaia = table.join(
#     source_table, rdls_table,
#     join_type='inner',
#     keys=['coord'],
#     table_names=['source', 'gaia'],
#     uniq_col_name='{table_name}_{col_name}',
#     join_funcs={'coord': join_skycoord(tol)})
#                                           

if __name__ == '__main__':
    log.setLevel('DEBUG')
    # Base Astrometry test
    #photometry = CorPhotometry()
    #photometry.reduce_base_astrometry()
    
    from IoIO.simple_show import simple_show
    from IoIO.cordata import CorData
    from IoIO.cor_process import cor_process
    from IoIO.cormultipipe import add_raw_fname, detflux, planet_to_object
    from IoIO.calibration import Calibration
    from IoIO.horizons import obj_ephemeris, galsat_ephemeris
    from astropy.wcs.utils import proj_plane_pixel_scales
    c = Calibration(reduce=True)
    #fname = '/data/IoIO/raw/2021-10-28/Mercury-0001_Na_on.fit'
    #rccd = CorData.read(fname)
    #ccd = cor_process(rccd, calibration=c, auto=True)
    #ccd = add_raw_fname(ccd, in_name=fname)
    #ccd = obj_ephemeris(ccd, horizons_id='199',
    #                    horizons_id_type='majorbody',
    #                    obj_ephm_prefix='Mercury')
    #ccd = detflux(ccd)
    ##photometry = CorPhotometry(ccd, seeing=10, back_rms_scale=2)
    #photometry = CorPhotometry(ccd)
    ##wcs = photometry.wcs
    ##photometry.wide_source_table.show_in_browser()
    #ccd = add_astrometry(ccd, in_name=fname, photometry=photometry,
    #                     mask_ND_before_astrometry=True,
    #                     write_to_central_photometry=False)
    ##ccd.write('/data/Mercury/2021-10-28/Mercury-0001_Na_hand.fits', overwrite=True)
    #ccd.write('/tmp/test.fits', overwrite=True)

    ##fname = '/data/IoIO/raw/20211028/0029P-S001-R001-C001-Na_off_dupe-1.fts'
    #rccd = CorData.read(fname)
    #ccd = cor_process(rccd, calibration=c, auto=True)
    #ccd = add_raw_fname(ccd, in_name=fname)
    #photometry = CorPhotometry(ccd)
    #wcs = photometry.wcs
    #print(wcs)
    ###fname = '/data/IoIO/raw/20220414/KPS-1b-S001-R001-C001-R.fts'
    directory = '/data/IoIO/raw/2018-05-08/'
    fname = os.path.join(directory, 'SII_on-band_007.fits')
    ##fname = '/data/IoIO/raw/2018-05-08/SII_on-band_007.fits'
    ##fname = '/data/IoIO/raw/2020-09_Astrometry/Main_Astrometry_East_of_Pier.fit'
    rccd = CorData.read(fname)
    ccd = cor_process(rccd, calibration=c, auto=True)
    ccd = add_raw_fname(ccd, in_name=fname)
    ccd = planet_to_object(ccd, planet='Jupiter')
    ccd = galsat_ephemeris(ccd)
    photometry = CorPhotometry(ccd)
    ccd = add_astrometry(ccd, in_name=fname, photometry=photometry,
                         mask_ND_before_astrometry=True,
                         write_to_central_photometry=False)
    ##photometry.source_table.show_in_browser()
    #wcs = photometry.proxy_wcs
    #print(wcs)
    ##pixscale = (np.linalg.norm(np.mean(np.abs(wcs.wcs.cdelt))
    ##             np.linalg.norm(wcs.wcs.pc[0,:])
    ##            np.linalg.norm(wcs.wcs.pc[1,:]))
    ##print(pixscale)
    ##pixscale = proj_plane_pixel_scales(wcs) * u.deg
    
    
    
    #pixscale = np.linalg.norm(wcs.wcs.pc)  *u.deg
    #pixscale = np.mean(np.abs(wcs.wcs.cdelt)) *u.deg
    #print(pixscale.to(u.arcsec))

    ##fname = '/data/IoIO/raw/20220414/KPS-1b-S001-R001-C001-R.fts'
    #fname = '/data/IoIO/raw/2018-01_Astrometry/PinPointSolutionEastofPier.fit'
    #rccd = CorData.read(fname)
    #c = Calibration(reduce=True)
    #ccd = cor_process(rccd, calibration=c, auto=True, gain=False)
    #p = CorPhotometry(ccd=ccd)
    #p.source_table.show_in_browser()
    #print(p.wcs)
    #print(p.solve_field_stdout)
    #print(p.solve_field_stderr)
    
    ## Mercury test
    #import glob
    #import ccdproc as ccdp
    #from IoIO.cordata import CorData
    #from IoIO.cor_process import cor_process
    #from IoIO.calibration import Calibration
    #directory = '/data/IoIO/raw/2021-10-28/'
    #collection = ccdp.ImageFileCollection(
    #    directory, glob_include='Mercury*',
    #    glob_exclude='*_moving_to*')
    #flist = collection.files_filtered(include_path=True)
    #c = Calibration(reduce=True)
    #photometry = CorPhotometry()
    #for fname in flist:
    #    log.info(f'trying {fname}')
    #    rccd = CorData.read(fname)
    #    ccd = cor_process(rccd, calibration=c, auto=True)
    #    ccd = add_astrometry(ccd, photometry=photometry, solve_timeout=1)
    #    #photometry.show_segm()
    #    source_table = photometry.source_table
    #    source_table.show_in_browser()

    ## Exoplanet & Jupiter test
    #from IoIO.cordata import CorData
    #from IoIO.cor_process import cor_process
    #from IoIO.calibration import Calibration
    #c = Calibration(reduce=True)    
    #photometry = CorPhotometry()
    ##fname = '/data/IoIO/raw/20220414/KPS-1b-S001-R001-C001-R.fts'
    ##directory = '/data/IoIO/raw/2018-05-08/'
    ##fname = os.path.join(directory, 'SII_on-band_007.fits')
    #fname = '/data/IoIO/raw/2019-04_Astrometry/Main_Astrometry_West_of_Pier.fit'
    #rccd = CorData.read(fname)
    ##ccd = cor_process(rccd, calibration=c, auto=True)
    ##photometry.ccd = ccd
    #photometry.ccd = rccd
    ##photometry.source_table.show_in_browser()
    #print(photometry.wcs)
    ##photometry.wide_source_table.show_in_browser()
    ##photometry.wide_rdls_table.show_in_browser()
    ##photometry.source_gaia_join.show_in_browser()
    ##photometry.source_gaia_join.write('/tmp/test_gaia.ecsv', overwrite=True)
