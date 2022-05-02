"""Adds astrometry capability to IoIO Photometry object, including
guaranteed WCS header

"""

import os
import glob
import warnings
import argparse
from tempfile import TemporaryDirectory
import subprocess

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io.fits import getheader
from astropy.table import Table, MaskedColumn, Column, join, join_skycoord
from astropy import table
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from astroquery.simbad import Simbad

from reproject.mosaicking import find_optimal_celestial_wcs

import ccdproc as ccdp

from precisionguide import PGData

import IoIO.sx694 as sx694
from IoIO.utils import FITS_GLOB_LIST, multi_glob, savefig_overwrite
from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE, rot_wcs,
                             Photometry, PhotometryArgparseMixin)
from IoIO.cormultipipe import (RAW_DATA_ROOT, ASTROMETRY_ROOT,
                               astrometry_outname,
                               CorMultiPipeBinnedOK, nd_filter_mask)
from IoIO.calibration import Calibration

MIN_SOURCES_TO_SOLVE = 5
MAX_SOURCES_TO_SOLVE = 100
ASTROMETRY_GLOB = '*_Astrometry'
KEYS_TO_SOURCE_TABLE = ['DATE-AVG',
                        ('DATE-AVG-UNCERTAINTY', u.s),
                        ('EXPTIME', u.s),
                        'FILTER',
                        'AIRMASS']

# Proxy WCS settings.  
MIN_CENTER_QUALITY = 5


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

# Necessary for trying to construct PIERSIDE
MIN_PIERFLIP_TIME = 8*u.min

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
    if wcs.wcs.has_cd():
        mat = wcs.wcs.cd
    elif wcs.wcs.has_pc():
        mat = wcs.wcs.pc
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
    if cdwcs == PIERSIDE_WEST:
        return 'WEST'
    if cdwcs == -PIERSIDE_WEST:
        return 'EAST'
    log.error(f'WCS solution not consistent with '
              f'IoIO coronagraph {wcs}')
    return 'ERROR'

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
        cmp = CorMultiPipeBinnedOK(
            calibration=calibration,
            auto=True,
            photometry=self,
            #solve_timeout=600, # Try to solve all [didn't help]
            solve_by_proxy=False, # make sure original solutions stand
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
        self.reduce_base_astrometry()
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

    @property
    def proxy_wcs(self):
        if (self.ccd.center_quality < self.min_center_quality
            or self.ccd.meta.get('OBJECT_TO_OBJCTRADEC') is None):
            # There is not much we can do in these cases -- we need to
            # know where our object is in both pixels and RA and DEC
            self._solved = False
            return None
        
        # First, find the closest astrometry directory before our
        # observation.  Below, we handle the case where the directory
        # is after our observation.
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

        # Avoid UNKNOWN PIERSIDE, if possible
        # https://stackoverflow.com/questions/406230/regular-expression-to-match-a-line-that-doesnt-contain-a-word
        try:
            wcs_collect = wcs_collect.filter(
                pierside='^((?!unknown).)*$',
                regex_match=True)
        except FileNotFoundError:
            wcs_collect = wcs_collect

        ## pierside is a little more complex
        #st = wcs_collect.summary
        #ew_mask = st['pierside'] != 'UNKNOWN'
        #if np.any(ew_mask):
        #    wcs_collect = ccdp.ImageFileWcs_Collect(
        #        best_dir, keywords=['date-obs',
        #                            'pierside',
        #                            'xbinning',
        #                            'ybinning'])

        # Match PIERSIDE after filtering UNKNOWN cases
        try:
            wcs_collect = wcs_collect.filter(
                pierside=self.ccd.meta['pierside'])
        except FileNotFoundError:
            wcs_collect = wcs_collect

        wcs_tobs = Time(wcs_collect.values('date-obs'))
        tobs = Time(date_obs)
        dts = tobs - wcs_tobs
        ibest = np.argmin(np.abs(dts))
        self._proxy_wcs_file = wcs_collect.files_filtered(
            include_path=True)[ibest]
        wcs_xy_binning = np.asarray(
            (wcs_collect.values('xbinning')[ibest],
             wcs_collect.values('ybinning')[ibest]))
        wcs_raw_pierside = wcs_collect.values('pierside')[ibest]
        wcs = WCS(self._proxy_wcs_file)

        # Scale WCS transformations to match ccd pixel size.  All of
        # our Astrometry solutions have run through the wcslib stuff,
        # which translates older CD matrices to PC-style.  
        # https://stackoverflow.com/questions/36384760/transforming-a-row-vector-into-a-column-vector-in-numpy
        # Calabretta and Greisen (2002) define the PC
        wcs_to_ccd_scale = ccd_xy_binning / wcs_xy_binning
        wcs_to_ccd_scale_col = wcs_to_ccd_scale[..., None]
        wcs.wcs.pc *= wcs_to_ccd_scale_col
        #wcs._naxis /= wcs_to_ccd_scale

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
            wcs_pierside = wcs_collect.values('pierside')[ibest]
            
        ccd_pierside = self.ccd.meta['pierside']
        if ccd_pierside == 'UNKNOWN':
            # The best we can do in this case is just propagate our
            # close wcs pierside if it is known.  We just
            # don't have the information necessary to prescribe a wcs
            # pier flip
            if (dts[ibest] < MIN_PIERFLIP_TIME
                and wcs_raw_pierside != 'UNKNOWN'):
                ccd_pierside = wcs_pierside
            flip = False
        elif ccd_pierside != 'EAST' and ccd_pierside != 'WEST':
            log.error(f'Unexpected ccd PIERSIDE = {ccd_pierside}')
            flip = False
        elif wcs_pierside != 'EAST' and wcs_pierside != 'WEST':
            log.error(f'Unexpected WCS PIERSIDE = {wcs_pierside}')
            flip = False
        elif ccd_pierside == wcs_pierside:
            flip = False
        else:
            flip = True
        self._pierside = ccd_pierside
        if flip:
            wcs = rot_wcs(wcs, 180*u.deg)

        # Point the WCS to our object.  Note C ordering and zero
        # reference of obj_center.  CRPIX is FORTRAN order and 1
        # reference
        wcs.wcs.crval = np.asarray((self.ccd.sky_coord.ra.deg,
                                    self.ccd.sky_coord.dec.deg))
        wcs.wcs.crpix = self.ccd.obj_center[::-1] + 1
        # Setting solved=True allows accumulated solves to extend
        # PIERSIDE during times of PIERSIDE = UNKNOWN
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

        ra = self.ccd.sky_coord.ra.to_string(sep=':')
        dec = self.ccd.sky_coord.dec.to_string(alwayssign=True, sep=':')
        ccd_shape = np.asarray(self.ccd.shape)
        pdiameter = np.average(ccd_shape) * u.pixel
        # PIXSCALE is binned
        pixscale = self.ccd.meta.get('PIXSCALE')
        if pixscale is None:
            # failsafe that should not happen if sx694.metadata has run
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

        astrometry_command = \
            f'solve-field --x-column xcentroid --y-column ycentroid ' \
            f'--ra {ra} --dec {dec} --radius {diameter.value:.2f} ' \
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
            elif not self.solve_by_proxy:
                # PinPoint WCS does get read, so might as well put it
                # in our object, since this gives us a way to get
                # astrometry in a pinch
                wcs = self.ccd.wcs
                self._pierside = pierside(wcs)
                # For the Astrometry base files.  Not marking them as
                # solved lets us find them easier in the
                # ASTROMETRY_ROOT sytem, not that I have been able to
                # figure out how to get astrometry.net to solve
                # them....
                self._solved = False
            else:
                wcs = self.proxy_wcs
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

def add_astrometry(ccd_in, bmp_meta=None,
                   photometry=None,
                   mask_ND_before_astrometry=False,
                   in_name=None,
                   outdir=None,
                   create_outdir=None,
                   solve_timeout=None,
                   keep_intermediate=False,
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
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=fname, outdir=outdir,
                create_outdir=create_outdir,
                solve_timeout=solve_timeout,
                keep_intermediate=keep_intermediate,
                **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=in_name, outdir=outdir,
                create_outdir=create_outdir,
                solve_timeout=solve_timeout,
                keep_intermediate=keep_intermediate,
                **kwargs)
                    for ccd in ccd_in]
    ccd = ccd_in.copy()
    photometry = photometry or CorPhotometry()
    if mask_ND_before_astrometry:
        photometry.ccd = nd_filter_mask(ccd)
    else:
        photometry.ccd = ccd
    photometry.solve_timeout = solve_timeout or photometry.solve_timeout
    if keep_intermediate:
        if outdir is not None:
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
    if photometry.wcs is None:
        # Pathological case: no wcs solution and no proxy solutiion
        return ccd
    
    ccd.wcs = photometry.wcs
    ccd.meta['HIERARCH ASTROMETRY_NET_SOLVED'] = (
        photometry.solved and photometry.proxy_wcs_file is None)
    ccd.meta['PROXYWCS'] = photometry.proxy_wcs_file
    ccd.meta['HIERARCH PHOTUTILS_NSOURCES'] = len(photometry.source_table)
    ccd_pierside = ccd.meta['PIERSIDE']
    if (ccd_pierside != 'UNKNOWN'
        and ccd_pierside != photometry.pierside):
        raise ValueError(f'CCD pierside and photometry pierside do not '
                         f'agree {ccd_pierside}, {photometry.pierside}')
    # Reset PIERSIDE in case proxy wcs was able to fill in a missing
    # pierside
    ccd.meta['PIERSIDE'] = photometry.pierside
    # I am currently not putting the wcs into the metadata because I
    # don't need it -- it is available as ccd.wcs or realtively easily
    # extracted from disk like I do in Photometry.astrometry.  I am
    # also not putting the SourceTable into the metadata, because it
    # is still hanging around in the Photometry object.
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
    from IoIO.cordata import CorData
    from IoIO.cor_process import cor_process
    from IoIO.calibration import Calibration
    c = Calibration(reduce=True)
    ##fname = '/data/IoIO/raw/20220414/KPS-1b-S001-R001-C001-R.fts'
    #directory = '/data/IoIO/raw/2018-05-08/'
    #fname = os.path.join(directory, 'SII_on-band_007.fits')
    fname = '/data/IoIO/raw/2018-05-08/SII_on-band_007.fits'
    rccd = CorData.read(fname)
    ccd = cor_process(rccd, calibration=c, auto=True)
    photometry = CorPhotometry(ccd)
    wcs = photometry.proxy_wcs
    print(wcs)

    #fname = '/data/IoIO/raw/2018-01_Astrometry/PinPointSolutionEastofPier.fit'
    #rccd = CorDataNDparams.read(fname)
    #c = Calibration(reduce=True)
    #ccd = cor_process(rccd, calibration=c, auto=True, gain=False)
    #p = CorPhotometry(ccd=ccd)

    ## Base Astrometry test
    #photometry = CorPhotometry()
    #photometry.reduce_base_astrometry()
    
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
