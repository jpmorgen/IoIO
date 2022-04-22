"""Adds astrometry capability to IoIO Photometry object.  Must be
named uniquely to avoid conflicting with astrometry.net's astrometry
module so that jobs can be run in the cwd

"""

import os
import argparse
from tempfile import TemporaryDirectory
import subprocess

import numpy as np

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table, MaskedColumn, Column, join_skycoord
from astropy import table
from astropy.wcs import WCS

from astroquery.simbad import Simbad

from precisionguide import PGData

from IoIO.photometry import (SOLVE_TIMEOUT, JOIN_TOLERANCE, Photometry,
                             PhotometryArgparseMixin)
from IoIO.utils import savefig_overwrite
from IoIO.cordata import CorData
from cormultipipe import nd_filter_mask

MIN_SOURCES_TO_SOLVE = 5
KEYS_TO_SOURCE_TABLE = ['DATE-AVG',
                        ('DATE-AVG-UNCERTAINTY', u.s),
                        ('EXPTIME', u.s),
                        'FILTER',
                        'AIRMASS']

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
                 solve_timeout=SOLVE_TIMEOUT,
                 keys_to_source_table=KEYS_TO_SOURCE_TABLE,
                 **kwargs):
        super().__init__(ccd=ccd,
                         keys_to_source_table=keys_to_source_table,
                         **kwargs)
        self.outname = outname
        self.solve_timeout = solve_timeout

    def init_calc(self):
        super().init_calc()
        self._rdls_table = None
        self._gaia_table = None
        self._source_gaia_join = None

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
            self._solved = False
            return None

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

        # If *_ephemeris has run, objct* coords are very high quality.  
        ra = Angle(self.ccd.meta['objctra'])
        ra = ra.to_string(sep=':')
        dec = Angle(self.ccd.meta['objctdec'])
        dec = dec.to_string(alwayssign=True, sep=':')
        naxis1 = self.ccd.meta['naxis1']
        naxis2 = self.ccd.meta['naxis2']
        pixscale = self.ccd.meta['PIXSCALE']
        radius = np.linalg.norm((naxis1, naxis2)) * u.pixel
        radius = radius * pixscale *u.arcsec/u.pixel
        radius = radius.to(u.deg)
        if (isinstance(self.ccd, PGData)
            and self.ccd.center_quality > 5):
            # Reference CRPIX* to object's PGData center pixel
            # Note obj_center is python ordering
            c = self.ccd.obj_center
            crpix_str = f'--crpix-x {c[1]} --crpix-y {c[0]}'
        else:
            crpix_str = ''
        if self.solve_timeout is not None:
            cpulimit_str = f'--cpulimit {self.solve_timeout}'
        else:
            cpulimit_str = ''

        astrometry_command = \
            f'solve-field --x-column xcentroid --y-column ycentroid ' \
            f'--ra {ra} --dec {dec} --radius {2*radius.value:.2f} ' \
            f'--width {naxis1} --height {naxis2} ' \
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
                self._solved = True
                with fits.open(outroot + '.wcs') as HDUL:
                    self._wcs = WCS(HDUL[0].header)
                # The astrometry.net code or CCDData.read seems to set
                # RADESYS = FK5 deep in the bowls of wcs so nothing I
                # do affects it
                self._rdls_table = Table.read(outroot + '.rdls')
                #self.source_table.show_in_browser()
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
        j = table.join(
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
    if photometry.wcs is not None:
        ccd.wcs = photometry.wcs
    ccd.meta['HIERARCH PHOTUTILS_NSOURCES'] = len(photometry.source_table)
    # I am currently not putting the wcs into the metadata because I
    # don't need it -- it is available as ccd.wcs or realtively easily
    # extracted from disk like I do in Photometry.astrometry.  I am
    # also not putting the SourceTable into the metadata, because it
    # is still hanging around in the Photometry object.
    return ccd    

def object_to_objctradec(ccd_in, **kwargs):
    """cormultipipe post-processing routine to query Simbad for RA and DEC

    """    
    ccd = ccd_in.copy()
    s = Simbad()
    obj = ccd.meta['OBJECT']
    simbad_results = s.query_object(obj)
    if simbad_results is None:
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

    # Exoplanet test
    from IoIO.cordata import CorData
    from IoIO.cor_process import cor_process
    from IoIO.calibration import Calibration
    c = Calibration(reduce=True)    
    photometry = CorPhotometry()
    fname = '/data/IoIO/raw/20220414/KPS-1b-S001-R001-C001-R.fts'
    rccd = CorData.read(fname)
    ccd = cor_process(rccd, calibration=c, auto=True)
    photometry.ccd = ccd
    photometry.source_table.show_in_browser()
    photometry.wide_source_table.show_in_browser()
    photometry.wide_rdls_table.show_in_browser()
    photometry.source_gaia_join.show_in_browser()
    photometry.source_gaia_join.write('/tmp/test_gaia.ecsv', overwrite=True)
