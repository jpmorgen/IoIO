"""Adds astrometry capability to IoIO Photometry object.  Must be
named uniquely to avoid conflicting with astrometry.net's astrometry
module so that jobs can be run in the cwd

"""

import os
from tempfile import TemporaryDirectory
import subprocess

import numpy as np

from astropy import log
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table, MaskedColumn, Column, join_skycoord
from astropy import table
from astropy.wcs import WCS

from IoIO.photometry import Photometry
from IoIO.cordata import CorData

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
        cpulimit : float
            Plate solve time limit (seconds)

    """
    def __init__(self,
                 ccd=None,
                 outname=None,
                 cpulimit=None,
                 join_tolerance=None,
                 **kwargs):
        super().__init__(ccd=ccd, **kwargs)
        self.outname = outname
        self.cpulimit = cpulimit
        if join_tolerance is None:
            join_tolerance = 5*u.arcsec
        self.join_tolerance = join_tolerance

    def init_calc(self):
        super().init_calc()
        self._rdls_table = None
        self._gaia_table = None
        self._source_gaia_join = None

    @property
    def source_table(self):
        if self._source_table is not None:
            return self._source_table
        # https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
        self._source_table = self.source_catalog.to_table(
            ['label',
             'xcentroid',
             'ycentroid',
             'elongation',
             'equivalent_radius',
             'min_value',
             'max_value',
             'segment_flux',
             'segment_fluxerr'])
        self._source_table.sort('segment_flux', reverse=True)
        return self._source_table
        
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
        if self.source_catalog is None:
            self._solved = False
            return None

        # Ack.  Astropy is not making this easy.  Writing the
        # QTable fails entirely, possibly because it has mixed
        # Quantity and non-Quantity columns.  Error has something
        # about not being able to represent an object
        # Save off a separate qtable so we have our quantities
        source_table = self.source_table        
        if source_table.has_mixin_columns:
            # Move collection of Quantity columns to regular
            # columns with quantity attributes.  This lets the
            # write happen, but, unfortunately does not record the
            # units.  
            source_table = Table(source_table)

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
        if (isinstance(self.ccd, CorData)
            and self.ccd.center_quality > 5):
            # Set our reference CRPIX* to Jupiter's position.  Note
            # obj_center is python ordering
            c = self.ccd.obj_center
            crpix_str = f'--crpix-x {c[1]} --crpix-y {c[0]}'
        else:
            crpix_str = ''
        if self.cpulimit is not None:
            cpulimit_str = f'--cpulimit {self.cpulimit}'
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
            source_table.write(xyls_file, format='fits', overwrite=True)
            p = subprocess.run(astrometry_command, shell=True,
                               capture_output=True)
            self.solve_field_stdout = p.stdout
            self.solve_field_stderr = p.stderr
            if os.path.isfile(outroot+'.solved'):
                self._solved = True
                with fits.open(outroot + '.wcs') as HDUL:
                    self._wcs = WCS(HDUL[0].header)
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

def add_astrometry(ccd_in, bmp_meta=None, photometry=None,
                   in_name=None, outdir=None, create_outdir=None,
                   cpulimit=None, keep_intermediate=False, **kwargs):
    """cormultipipe post-processing routine to add wcs to ccd"""
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=fname, outdir=outdir,
                create_outdir=create_outdir,
                cpulimit=cpulimit,  keep_intermediate=keep_intermediate,
                **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [add_astrometry(
                ccd, bmp_meta=bmp_meta, photometry=photometry,
                in_name=in_name, outdir=outdir,
                create_outdir=create_outdir,
                cpulimit=cpulimit,  keep_intermediate=keep_intermediate,
                **kwargs)
                    for ccd in ccd_in]            
    ccd = ccd_in.copy()
    photometry = photometry or CorPhotometry()
    photometry.ccd = ccd
    photometry.cpulimit = cpulimit or photometry.cpulimit
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
    ccd.wcs = photometry.wcs
    # I am currently not putting the wcs into the metadata because I
    # don't need it -- it is available as ccd.wcs or realtively easily
    # extracted from disk like I do in Photometry.astrometry.  I am
    # also not putting the SourceTable into the metadata, because it
    # is still hanging around in the Photometry object.
    return ccd    

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

