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
from astropy.table import Table
from astropy.wcs import WCS

from IoIO.photometry import Photometry
from IoIO.cordata import CorData

class CorPhotometry(Photometry):
    def __init__(self,
                 ccd=None,
                 cpulimit=None,
                 **kwargs):
        super().__init__(ccd=ccd, **kwargs)
        self.cpulimit = cpulimit

    def init_calc(self):
        super().init_calc()
        self._rdls = None

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
             'local_background',
             'segment_flux',
             'segment_fluxerr'])
        self._source_table.sort('segment_flux', reverse=True)
        return self._source_table

        
    def astrometry(self, outname=None, cpulimit=None):
        """Plate solve ccd image, setting ccd's WCS object

        Parameters
        ----------
        outname : str or None

            Presumed name of FITS file that is being plate solved.
            The file need not exist yet, since we are passing the
            already extracted star positions to solve-field.  The
            extension will be removed and the rest of the name used
            for the solve-field input and output.  If `None`, a
            temporary directory is created and files are created
            there.

        Returns
        -------
        wcs : `~astropy.wcs.WCS` or None

        """
        if self._solved:
            return self.ccd.wcs
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
        if cpulimit is not None:
            cpulimit_str = f'--cpulimit {cpulimit}'
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
                    wcs = WCS(HDUL[0].header)
                self.ccd.wcs = wcs
                self._rdls = Table.read(outroot + '.rdls')
                #self.source_table.show_in_browser()
                return wcs

    @property
    def rdls(self):
        if self._rdls is not None:
            return self._rdls
        if self.solved:
            return self._rdls
        return None

    def rdls_to_source_table(self):
        """Add rdls columns to source_table"""
        if self.rdls is None:
            return
        
    def obj_to_source_table(self):
        if self.source_table is None:
            return
        obj = self.ccd.meta.get('OBJECT')
        objctra = self.ccd.meta.get('OBJCTRA')
        objctdec = self.ccd.meta.get('OBJCTDEC')
        obj_coord = SkyCoord(objctra, objctdec, unit=(u.hour, u.deg))
        dobj = obj_coord.separation(self.sky_coords)
        #dobj = [obj_coord.separation(source_coord)
        #        for source_coord in self.sky_coords]
        dobj = dobj.to(u.arcsec)
        idx = np.argmin(dobj)
        #idx, dang, _ = obj_coord.match_to_catalog_sky(self.sky_coords)
        #dang = dang.to(u.arcsec)
        # Add code to check how close I am
        #if dang > self.seeing*u.pixel
        #if dang > 3*u.arcsec:
        #    log.warning(f'{obj} OBJCTRA + OBJCTDEC are {dang} off')
        # String column ends up being of fixed width
        objs = np.full(len(self.source_table),
                       fill_value=' '*len(obj))
        objs[idx] = obj
        self.source_table['OBJECT'] = objs
        self.source_table['DOBJ'] = dobj

def add_astrometry(ccd_in, bmp_meta=None, photometry=None,
                   in_name=None, outdir=None, create_outdir=None,
                   cpulimit=None, **kwargs):
    """cormultipipe post-processing routine to add wcs to ccd"""
    if isinstance(ccd_in, list):
        if isinstance(in_name, list):
            return [add_astrometry(ccd, bmp_meta=bmp_meta, 
                                   photometry=photometry,
                                   in_name=fname, outdir=outdir,
                                   create_outdir=create_outdir,
                                   cpulimit=cpulimit, **kwargs)
                    for ccd, fname in zip(ccd_in, in_name)]
        else:
            return [add_astrometry(ccd, bmp_meta=bmp_meta, 
                                   photometry=photometry,
                                   in_name=in_name, outdir=outdir,
                                   create_outdir=create_outdir,
                                   cpulimit=cpulimit, **kwargs)
                    for ccd in ccd_in]            
    ccd = ccd_in.copy()
    bmp_meta = bmp_meta or {}
    photometry = photometry or CorPhotometry()
    photometry.ccd = ccd
    photometry.cpulimit = photometry.cpulimit or cpulimit
    if outdir is not None:
        bname = os.path.basename(in_name)
        outname = os.path.join(outdir, bname)
        if create_outdir:
            os.makedirs(outdir, exist_ok=True)
    else:
        # This is safe because the astrometry stuff does not actually
        # write to the input filename, just use it as a base
        outname = in_name
    wcs = photometry.astrometry(outname=outname)
    # I am currently not putting the wcs into the metadata because I
    # don't need it -- it is available as ccd.wcs or realtively easily
    # extracted from disk like I do in Photometry.astrometry.  I am
    # also not putting the SourceTable into the metadata, because it
    # is still hanging around in the Photometry object.
    return ccd    

