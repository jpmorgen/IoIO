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
from astropy.coordinates import Angle
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
        if self.source_catalog is None:
            return None

        source_table = self.source_catalog.to_table(
            ['label',
             'xcentroid',
             'ycentroid',
             'min_value',
             'max_value',
             'local_background',
             'segment_flux',
             'segment_fluxerr'])
        source_table.sort('segment_flux', reverse=True)
        # Ack.  Astropy is not making this easy.  Writing the
        # QTable fails entirely, possibly because it has mixed
        # Quantity and non-Quantity columns.  Error has something
        # about not being able to represent an object
        if source_table.has_mixin_columns:
            # Move collection of Quantity columns to regular
            # columns with quantity attributes.  This lets the
            # write happen, but, unfortunately does not record the
            # units
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
        print(astrometry_command)
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
                with fits.open(outroot + '.wcs') as HDUL:
                    wcs = WCS(HDUL[0].header)
                self.ccd.wcs = wcs
                return wcs
