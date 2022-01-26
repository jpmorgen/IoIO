#!/usr/bin/python3
"""Process C/2020 F3 (NEOWISE) observations"""

import os
import glob 

from astropy import log

from ccdmultipipe import as_single

from cormultipipe import (RAW_DATA_ROOT,
                          get_dirs_dates, reduced_dir,
                          Calibration, OffCorMultiPipe, FixFnameCorMultipipe,
                          nd_filter_mask, mask_nonlin_sat, detflux)

class FixFnameOffCorMultipipe(FixFnameCorMultipipe, OffCorMultiPipe):
    pass

NEOWISE_ROOT = '/data/NEOWISE_2020F3'

if __name__ == "__main__":

    log.setLevel('DEBUG')
    data_root = RAW_DATA_ROOT
    reduced_root = NEOWISE_ROOT
    reduced_root = "/tmp"
    start = "2020-07-08"
    stop = "2021-01-01"
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)

    dirs, _ = zip(*dirs_dates)
    assert len(dirs) > 0

    glob_include = ['NEOWISE-*', 'CK20F030-*']
    c = Calibration(reduce=True)
    cmp = FixFnameOffCorMultipipe(auto=True, calibration=c,
                                  fits_fixed_ignore=True,
                                  outname_ext='.fits',
                                  post_process_list=[nd_filter_mask,
                                                     mask_nonlin_sat,
                                                     detflux,
                                                     as_single])
    for d in dirs:
        flist = []
        for gi in glob_include:
            flist += glob.glob(os.path.join(d, gi))
        if len(flist) == 0:
            log.debug(f'No NEOWISE observations in {d}')
            continue
        reddir = reduced_dir(d, reduced_root)
        pout = cmp.pipeline(flist, outdir=reddir,
                            create_outdir=True, overwrite=True)

##### directory = '/data/IoIO/raw/2020-07-08'
##### reddir = reduced_dir(directory, '/data/NEOWISE_2020F3/analysis', create=True)
##### 
##### c = Calibration(reduce=True)
##### cmp = OffCorMultiPipe(auto=True, calibration=c,
#####                       post_process_list=[nd_filter_mask, mask_nonlin_sat, detflux, as_single])
##### flist = glob.glob(os.path.join(directory, 'NEOWISE-*_Na*'))
##### pout = cmp.pipeline(flist, outdir=reddir, create_outdir=True, overwrite=True)
##### 
##### # image e/s	SNR
##### # 1	39	6
##### # 2	38 	5
##### # 3	65	26
##### # 4	76	27
##### # 5	131	42
##### # 6	178	55
##### # 7	209	57
##### # 8	221	38
##### # 10	273	42 Starting to drift off image
##### # 12	267	23
##### # 13	gone
##### 
##### # So for the 11 images that Carl is using, 208.6 is an OK number to use
##### 
##### #fname1 = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0007_Na-on.fit'
##### #fname2 = '/data/io/IoIO/raw/2020-07-08/NEOWISE-0008_Na_off.fit'
##### #pout = cmp.pipeline([fname1, fname2], outdir='/data/NEOWISE_2020F3/analysis/2020-07-08/', create_outdir=True, overwrite=True)
##### 
##### # Star at bottom of MaxIm image is intensity 208.6, SNR 57.  So what
##### # star is that?  NEOWISE ephemeris:
##### 
##### #DATE-OBS	= '2020-07-08T11:24:35'
##### # RA and DEC = 06 21 37.83 +38 03 19.2
##### 
##### # http://vizier.cfa.harvard.edu/viz-bin/VizieR-2
##### # Ahh, I am now seeing the USNO only has B and R.  Interesting set of
##### # differences in Vmag ~8 for star near 06 21 07.7	+37 56 35
##### # HD 44015
##### # Looks like that is in the right place
##### 
##### # Star			Vmag	Spectral Type
##### # SAO 59001		8.2	K0
##### # TICID1 2927		8.5
##### # TYC 2927 1717 1	8.00
##### # IRAS 06177+3757	8.113
##### # ASCC-2.5 V3		7.999	K0 III
##### # GSC lots		8.0-8.11
##### # TASS Mark IV		8.065
##### # JMMC Stellar Diameters
##### # HD 44015		7.997	K0
##### # 3rd Bibliog. Cat. of Stellar Radial Vel
##### #			7.70	K0 III
##### # Tycho-2 Spectral Type Catalog
##### #			8.113 (VTmag)
##### # MSX Infrared Astrometric Catalog (Egan+ 1996)ReadMe+ftp
##### #			8.00
##### # Teff and metallicities for Tycho-2 stars (Ammons+, 2006)
##### #			8.113
##### # All-sky spectrally matched Tycho2 stars (Pickles+, 2010)ReadMe+ftp
##### #			7.992	wK1III
##### 
##### # OK, lets just take 8.0 K0 III for the same of argument
##### 
##### # Find a representative K0 III in Burnashev
##### 
##### # http://sirius.bu.edu/planetary/obstools/starflux/
##### 
##### # 0941	019476	038609	Kap 27	Per	03 09 29.8	+44 51 26	3.80	K0III
##### 
##### # Wavelength (Å)
##### #	Flux (photons/cm² /Å /s)
##### # 5850	37.7
##### # 5875	37.8
##### # 5900	37.8
##### # 5925	38.1
##### 
##### # Rayleigh calibration as per Mercury.notebk
##### # Fri Jan 29 12:37:23 2021 EST  jpmorgen@snipe
##### 
##### # Call it
##### 
##### F = 37.8 * 2.512*10**(3.80-8.0) *u.photon*u.cm**-2*u.A**-1*u.s**-1
##### FWHM = 12*u.A
##### Ff = F*FWHM
##### theta = (plate_scale * u.arcsec)
##### I = 4*np.pi*Ff / (theta**2)
##### I = I.to(u.R)
##### # Use flux from MaxIm for now
##### S = 208.6 * u.electron/u.s
##### 
##### eps_to_R = I / S
##### print(eps_to_R)
##### # 3805.915711594373 R s / electron
##### 
##### on = OffCorData.read('/data/NEOWISE_2020F3/analysis/2020-07-08/NEOWISE-0007_Na-on_r.fit')
##### off = OffCorData.read('/data/NEOWISE_2020F3/analysis/2020-07-08/NEOWISE-0008_Na_off_r.fit')
##### 
##### # --> Need to find center, since I was wondering a bit
##### 
##### # from ~/IoIO_reduction.notebk
##### # Tue Feb 09 22:35:28 2021 EST  jpmorgen@snipe
##### off_on_ratio = 4.72*u.dimensionless_unscaled
##### off_scaled = off.divide(off_on_ratio, handle_meta='first_found')
##### off_sub = on.subtract(off_scaled, handle_meta='first_found')
##### 
##### off_sub = off_sub.multiply(eps_to_R, handle_meta='first_found')
##### 
##### off_sub.write('/tmp/test.fits', overwrite=True)
##### 
##### # A little over-subtracted to the tune of 50 R or so.
