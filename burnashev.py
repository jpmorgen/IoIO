"""Process Burnashev data

The data in digital form are available from 

http://cdsarc.u-strasbg.fr/viz-bin/cat/III/126

in particular the ftp section:

https://cdsarc.cds.unistra.fr/ftp/III/126/

"""

import os
import gzip

import numpy as np

from astropy import units as u

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region

BURNASHEV_ROOT = '/data/Burnashev/'
N_SPEC_POINTS = 200
DELTA_LAMBDA = 2.5*u.nm

def read_cat():
    # The ReadMe explains that we want part3
    with gzip.open(os.path.join(BURNASHEV_ROOT, 'part3.dat.gz')) as f:
        bcat = f.read()
    # cat is an array of bytes, so we need to further decode

    # Split on newlines
    bcat = bcat.split(b'\n')
    cat = []
    for line in bcat:
        if len(line) == 0:
            # EOF
            break

        vmag = line[19:24]
        try:
            vmag = float(vmag)
        except:
            vmag = float('NaN')
        nobs = line[54:57]
        try:
            nobs = int(nobs)
        except:
            nobs = float('NaN')

        log_e = []
        # N_SPEC_POINTS floats 10 characters wide
        for i in range(N_SPEC_POINTS):
            start = 62 + i*(10)
            stop = start + 10
            p = float(line[start:stop])
            log_e.append(p)
        #log_e = np.asarray(log_e)*np.log(u.milliWatt * u.m**-2 * u.cm**-1)
        #log_e = np.asarray(log_e)*np.log(u.erg/u.s/u.cm**2/u.cm)
        log_e = np.asarray(log_e)
        flux = 10**log_e*u.erg/u.s/u.cm**2/u.cm

        entry = {'Name': line[0:19].decode("utf-8").strip(),
                 'Vmag': vmag,
                 'n_Vmag': line[24:29].decode("utf-8").strip(),
                 'SpType': line[30:40].decode("utf-8").strip(),
                 'Date': line[40:50].decode("utf-8").strip(),
                 'Origin': line[51:54].decode("utf-8").strip(),
                 'Nobs': nobs,
                 'lambda1': float(line[58:62]),
                 'logE': log_e,                 
                 }
        cat.append(entry)
    return cat

def get_spec(cat_entry):
    lambdas = cat_entry['lambda1']*u.nm + np.arange(N_SPEC_POINTS)*DELTA_LAMBDA
    flux = 10**cat_entry['logE'] * u.milliWatt * u.m**-2 * u.cm**-1
    spec = Spectrum1D(spectral_axis=lambdas, flux=flux)
    return spec

cat = read_cat()
star = cat[234]
spec = get_spec(star)
filter_bandpass = SpectralRegion(5000*u.AA, 6000*u.AA)
sub_spec = extract_region(spec, filter_bandpass)

with open(os.path.join(BURNASHEV_ROOT, 'stars.dat')) as f:
    bstars = f.read()

    bstars = bstars.split('\n')
    stars = []
    for line in bstars:
        if len(line) == 0:
            break
        
        entry = {'Name': line[0:19].strip(),
                 'RAh': int(line[19:22]),
                 'RAm': int(line[22:25]),
                 'RAs': float(line[25:30]),
                 'DECd': float(line[30:34]),
                 'DECm': float(line[34:37]),
                 'DECs': float(line[37:40]),
                 'SpType': line[40:51].strip(),
                 }
        stars.append(entry)
