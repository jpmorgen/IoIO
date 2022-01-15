"""Process Burnashev data

The data in digital form are available from 

http://cdsarc.u-strasbg.fr/viz-bin/cat/III/126

in particular the ftp section:

https://cdsarc.cds.unistra.fr/ftp/III/126/

"""

import os
import gzip

import numpy as np

from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region

BURNASHEV_ROOT = '/data/Burnashev/'
N_SPEC_POINTS = 200
DELTA_LAMBDA = 2.5*u.nm

class Burnashev():
    def __init__(self):
        self._cat = None
        self._stars = None
        
    @property
    def catalog(self):
        """Burnashev catalog, with star names only, no RA and DEC.  Some non-stellar objects, calibrations, etc. included"""
        if self._cat is not None:
            return self._cat
            
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
        self._cat = cat
        return cat

    @property
    def stars(self):
        """Star names, RA and DEC of Burnashev astronomical targets"""
        if self._stars is not None:
            return self._stars
        with open(os.path.join(BURNASHEV_ROOT, 'stars.dat')) as f:
            bstars = f.read()

        bstars = bstars.split('\n')
        stars = []
        for line in bstars:
            if len(line) == 0:
                break

            name = line[0:19].strip()
            RAh = int(line[19:22])
            RAm = int(line[22:25])
            RAs = float(line[25:30])
            DECd = int(line[30:34])
            DECm = int(line[34:37])
            DECs = float(line[37:40])

            c = SkyCoord(f'{RAh}h{RAm}m{RAs}s', f'{DECd}d{DECm}m{DECs}s')

            entry = {'Name': name,
                     'coord': c}
            stars.append(entry)
        self._stars = stars
        return stars

    def closest_name(self, coord):
        """Given SkyCoord, find closest star name in catalog"""
        angles = []
        for star in self.stars:
            a = my_star.separation(star['coord'])
            angles.append(a)
        min_angle = min(angles)
        min_idx = angles.index(min_angle)
        name = self.stars[min_idx]['Name']
        return name

    def entry_by_name(self, name):
        """Given a star's catalog name (or subset), return catalog entry"""
        for entry in self.catalog:
            if name in entry['Name']:
                return entry
        return None

    def entry_by_coord(self, coord):
        """"Returns closets catalog entry to star at coord"""
        name = self.closest_name(coord)
        entry = self.entry_by_name(name)
        return entry

    def get_spec(self, name_or_coord):
        """Returns a spectrum for a star.  Star specifided either by name or coordinate"""
        if isinstance(name_or_coord, str):
            entry_getter = self.entry_by_name
        elif isinstance(name_or_coord, SkyCoord):
            entry_getter = self.entry_by_coord
        entry = entry_getter(name_or_coord)
        if entry is None:
            log.warning('No entry found')
            return None
        lambdas = entry['lambda1']*u.nm + np.arange(N_SPEC_POINTS)*DELTA_LAMBDA
        flux = 10**entry['logE'] * u.milliWatt * u.m**-2 * u.cm**-1
        spec = Spectrum1D(spectral_axis=lambdas, flux=flux)
        return spec

# --> TODO: some sort of filter tool

#cat = read_cat()
#star = cat[234]
#spec = get_spec(star)
#filter_bandpass = SpectralRegion(5000*u.AA, 6000*u.AA)
#sub_spec = extract_region(spec, filter_bandpass)
#
#stars = read_stars()

b = Burnashev()
my_star = SkyCoord(f'12h23m42.2s', f'-26d18m22.2s')
name = b.closest_name(my_star)
print(b.entry_by_name(name))

print(b.entry_by_name('BS 0057'))

print(b.entry_by_name('Margaret'))

spec1 = b.get_spec('BS 0057')

spec2 = b.get_spec(my_star)
