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

from astroquery.vizier import Vizier

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
        """Returns raw Burnashev catalog as list of dict.  

        """
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
        """Returns list dict of star name fields used in Burnashev
        catalog together with their RA and DEC.  

        """
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

    def closest_name_to(self, coord):
        """Find Burnashev name field and angular distance of closest
        catalog star to coord

        This method is important since the primary index of the the
        Burnashev catalog is BS designation (Bright Star catalogue,
        5th Revised Ed.; Hoffleit et al., 1991).  SIMBAD does not
        provide BS designations as IDs when searching using other
        catalog designations.  In other words, a SIMBAD search on BS
        24 will yield HD 493 as a valid identifier, but not the other
        way around.  Thus, the Burnashev catalog is most conveniently
        searched by coordinate.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Input coordinate

        Returns
        -------
        name, min_angle: tuple
            Burnashev catalog name field of star to `coord` and its angular
            distance from `coord`.

        See also
        -------
        `:meth:Burnashev.entry_by_name`

        """

        angles = []
        for star in self.stars:
            a = coord.separation(star['coord'])
            angles.append(a)
        min_angle = min(angles)
        min_idx = angles.index(min_angle)
        name = self.stars[min_idx]['Name']
        return (name, min_angle)

    def entry_by_name(self, name):
        """Return Burnashev catalog entry given its name

        Parameters
        ----------
        name: str
            Full or partial match to Burnashev catalog name field.
            Name fields are a concatenation of Bright Star catalogue,
            5th Revised Ed. (Hoffleit et al., 1991) designations in
            the form "BS NNNN" plus common identifiers in various
            abbreviated forms. e.g.: "BS 0015 ALF AND".  

        Returns
        -------
        entry : dict
            Burnashev catalog entry for name.  WARNING: no check for
            multiple matches is done -- only the first entry is
            returned.  

        See also
        --------
        `:meth:Burnashev.closest_name_by_coord`

        """
        # https://stackoverflow.com/questions/8653516/python-list-of-dictionaries-search
        entry = next((e for e in self.catalog if name in e['Name']), None)
        return entry

    def get_spec(self, name):
        """Returns a spectrum for a Burnashev spectrophotometric
        standard star.

        Parameters
        ----------
        name : str
            Burnashev catalog name field or subset thereof.  WARNING:
            it is assumed that `:meth:Burnashev.closest_by_name` has
            been used to retrieve the proper Burnashev name field.
            That said, names in the form "BS NNNN" may be safe. 

        """
        entry = self.entry_by_name(name)
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

#b = Burnashev()
#my_star = SkyCoord(f'12h23m42.2s', f'-26d18m22.2s')
#name, dist = b.closest_name_to(my_star)
#print(f'distance to {name} is {dist}')
#print(b.entry_by_name(name))
#
#print(b.entry_by_name('BS 0057'))
#
#print(b.entry_by_name('Margaret'))
#
#spec1 = b.get_spec('BS 0057')
#
#spec2 = b.get_spec(name)


burnashev_vizier_entry = Vizier.find_catalogs('Burnashev')
burnashev_catalog = Vizier.get_catalogs(burnashev_vizier_entry.keys())
