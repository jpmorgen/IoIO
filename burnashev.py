"""Module for Burnashev (1985) spectrophotometric standard star data

NOTE: Data must be dowloaded by hand from 

https://cdsarc.cds.unistra.fr/ftp/III/126/

Put all the files into a directory and not that directory in
BURNASHEV_ROOT

Hand download is required because astroquery.vizier is broken for this
catalog.  The spectra are stored as many columns in each star's row.
An astropy.vizier query returns the star names and coordinates, but
stores each spectrum as the string "Spectrum."

TODO: make a non-vizier download client for astroquerey

"""

import os
import gzip

import numpy as np
from scipy import integrate

import matplotlib.pyplot as plt

from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import (extract_region,
                                    LinearInterpolatedResampler)

from specutils.analysis import line_flux

BURNASHEV_ROOT = '/data/Burnashev/'
N_SPEC_POINTS = 200
DELTA_LAMBDA = 2.5*u.nm

class Burnashev():
    def __init__(self):
        self._cat = None
        self._stars = None
        self._catalog_coords = None        

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

    @property
    def catalog_coords(self):
        """Store catalog star coordinates in
        `~astropy.coordinates.SkyCoord` object
        """
        if self._catalog_coords is not None:
            return self._catalog_coords
        # This could also be done with a Pandas dataframe slice
        coords = [s['coord'] for s in self.stars]
        self._catalog_coords = SkyCoord(coords)
        return self._catalog_coords

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
        idx, angle, d3d = coord.match_to_catalog_sky(self.catalog_coords)
        name = self.stars[idx]['Name']
        return (name, angle)


        #angles = []
        #for star in self.stars:
        #    a = coord.separation(star['coord'])
        #    angles.append(a)
        #min_angle = min(angles)
        #min_idx = angles.index(min_angle)
        #name = self.stars[min_idx]['Name']
        #return (name, min_angle)

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
        lambdas = entry['lambda1']*u.AA + np.arange(N_SPEC_POINTS)*DELTA_LAMBDA
        flux = 10**entry['logE'] * u.erg/u.s/u.cm**2/u.cm
        #flux = 10**entry['logE'] * u.milliWatt * u.m**-2 * u.cm**-1
        spec = Spectrum1D(spectral_axis=lambdas, flux=flux)
        return spec


# Specific to IoIO
def get_filt(fname, **kwargs):
    # https://github.com/astropy/specutils/issues/880
    filt_arr = np.loadtxt(fname, **kwargs)
    wave = filt_arr[:,0]*u.nm
    trans = filt_arr[:,1]
    # Clean up datatheif reading
    #trans = np.asarray(trans)
    trans[trans < 0] = 0
    trans *= u.percent
    filt = Spectrum1D(spectral_axis=wave, flux=trans)
    return filt
        
def flux_in_filt(spec, filt, resampler=None, energy=False):
    # Make filter spectral axis unit consistent with star spectrum.
    # Need to make a new filter spectrum as a result
    filt_spectral_axis = filt.spectral_axis.to(spec.spectral_axis.unit)
    filt = Spectrum1D(spectral_axis=filt_spectral_axis,
                      flux=filt.flux)
    filter_bandpass = SpectralRegion(np.min(filt.spectral_axis),
                                     np.max(filt.spectral_axis))
    # Work only with our filter bandpass
    spec = extract_region(spec, filter_bandpass)
    
    if (len(spec.spectral_axis) == len(filt.spectral_axis)
        and np.all((spec.spectral_axis == filt.spectral_axis))):
        # No need to resample
        pass
    else:
        if resampler is None:
            resampler = LinearInterpolatedResampler()
        # Resample lower resolution to higher
        spec_med_dlambda = np.median(spec.spectral_axis[1:]
                                     - spec.spectral_axis[0:-1])
        filt_med_dlambda = np.median(filt.spectral_axis[1:]
                                 - filt.spectral_axis[0:-1])
        if spec_med_dlambda > filt_med_dlambda:
            spec = resampler(spec, filt.spectral_axis)
        else:
            filt = resampler(filt, spec.spectral_axis)

    filt = resampler(filt, spec.spectral_axis)
    #f, ax = plt.subplots()
    #ax.step(filt.spectral_axis, filt.flux)
    #ax.set_title(f"{filt_name}")
    #plt.show()

    spec = spec * filt
    # f, ax = plt.subplots()
    # ax.step(spec.spectral_axis, spec.flux)
    # ax.set_title(f"{filt_name}")
    # plt.show()

    if energy:
        filt_flux = line_flux(spec)
        filt_flux = filt_flux.to(u.erg * u.cm**-2 * u.s**-1)
    else:
        # Photon.  Integrate by hand with simple trapezoidal,
        # non-interpolated technique.  THis is OK, since our filter
        # curves are nicely sampled
        spec_dlambda = spec.spectral_axis[1:] - spec.spectral_axis[0:-1]
        av_bin_flux = (spec.photon_flux[1:] + spec.photon_flux[0:-1])/2
        filt_flux = np.nansum(spec_dlambda*av_bin_flux)
    return filt_flux

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

    
# Check the standard Alpha Lyrae
my_star = SkyCoord('18h 36m 56.33635s', '+38d 47m 01.2802s')

b = Burnashev()
# my_star = SkyCoord(f'12h23m42.2s', f'-26d18m22.2s')
name, dist = b.closest_name_to(my_star)
orig_spec = b.get_spec(name)

filt_root = '/data/IoIO/observing'
filt_names = ['U', 'B', 'V', 'R', 'I',
              'SII_on', 'SII_off', 'Na_on', 'Na_off']

for filt_name in filt_names:
    fname = os.path.join(filt_root, filt_name+'.txt')
    try:
        filt = get_filt(fname)
    except:
        filt = get_filt(fname, delimiter=',')
    #ax = plt.subplots()[1]  
    #ax.plot(filt.spectral_axis, filt.flux)  
    #ax.set_xlabel("Wavelenth")  
    #ax.set_ylabel("Transmission")
    #ax.set_title(f"{filt_name}")
    #plt.show()

    spec = orig_spec
    filt_flux = flux_in_filt(orig_spec, filt)
    print(f'{filt_name} flux = {filt_flux}')
    
    ## Make filter spectral axis consistent with star spectrum.  Need
    ## to make a new filter spectrum as a result
    #filt_spectral_axis = filt.spectral_axis.to(spec.spectral_axis.unit)
    #filt = Spectrum1D(spectral_axis=filt_spectral_axis,
    #                  flux=filt.flux)
    #filter_bandpass = SpectralRegion(np.min(filt.spectral_axis),
    #                                 np.max(filt.spectral_axis))
    ## Work only with our bandpass
    #spec = extract_region(spec, filter_bandpass)
    ##resampler = FluxConservingResampler()
    #resampler = LinearInterpolatedResampler()
    ##resampler = SplineInterpolatedResampler()
    #spec = resampler(spec, filt.spectral_axis) 
    ##filt = resampler(filt, spec.spectral_axis)
    #f, ax = plt.subplots()
    #ax.step(filt.spectral_axis, filt.flux)
    #ax.set_title(f"{filt_name}")
    #plt.show()
    #
    #spec = spec * filt
    #f, ax = plt.subplots()
    #ax.step(spec.spectral_axis, spec.flux)
    #ax.set_title(f"{filt_name}")
    #plt.show()
    #
    #dlambda = filter_bandpass.upper - filter_bandpass.lower
    #bandpass_flux = np.nansum(spec.flux)*dlambda
    #bandpass_flux = bandpass_flux.to(u.erg/u.s/u.cm**2)
    #print(f'{filt_name} flux = {bandpass_flux}')

