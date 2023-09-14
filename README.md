The latest release of this software is available at

https://zenodo.org/badge/latestdoi/157604461

The Planetary Science Institute's Io Input/Output facility (IoIO) is a
small-aperture (35 cm) coronagraph constructed for the study of
material ejected from Jupiter's moon Io (the output) and entering
Jupiter's magnetosphere (the input).  A full description of the
instrument and its first results is contained in Morgenthaler, J.,
Rathbun, J., Schmidt, C., Baumgardner, J, Schneider, N. "Large
Volcanic Event on Io Inferred from Jovian Sodium Nebula Brightening,"
ApJL L23, 2019.  This archive contains the latest code for recording,
reducing and presenting the data.

Portions of this software depend on the following packages/software:

BigMultiPipe: https://zenodo.org/badge/latestdoi/329778044
CCDMultiPipe: https://zenodo.org/badge/latestdoi/475075489
Precisionguide: https://zenodo.org/badge/latestdoi/475075489
Burnashev: https://zenodo.org/badge/latestdoi/585668465
Astropy_FITS_key: https://zenodo.org/badge/latestdoi/361299826

Brief description of files in this repository:

remote_pipeline.sh: runs all components of the data processing pipeline

calibration.py: creates database of master biases, darkes and flats,
and picks best match for CCD image being processed

standard_star.py: processes Burnashev standard star observations,
creating flux calibration and extinction coefficient databases

exoplanets.py: photometric and astrometric processing of exoplanet
transit obesrvations

na_nebula.py and torus.py: process the IoIO Na nebula and Io plasma
torus (IPT) observations.

morgenthaler_2023.py: create time-history plot of Na nebula and IPT
for 2023 publication

movie.py: create movie of Na nebula and/or IPT images using PNGs
created by na_nebula.py and torus.py

fix*.py: these run sections of the pipeline system to fix minor bugs.
They will be phased out when the pipeline is rerun.

IoIO_old.py and precisionguide_old: code currently controlling the
telescope.  These will be phased out when the telescope control code
is transferred into the new version of precisionguide.

ioio.notebk: software developement notebook

