#!/bin/bash

# Pipeline commands that are run by IoIO1U1 xfr.csh

cd /home/jpmorgen/py/IoIO/
source $(pipenv --venv)/bin/activate
python -m calibration --fits_fixed_ignore >> /data/IoIO/Logs/reduction/calibration_running.log 2>&1
# standard_star is broken right now and should be reworked to be more
# astropy-compliant with QTables.  Part of what broke it was a
# required upgrade of astroquery, needed to connect to JPL HORIZONS
#python -m standard_star --fits_fixed_ignore --write_summary_plots >> /data/IoIO/Logs/reduction/standard_star_running.log 2>&1
#python -m exoplanets --fits_fixed_ignore >> /data/IoIO/Logs/reduction/exoplanets_running.log 2>&1
# These might work as is or need an explicitly set previous verison of astropy
#python -m torus >> /data/IoIO/Logs/reduction/torus_running.log 2>&1
#python -m na_nebula >> /data/IoIO/Logs/reduction/na_nebula_running.log 2>&1
