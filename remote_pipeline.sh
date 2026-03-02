#!/bin/bash

# Pipeline commands that are run by IoIO1U1 xfr.csh

# Sat Feb 28 16:44:16 2026 EST  jpmorgen@snipe
# Installed pyenv and did a
# pyenv install 3.11.2
# In this directory,
# pyenv local 3.11.2 [which writes the .python-version file]
# Then cd triggers pyenv to alter PATH to point to the version of
# python in .python-version
# Used pipenv to install the needed packages.  That maintains the
# Pipfile and Pipefile.lock which makes this project portable.
# HOWEVER, it is most portable using vanilla Python, which Debian 12
# did not quite have (distutils was a different version)
#
# Sun Mar 01 18:04:14 2026 EST  jpmorgen@snipe
# Ack.  My antiquated way of doing things without a proper package
# __init__.py relies on PYTHONPATH to find IoIO, precisionguide.  In
#

# --> This will go away when I refactor the hard-coded paths away and
# make a proper package structure
if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH=${HOME}/py
fi

cd /home/jpmorgen/py/IoIO/
source $(pipenv --venv)/bin/activate
python -m calibration --fits_fixed_ignore >> /data/IoIO/Logs/reduction/calibration_running.log 2>&1
python -m standard_star --fits_fixed_ignore --write_summary_plots >> /data/IoIO/Logs/reduction/standard_star_running.log 2>&1
python -m exoplanets --fits_fixed_ignore >> /data/IoIO/Logs/reduction/exoplanets_running.log 2>&1
python -m torus >> /data/IoIO/Logs/reduction/torus_running.log 2>&1
python -m na_nebula >> /data/IoIO/Logs/reduction/na_nebula_running.log 2>&1
