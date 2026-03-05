#!/bin/bash

# Pipeline commands that are run by IoIO1U1 xfr.csh

# Sat Feb 28 16:44:16 2026 EST  jpmorgen@snipe
# Installed pyenv and installed a particular Python version:
# pyenv install 3.11.2
# In this directory,
# pyenv local 3.11.2 [which writes the .python-version file]
# In interactive mode, cd triggers pyenv to alter PATH to point to the
# version of python in .python-version.  However, now that bash is my
# shell, non-interactive remote commands do not source .profile or
# .bashrc, so I am running blind
#
# The virtual environment is maintained via venv, which is driven by
# the 3rd-party pipenv, which I installed at the Debian package level,
# which is why even without any of my PATH customizations, pipenv is
# found by this script.  Doing:
# pipenv install --python 3.11
# produced links to the desired version of python in the virtualenv's
# bin directory.  Felt like I needed to have pyenv installed for that
# cryptic 3.11 to really work (which it did).  Not sure what magic was
# really going on under the scenes for that to work.
#
# Used pipenv to install the needed packages.  That maintains the
# Pipfile and Pipefile.lock which makes this project portable.
# HOWEVER, it is most portable using vanilla Python, which Debian 12
# did not quite have (distutils was a different version).  Also,
# pipenv apparently doesn't do packaging.  That is yet another level
# and there are now 3rd party tools that do pipenv stuff and packaging
# all in one....
#
# When the virtualenv's activate script is sourced, the virtualenv's
# bin is inserted into the PATH.  Pipenv makes that easy, but really
# the whole Python version in this environment is determined by the
# virtualenv, not pyenv at that point.  Pyenv and Pipenv just add
# window dressing around to assemble the parts in the various
# directories (e.g. possibly putting things in the virtualenv's bin,
# certainly using pip to install packages in the virtualenv, etc.)
#
# Sun Mar 01 18:04:14 2026 EST  jpmorgen@snipe
# Ack.  My antiquated way of doing things without a proper package
# __init__.py relies on PYTHONPATH to find IoIO, precisionguide, etc.
# Fixed that partially by putting these stub projects into IoIO, but
# that didn't really solve the PYTHONPATH problem, since IoIO itself
# needs to be a package installed with "pip install -e <dir>" for
# continued local development without a PYTHONPATH
#

# --> This will go away when I make IoIO a proper package installed
# temporarily with pip install -e IoIO
if [ -z "${PYTHONPATH}" ]; then
    export PYTHONPATH=${HOME}/py
fi

# Was ising python -m instead of just calling the scipts directly to
# get around the fact that I had #!/usr/bin/python3 at the beginning
# of these scripts.  Change to more virtual environment-friendly
# #!/usr/bin/env python3

cd /home/jpmorgen/py/IoIO/
source $(pipenv --venv)/bin/activate
./calibration.py --fits_fixed_ignore >> /data/IoIO/Logs/reduction/calibration_running.log 2>&1
./standard_star.py --fits_fixed_ignore --write_summary_plots >> /data/IoIO/Logs/reduction/standard_star_running.log 2>&1
./exoplanets.py --fits_fixed_ignore >> /data/IoIO/Logs/reduction/exoplanets_running.log 2>&1
./torus.py >> /data/IoIO/Logs/reduction/torus_running.log 2>&1
./na_nebula.py >> /data/IoIO/Logs/reduction/na_nebula_running.log 2>&1
