#! /bin/bash

# Pipeline commands that are run by IoIO1U1 xfr.csh

/home/jpmorgen/py/IoIO/calibration.py --fits_fixed_ignore >> /data/IoIO/Logs/reduction/calibration_running.log 2>&1
/home/jpmorgen/py/IoIO/standard_star.py --fits_fixed_ignore --write_summary_plots >> /data/IoIO/Logs/reduction/standard_star_running.log 2>&1
/home/jpmorgen/py/IoIO/exoplanets.py --fits_fixed_ignore >> /data/IoIO/Logs/reduction/exoplanets_running.log 2>&1
#/home/jpmorgen/py/IoIO/torus.py >> /data/IoIO/Logs/reduction/torus_running.log 2>&1
#/home/jpmorgen/py/IoIO/na_nebula.py >> /data/IoIO/Logs/reduction/na_nebula_running.log 2>&1
