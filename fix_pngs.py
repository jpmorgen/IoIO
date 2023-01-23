#!/usr/bin/python3

"""Fix up outfname extension problem in 2017 and plot all PNGs on the same axes"""

import os

import numpy as np

import astropy.units as u
from astropy.table import QTable

from bigmultipipe import BigMultiPipe

from IoIO.utils import ColnameEncoder, plot_planet_subim, savefig_overwrite
from IoIO.cordata import CorData

#fname = '/data/IoIO/Na_nebula/20221117/Na_on-band_002-back-sub.fits'
#
#ccd = CorData.read(fname)
#
#plot_planet_subim(ccd,
#                  plot_planet_rot_from_key=['Jupiter_NPole_ang'],
#                  outname='/tmp/test.png',
#                  planet_subim_figsize=[6, 4],
#                  planet_subim_dx=45*u.R_jup,
#                  planet_subim_dy=40*u.R_jup)
                  

PROCESS_SIZE = 2E9 # one Na image seems to stay under 2G


def back_from_edge(summary_table=None, in_name=None, outname=None,
                   **kwargs):
    t = QTable.read(summary_table)
    sb_encoder = ColnameEncoder('annular_sb', formatter='.1f')
    largest_ap = sb_encoder.largest_colbase(t.colnames)
    idx = np.flatnonzero(t['outname'] == in_name)
    return np.squeeze(t[idx][largest_ap])

class FixPNGMultiPipe(BigMultiPipe):
    def __init__(self, **kwargs):                 
        super().__init__(num_processes=0.8,
                         process_size=PROCESS_SIZE,
                         **kwargs)
        
    def file_read(self, in_name, **kwargs):
        if not os.path.exists(in_name):
            print(f'ERROR: File not found {in_name}')
            return None
        return CorData.read(in_name)

    def outname_create(self, in_name, **kwargs):
        outroot, _ = os.path.splitext(in_name)
        return outroot + '.png'

    def file_write(self, data, outname, in_name=None, **kwargs):
        print(outname)
        plot_planet_subim(data, 
                          plot_planet_rot_from_key=['Jupiter_NPole_ang'],
                          outname=outname,
                          in_name=in_name,
                          **kwargs,)
        return outname

na_table = '/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv'
t = QTable.read(na_table)
t.sort('tavg')
n = [os.path.splitext(o)[0] + '.fits' for o in t['outname']]
t['outname'] = n
t.write(na_table, overwrite=True)

png_pipe = FixPNGMultiPipe()
png_pipe.pipeline(t['outname'], planet_subim_figsize=[6, 4],
                  planet_subim_dx=45*u.R_jup,
                  planet_subim_dy=40*u.R_jup,
                  summary_table=na_table,
                  planet_subim_backcalc=back_from_edge)

#t = QTable.read('/data/IoIO/Torus/Torus.ecsv')
#t.sort('tavg')
#n = [os.path.splitext(o)[0] + '.fits' for o in t['outname']]
#t['outname'] = n
#t.write('/data/IoIO/Torus/Torus.ecsv', overwrite=True)
#
#png_pipe = FixPNGMultiPipe()
#png_pipe.pipeline(t['outname'],
#                  planet_subim_dx=10*u.R_jup,
#                  planet_subim_dy=5*u.R_jup)
