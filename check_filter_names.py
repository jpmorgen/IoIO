#!/usr/bin/python3

"""IoIO CCD bias, dark and flat calibration system"""

import os
import argparse

from astropy import log
from astropy.io.fits import getheader

from utils import get_dirs_dates
from cor_process import standardize_filt_name
from cormultipipe import RAW_DATA_ROOT

#import os
#import re
#import time
#import psutil
#import datetime
#import glob
#import argparse
#from pathlib import Path
#
#import numpy as np
#from scipy import interpolate
#
#import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator
#
#import pandas as pd
#
#from astropy import log
#from astropy import units as u
#from astropy.io.fits import Header, getheader
#from astropy.table import QTable
#from astropy.nddata import CCDData
#from astropy.time import Time
#from astropy.stats import mad_std, biweight_location
#
#from photutils import Background2D, MedianBackground
#import ccdproc as ccdp
#
#from bigmultipipe import WorkerWithKwargs, NestablePool
#from bigmultipipe import num_can_process, prune_pout
#
#import sx694
#
#from utils import assure_list, get_dirs_dates, add_history
#
#from cordata_base import CorDataBase, CorDataNDparams
#from cormultipipe import IoIO_ROOT, RAW_DATA_ROOT
#from cormultipipe import MAX_NUM_PROCESSES, MAX_CCDDATA_BITPIX, MAX_MEM_FRAC
#from cormultipipe import COR_PROCESS_EXPAND_FACTOR, ND_EDGE_EXPAND
#from cormultipipe import CorMultiPipeBase, CorMultiPipeNDparams
#from cormultipipe import light_image, mask_above_key

def filt_check_dir(directory):
    log.info(f'Checking {directory}')
    fnames = os.listdir(directory)
    for f in fnames:
        _, extension = os.path.splitext(f)
        if extension not in ['.fits', '.fit']:
            continue
        isbadname = False
        badnames = ['oving_to', 'Mercury', 'Venus', '_sequence', 'PinPoint']
        for bn in badnames:
            if bn in f:
                isbadname = True
                break
        if isbadname:
            continue

        try:
            hdr = getheader(os.path.join(directory, f))
            if hdr['IMAGETYP'] != 'LIGHT':
                continue         
            hdr = standardize_filt_name(hdr)
            filt = hdr['FILTER']
            ofilt = hdr.get('OFILTER')
            
        except Exception as e:
            log.info(f'{e} {os.path.join(directory, f)}')
            continue

        if filt == 'open':
            continue

        # See if we can match our filt to the fname
        if filt in f:
            # Success, so just move on quietly
            continue

        if ofilt and ofilt in f:
            # Old filter matches
            continue

        if ('IPT_Na_R' in f
            or 'Na_IPT_R' in f
            or 'PrecisionGuideDataFile' in f):
            # These are sequence names in early 2018 that is just too inscrutable
            continue

        ## Try some cases I have run across in 2017 and 2018
        if 'IPT-' in f:
            line = 'SII'
        elif 'Na' in f:
            line = 'Na'
        else:
            line = ''
        #if 'on-band' in f:
        #    on_off = 'on'
        #elif 'off-band' in f:
        #    on_off = 'off'
        #else:
        #    on_off = ''

        if 'cont' in f:
            on_off = 'off'
        if 'on' in f:
            on_off = 'on'
        elif 'off' in f:
            on_off = 'off'
        else:
            on_off = ''

        if f'{line}_{on_off}' != filt:
            log.error(f'FILTER = {filt}; {os.path.join(directory, f)}')

        #else:
        #    fname_compare = f        
        #
        #    if 'on-band.fit' in bf:
        #        on_off = 'on'
        #    elif 'off-band.fit' in bf:
        #        on_off = 'off'


def filt_check_tree(directory=RAW_DATA_ROOT):
    dirs = [dd[0] for dd in get_dirs_dates(directory)]
    for d in dirs:
        filt_check_dir(d)

def filt_check_cmd(args):
    if args.directory:
        filt_check_dir(args.directory)
    else:
        filt_check_tree(args.tree)

if __name__ == "__main__":
    log.setLevel('DEBUG')
    parser = argparse.ArgumentParser('Spot inconsistencies between filter names')
    parser.add_argument(
        '--tree',
        help=f'(default action) Root of directory tree to check (default directory: {RAW_DATA_ROOT})',
        metavar='DIRECTORY',
        default=RAW_DATA_ROOT)
    parser.add_argument(
        '--directory',
        help=f'Single directory to check')
    parser.set_defaults(func=filt_check_cmd)

    # Final set of commands that makes argparse work
    args = parser.parse_args()
    # This check for func is not needed if I make subparsers.required = True
    if hasattr(args, 'func'):
        args.func(args)
