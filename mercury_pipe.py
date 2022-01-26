#!/usr/bin/python3
"""Reduce Mercury observations"""

import os
import glob

from astropy import log, time, units as u

from ccdmultipipe import as_single

from cormultipipe import (RAW_DATA_ROOT,
                          get_dirs_dates, reduced_dir,
                          Calibration, OffCorMultiPipe, FixFnameCorMultipipe,
                          mask_nonlin_sat, detflux)

MERCURY_ROOT = '/data/Mercury'
GLOB_INCLUDE = ['Mercury*']

class FixFnameOffCorMultipipe(FixFnameCorMultipipe, OffCorMultiPipe):
    pass

if __name__ == "__main__":
    log.setLevel('DEBUG')

    # --> Need a start-stop with this and might make this a general
    # --> routine that accepts the CorMultiPipe object and arguments
 
    data_root = RAW_DATA_ROOT
    reduced_root = MERCURY_ROOT

    start = "2021-09-01"
    stop = "2021-12-31"
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)

    dirs, _ = zip(*dirs_dates)
    assert len(dirs) > 0

    c = Calibration(reduce=True)

    cmp = OffCorMultiPipe(auto=True, calibration=c,
                          fits_fixed_ignore=True, outname_ext='.fits', 
                          post_process_list=[mask_nonlin_sat, detflux,
                                             as_single])
    for d in dirs:
        flist = []
        for gi in GLOB_INCLUDE:
            flist += glob.glob(os.path.join(d, gi))
        if len(flist) == 0:
            log.debug(f'No observations in {d}')
            continue
        reddir = reduced_dir(d, reduced_root)
        pout = cmp.pipeline(flist, outdir=reddir,
                            create_outdir=True, overwrite=True)

