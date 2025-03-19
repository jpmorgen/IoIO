#!/usr/bin/python3

import os

import numpy as np

from astropy import log
import astropy.units as u
from astropy.table import QTable, MaskedColumn

from bigmultipipe import cached_pout, prune_pout

from ccdmultipipe import as_single

from IoIO.ioio_globals import IoIO_ROOT, RAW_DATA_ROOT
from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob)
from IoIO.cormultipipe import (CorMultiPipeBase, mask_nonlin_sat,
                               nd_filter_mask)
from IoIO.calibration import Calibration
from IoIO.cor_photometry import (CorPhotometry, add_astrometry,
                                 write_photometry)

BASE = 'GPS_Satellites'
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)

GPS_GLOB_LIST = ['Calibration*', 'Autosave*', 'GPS_Satellite*']

class GPSMultiPipe(CorMultiPipeBase):
    def pre_process(self, data, **kwargs):
        # Take out full-frame requirement
        return (data, kwargs)

    def file_write(self, ccd, outname,
                   in_name=None, photometry=None,
                   **kwargs):
        written_name = super().file_write(
            ccd, outname, photometry=photometry, **kwargs)
        write_photometry(in_name=in_name, outname=outname,
                         photometry=photometry,
                         write_wide_source_table=True,
                         **kwargs)
        outroot, _ = os.path.splitext(outname)
        #try:
        #    photometry.plot_object(outname=outroot + '.png')
        #except Exception as e:
        #    log.warning(f'Not able to plot object for {outname}: {e}')
        return written_name

def gps_pipeline(calibration=None,
                 photometry=None,
                 glob_include=GPS_GLOB_LIST,
                 glob_exclude_list=None,
                 fits_fixed_ignore=True,
                 num_processes=None,
                 **kwargs):

    calibration = calibration or Calibration(reduce=True)
    photometry = photometry or CorPhotometry(precalc=True, **kwargs)

    fnames = multi_glob(directory, glob_include, glob_exclude_list)

    cmp = GPSMultiPipe(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        fail_if_no_solve=True,
        post_process_list=[mask_nonlin_sat,
                           nd_filter_mask,
                           add_astrometry,
                           as_single],
        fits_fixed_ignore=fits_fixed_ignore, 
        num_processes=num_processes,
        **kwargs)
    #pout = cmp.pipeline([fnames[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(fnames, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, fnames)
    if len(pout) == 0:
        log.warning(f'No good observations in series {fnames[0]}')
        return pout
    return pout

    
log.setLevel('DEBUG')

# This might be something like gps_directory
# --> with gps_tree, there should be a start at 2025-01-24, since I
# have no observations before that time
#directory='/data/IoIO/raw/2025-01-24/'
directory='/data/IoIO/raw/2025-03-19/'
outdir=None
outdir_root=OUTDIR_ROOT
read_pout=True
write_pout=True
create_outdir=True
glob_include=GPS_GLOB_LIST
glob_exclude_list=None
max_sep=5*u.arcsec
kwargs={}


outdir = outdir or reduced_dir(directory, outdir_root, create=False)
poutname = os.path.join(outdir, BASE + '.pout')
pout = cached_pout(gps_pipeline,
                   poutname=poutname,
                   read_pout=read_pout,
                   write_pout=write_pout,
                   create_outdir=create_outdir,
                   directory=directory,
                   glob_include=glob_include,
                   glob_exclude_list=glob_exclude_list,
                   **kwargs)

_ , pipe_meta = zip(*pout)
t = QTable(rows=pipe_meta)

t.sort('outname')
csvs = [f.replace('.fits', '.ecsv') for f in t['outname']]

c0 = QTable.read(csvs[0])
c1 = QTable.read(csvs[1])

# The shape of these returns matches c0
idx, d2d, d3d = c0['coord'].match_to_catalog_sky(c1['coord'])
colmask = d2d > max_sep
c0[csvs[1]] = MaskedColumn(data=d2d, name=csvs[1], mask=colmask)



# So this is the set of c0 coords that have matched c1
#c0[sep_constraint]
# This is the index into c0 of the matches.  This is probably what I
# should store on a per c1 basis to then compare to see if they are
# consistent.  Alterntely, I can add a masked column to c0 with these
# results.  The column name would have to be some sort of encoded
# DATE-AVG.  Oh, maybe better yet, the filename of the ecsv
#idx[sep_constraint]

#for obj_coord in c0['coord']:
#    seps = obj_coord.separation(c1['coord'])
#    match = np.argmin(seps)
#    if seps[match] > 5*u.arcsec:
#        continue


# from cordata import CorData
# from cor_process import cor_process
# ccd = CorData.read('/data/IoIO/raw/2025-01-24/Calibration-0010.fit')
# calibration = Calibration(reduce=True)
#
# ccd_out = cor_process(ccd, calibration=calibration,
#                       oscan=True,
#                       gain=True,
#                       error=True,
#                       min_value=True)





## This might be something like gps_pipeline (gps_directory is merged
## in here at the moment)
#directory='/data/IoIO/raw/2025-01-24/'
#glob_include=GPS_GLOB_LIST
#glob_exclude_list=None
#outdir=None
#outdir_root=OUTDIR_ROOT
#create_outdir=True
#calibration=None
#photometry=None
#kwargs={}
#
#calibration = calibration or Calibration(reduce=True)
#photometry = photometry or CorPhotometry(precalc=True, **kwargs)
#
#
#fnames = multi_glob(directory, glob_include, glob_exclude_list)
#poutname = os.path.join(outdir, BASE + '.pout')
