import os

import numpy as np

from astropy import log
from astropy import units as u
from astropy.table import QTable

from ccdproc import ImageFileCollection

from ccdmultipipe import as_single

from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob,
                        dict_to_ccd_meta)
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, mask_nonlin_sat,
                               nd_filter_mask,
                               objctradec_to_obj_center)
from IoIO.calibration import Calibration
from IoIO.cor_photometry import (CorPhotometry, add_astrometry,
                                 write_photometry)
from IoIO.horizons import comet_ephemeris

BASE = 'Comets'
PSISCOPE_ROOT = '/data/PSIScope'
OUTDIR_ROOT = os.path.join(IoIO_ROOT, BASE)
COMET_ROOT = os.path.join(IoIO_ROOT, 'Comets')
COMET_GLOB_LIST = ['CK*', '[0-9][0-9][0-9][0-9]P-*']
COMET_PHOT_BOX = 5

def comet_flist(directory,
                glob_include=COMET_GLOB_LIST,
                glob_exclude_list=None,
                **kwargs):
    return multi_glob(directory, glob_include, glob_exclude_list)

def comet_list(raw_data_root=RAW_DATA_ROOT,
               include_PSIScope=True,
               start=None,
               stop=None,
               **kwargs):
    dirs_dates = get_dirs_dates(raw_data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return []
    flist = []
    n_nights = 0
    for d in dirs:
        cfl = comet_flist(d, **kwargs)
        if len(cfl) > 0:
            n_nights += 1
        flist.extend(cfl)
    if include_PSIScope:
        dirs_dates = get_dirs_dates(os.path.join(PSISCOPE_ROOT, 'raw'))
        dirs, _ = zip(*dirs_dates)
        if len(dirs) == 0:
            log.warning('No PSIScope directories found')
        else:
            for d in dirs:
                cfl = comet_flist(d, **kwargs)
                if len(cfl) > 0:
                    n_nights += 1
                flist.extend(comet_flist(d, **kwargs))
    cl = []
    for f in flist:
        bname = os.path.basename(f)
        sb = bname.split('-')
        cl.append(sb[0])
    return list(set(cl)), n_nights

class CometMultiPipe(CorMultiPipeBase):
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
        return written_name

def comet_phot(ccd,
               bmp_meta=None,
               comet_phot_box=COMET_PHOT_BOX,
               **kwargs):
    """cormulipipe post-processing routine to add up counts at obj_center"""
    ll = ccd.obj_center - comet_phot_box/2
    ur = ccd.obj_center + comet_phot_box/2
    ll = np.round(ll).astype(int)
    ur = np.round(ur).astype(int)
    patch = ccd[ll[0]:ur[0], ll[1]:ur[1]]
    sb = np.sum(patch) / np.prod(ur - ll)
    comet_dict = {'comet_phot': sb*ccd.unit,
                  'comet_phot_box': comet_phot_box*u.pixel}
    ccd_out = dict_to_ccd_meta(ccd, comet_dict)
    bmp_meta.update(comet_dict)
    return ccd_out

def comet_collection(directory,
                     glob_include=COMET_GLOB_LIST,
                     glob_exclude_list=None,
                     **kwargs):
    flist = comet_flist(directory, **kwargs)
    if len(flist) == 0:
        return ImageFileCollection(directory, glob_exclude='*')
    return ImageFileCollection(directory, filenames=flist)
    
def comet_pipeline(directory_or_collection=None,
                   calibration=None,
                   photometry=None,
                   num_processes=None,
                   outdir=None,
                   outdir_root=COMET_ROOT,
                   create_outdir=True,
                   **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = comet_collection(directory, **kwargs)

    if len(collection.files) == 0:
        return
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True, **kwargs)
    outdir = outdir or reduced_dir(directory, outdir_root, create=False)

    cmp = CometMultiPipe(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        create_outdir=create_outdir,
        post_process_list=[mask_nonlin_sat,
                           nd_filter_mask,
                           add_astrometry,
                           comet_ephemeris,
                           objctradec_to_obj_center,
                           comet_phot,
                           as_single],
        **kwargs)
    pout = cmp.pipeline([collection.files[0]], outdir=outdir, overwrite=True)
    return pout

def comet_directory(directory_or_collection,
                    read_pout=True,
                    write_pout=True,
                    write_plot=True,
                    outdir=None,
                    outdir_root=OUTDIR_ROOT,
                    create_outdir=True,
                    show=False,
                    **kwargs):

    if isinstance(directory_or_collection, ImageFileCollection):
        # We are passed a collection when running multiple directories
        # in parallel
        directory = directory_or_collection.location
        collection = directory_or_collection
    else:
        directory = directory_or_collection
        collection = cometcollection(directory, **kwargs)

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
    poutname = os.path.join(outdir, BASE + '.pout')
    pout = cached_pout(comet_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       directory_or_collection=collection,
                       outdir=outdir,
                       create_outdir=create_outdir,
                       **kwargs)
    if pout is None or len(pout) == 0:
        return QTable()

    _ , pipe_meta = zip(*pout)
    t = QTable(rows=pipe_meta)
    return t

log.setLevel('DEBUG')
#comet_pipeline('/data/IoIO/raw/20211017')
#comet_pipeline('/data/IoIO/raw/20211028')

#c = comet_collection('/data/IoIO/raw/20211028')
#c = comet_collection('/data/IoIO/raw/20220112/')

cl = comet_list(include_PSIScope=False)
print(f'Number of IoIO comets = {len(cl[0])} Total observations {cl[1]}')
cl = comet_list()
print('Including PSIScope')
print(len(cl[0]), cl[1])

#pout = comet_pipeline('/data/IoIO/raw/20211028',
#                      fits_fixed_ignore=True)
