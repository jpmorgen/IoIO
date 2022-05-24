from astropy import log

from ccdmultipipe import as_single

from IoIO.utils import (reduced_dir, get_dirs_dates, multi_glob)
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT, CorMultiPipeBase,
                               mask_nonlin_sat, nd_filter_mask)
from IoIO.calibration import Calibration
from IoIO.cor_photometry import CorPhotometry, add_astrometry

COMET_ROOT = os.path.join(IoIO_ROOT, 'Comets')

def comet_pipeline(directory=None, # raw directory
                   glob_include='0029P*',
                   calibration=None,
                   photometry=None,
                   num_processes=None,
                   outdir=None,
                   outdir_root=COMET_ROOT,
                   create_outdir=True,
                   **kwargs):

    flist = multi_glob(directory, glob_list=glob_include)
    if len(flist) == 0:
        return
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = CorPhotometry(precalc=True, **kwargs)
    rd = reduced_dir(directory, outdir_root, create=False)

    cmp = CorMultiPipeBase(auto=True,
                           calibration=calibration,
                           photometry=photometry,
                           fits_fixed_ignore=True, outname_ext='.fits',
                           post_process_list=[mask_nonlin_sat,
                                              nd_filter_mask,
                                              add_astrometry,
                                              as_single],
                           create_outdir=create_outdir,
                       **kwargs)
    #pout = cmp.pipeline([flist[0]], outdir='/tmp', overwrite=True)
    pout = cmp.pipeline(flist, outdir=rd, overwrite=True)

log.setLevel('DEBUG')
#comet_pipeline('/data/IoIO/raw/20211017')
comet_pipeline('/data/IoIO/raw/20211028')
