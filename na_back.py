"""Find telluric component for Na image"""

import gc
import os
import pickle
from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import julian2num#, datestr2num

import pandas as pd

from astropy import log
from astropy import units as u
from astropy.time import Time
from astropy.nddata import CCDData
from astropy.stats import mad_std, biweight_location
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris, get_body

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile
from bigmultipipe import cached_pout, prune_pout

# --> These need to be cleaned up
from cormultipipe import RAW_DATA_ROOT, NA_OFF_ON_RATIO
from cormultipipe import CorMultiPipe, FwRedCorData, Calibration
from cormultipipe import reduced_dir, get_dirs_dates, valid_long_exposure
from cormultipipe import multi_row_selector, closest_in_time
from cormultipipe import multi_filter_proc, combine_masks
from cormultipipe import im_med_min_max, add_history

from IoIO.utils import savefig_overwrite
from photometry import Photometry

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

def na_back_process(data,
                    in_name=None,
                    bmp_meta=None,
                    photometry=None,
                    n_back_boxes=N_BACK_BOXES,
                    show=False,
                    off_on_ratio=NA_OFF_ON_RATIO,
                    **kwargs):
    """post-processing routine that processes a *pair* of ccd images
    in the order on-band, off-band"""
    if bmp_meta is None:
        bmp_meta = {}
    if photometry is None:
        photometry = Photometry(precalc=True,
                                n_back_boxes=n_back_boxes,
                                **kwargs)
    raoff0 = data[0].meta.get('RAOFF') or 0
    decoff0 = data[0].meta.get('DECOFF') or 0
    fluxes = []
    for ccd in data:
        raoff = ccd.meta.get('RAOFF') or 0
        decoff = ccd.meta.get('DECOFF') or 0
        if raoff != raoff0 or decoff != decoff0:
            log.warning(f'Mismatched RAOFF and DECOFF, skipping {in_name}')
            bmp_meta.clear()
            return None
        if (raoff**2 + decoff**2)**0.5 < 15:
            log.warning(f'Offset RAOFF {raoff} DECOFF {decoff} too small, skipping {in_name}')
            bmp_meta.clear()
            return None
        photometry.ccd = ccd
        exptime = ccd.meta['EXPTIME']*u.s
        flux = photometry.back_obj.background / exptime
        fluxes.append(flux)

    background = fluxes[0] - fluxes[1]/off_on_ratio
    if show:
        simple_show(background.value)

    # Unfortunately, a lot of data were taken with the filter wheel
    # moving.  This uses the existing bias light/dark patch routine to
    # get uncontaminated part --> consider making this smarter
    best_back, _ = im_med_min_max(background)

    # Note that we are using ccd, which at this point is the off-band
    # image.  But the on and off metadata for these quantities should
    # the same for both

    # Put time and sun angle into bmp_meta
    date_obs = ccd.meta.get('DATE-AVG') or ccd.meta.get('DATE-OBS')
    just_date, _ = date_obs.split('T')
    objctra = ccd.meta['OBJCTRA']
    objctdec = ccd.meta['OBJCTDEC']
    tm = Time(date_obs, format='fits')
    lon = ccd.meta.get('LONG-OBS') or ccd.meta.get('SITELONG')
    lat = ccd.meta.get('LAT-OBS') or ccd.meta.get('SITELAT')
    alt = ccd.meta.get('ALT-OBS') or 1096.342 * u.m
    loc = EarthLocation(lat=lat, lon=lon, height=alt)
    with solar_system_ephemeris.set('builtin'):
        sun = get_body('sun', tm, loc)
    ra = Angle(objctra, unit=u.hour)
    dec = Angle(objctdec, unit=u.deg)
    # Beware the default frame of SkyCoord is ICRS, which is relative
    # to the solar system Barycenter and jup [sun] is returned in GCRS,
    # which is relative ot the earth's center-of-mass.  separation()
    # is not commutative when the two different frames are used, when
    # one includes a solar system object (e.g. Jupiter), since the 3D
    # position of the point of reference and one of the objects is
    # considered.  Specifying the GCRS frame of jup for our telescope
    # RA and DEC SkyCoord does no harm for non-solar system objects
    # (distance is too far to matter) but does set the observing time,
    # which also does us no harm in this case, since it happens to be
    # the actual observing time.
    this_pointing = SkyCoord(frame=sun.frame, ra=ra, dec=dec)
    sun_angle = this_pointing.separation(sun)


    # Mesosphere is above the stratosphere, where the density of the
    # atmosphere diminishes to very small values.  So all attenuation
    # has already happened by the time we get up to the mesospheric
    # sodium layer
    # https://en.wikipedia.org/wiki/Atmosphere_of_Earth#/media/File:Comparison_US_standard_atmosphere_1962.svg
    airmass = data[0].meta.get('AIRMASS')
    tmeta = {'best_back': best_back.value,
             'date': just_date,
             'date_obs': tm,
             'jd': tm.jd,
             'raoff': raoff0,
             'decoff': decoff0,
             'ra': objctra,
             'dec': objctdec,
             'alt': objctalt,
             'az': objctaz,
             'airmass': airmass,
             'sun_angle': sun_angle.value}
    bmp_meta['Na_back'] = tmeta

    # In production, we don't plan to write the file, but prepare the
    # name just in case
    bmp_meta['outname'] = f'Jupiter_raoff_{raoff}_decoff_{decoff}_airmass_{airmass:.2}.fits'
    # Return one image
    data = CCDData(background, meta=data[0].meta, mask=data[0].mask)
    data.meta['OFFBAND'] = in_name[1]
    data.meta['HIERARCH N_BACK_BOXES'] = (n_back_boxes, 'Background grid for photutils.Background2D')
    data.meta['BESTBACK'] = (best_back.value,
                             'Best background value (electron/s)')
    add_history(data.meta,
                'Subtracted OFFBAND, smoothed over N_BACK_BOXES')
    return data

def na_back_pipeline(directory=None,
                     glob_include='Jupiter*',
                     calibration=None,
                     photometry=None,
                     n_back_boxes=N_BACK_BOXES,
                     num_processes=None,
                     outdir=None,
                     create_outdir=True,
                     **kwargs):

    outdir = outdir or reduced_dir(directory, create=False)
    collection = ImageFileCollection(directory,
                                     glob_include=glob_include)
    if collection is None:
        return []
    summary_table = collection.summary
    #print(summary_table['raoff'])
    try:
        raoffs = collection.values('raoff', unique=True)
        decoffs = collection.values('decoff', unique=True)
    except Exception as e:
        log.error(f'Problem with RAOFF/DECOFF in {directory}: {e}')
        return []
    f_pairs = []
    for raoff in raoffs:
        for decoff in decoffs:
            try:
                subc = collection.filter(raoff=raoff, decoff=decoff)
            except:
                log.debug(f'No match for RAOFF = {raoff} DECOFF = {decoff}')
                continue
            fp = closest_in_time(subc, ('Na_on', 'Na_off'),
                                 valid_long_exposure,
                                 directory=directory)
            f_pairs.extend(fp)
    if len(f_pairs) == 0:
        log.warning(f'No matching set of Na background files found '
                    f'in {directory}')
        return []

    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = Photometry(precalc=True,
                                n_back_boxes=n_back_boxes,
                                **kwargs)
        
    # Eventually put back in
    cmp = CorMultiPipe(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        create_outdir=create_outdir,
        post_process_list=[multi_filter_proc,
                           combine_masks,
                           na_back_process,
                           no_outfile],                           
        num_processes=num_processes,
        process_expand_factor=15,
        **kwargs)

    # but get ready to write to reduced directory if necessary
    #pout = cmp.pipeline([f_pairs[0]], outdir=outdir, overwrite=True)
    pout = cmp.pipeline(f_pairs, outdir=outdir, overwrite=True)
    pout, _ = prune_pout(pout, f_pairs)
    return pout

def na_back_directory(directory,
                      calibration=True,
                      read_pout=True,
                      write_pout=True,
                      write_plot=True,
                      create_outdir=True,
                      show=False,
                      **kwargs):
    rd = reduced_dir(directory, create=False)
    poutname = os.path.join(rd, 'Na_back.pout')
    pout = cached_pout(na_back_pipeline,
                       poutname=poutname,
                       read_pout=read_pout,
                       write_pout=write_pout,
                       create_outdir=create_outdir,
                       directory=directory,                       
                       **kwargs)
    if len(pout) == 0:
        log.debug(f'no Na background measurements found in {directory}')
        return {}
    _ , pipe_meta = zip(*pout)
    na_back_list = [pm['Na_back'] for pm in pipe_meta]
    df = pd.DataFrame(na_back_list)
    df.sort_values('jd')
    just_date = df['date'].iloc[0]

    instr_mag = u.Magnitude(df['best_back']*u.electron/u.s/u.pix**2)
    df['instr_mag'] = instr_mag

    #tdf = df.loc[df['airmass'] < 2.0]
    tdf = df.loc[df['airmass'] < 2.5]
    mean_back = np.mean(tdf['best_back'])
    std_back = np.std(tdf['best_back'])
    biweight_back = biweight_location(tdf['best_back'])
    mad_std_back = mad_std(tdf['best_back'])


    # https://stackoverflow.com/questions/20664980/pandas-iterate-over-unique-values-of-a-column-that-is-already-in-sorted-order
    # and
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

    #https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    offset_cycler = cycler(color=['r', 'g', 'b', 'y'])
    plt.rc('axes', prop_cycle=offset_cycler)

    f = plt.figure(figsize=[8.5, 11])
    plt.suptitle(f"Na background {just_date}")
    offset_groups = df.groupby(['raoff', 'decoff']).groups
    ax = plt.subplot(3, 1, 1)
    for offset_idx in offset_groups:
        gidx = offset_groups[offset_idx]
        gdf = df.iloc[gidx]
        plot_dates = julian2num(gdf['jd'])
        plt.plot_date(plot_dates, gdf['best_back'],
                      label=f"dRA {gdf.iloc[0]['raoff']} "
                      f"dDEC {gdf.iloc[0]['decoff']} armin")
        plt.axhline(y=biweight_back, color='red')
        plt.axhline(y=biweight_back+mad_std_back,
                    linestyle='--', color='k', linewidth=1)
        plt.axhline(y=biweight_back-mad_std_back,
                    linestyle='--', color='k', linewidth=1)
        plt.text(0.5, biweight_back + 0.1*mad_std_back, 
                 f'{biweight_back:.4f} +/- {mad_std_back:.4f}',
                 ha='center', transform=ax.get_yaxis_transform())
        plt.xlabel('date')
        plt.ylabel('electron/s')
    ax.legend()

    ax = plt.subplot(3, 1, 2)
    for offset_idx in offset_groups:
        gidx = offset_groups[offset_idx]
        gdf = df.iloc[gidx]
        plt.plot(gdf['airmass'], gdf['instr_mag'], '.')
        #plt.axhline(y=biweight_back, color='red')
        #plt.axhline(y=biweight_back+mad_std_back,
        #            linestyle='--', color='k', linewidth=1)
        #plt.axhline(y=biweight_back-mad_std_back,
        #            linestyle='--', color='k', linewidth=1)
        plt.xlabel('Airmass')
        plt.ylabel('mag (electron/s/pix^2')

    ax = plt.subplot(3, 1, 3)
    for offset_idx in offset_groups:
        gidx = offset_groups[offset_idx]
        gdf = df.iloc[gidx]
        plt.plot(gdf['alt'], gdf['best_back'], '.')
        plt.axhline(y=biweight_back, color='red')
        plt.axhline(y=biweight_back+mad_std_back,
                    linestyle='--', color='k', linewidth=1)
        plt.axhline(y=biweight_back-mad_std_back,
                    linestyle='--', color='k', linewidth=1)
        plt.xlabel('Alt')
        plt.ylabel('electron/s')

    f.subplots_adjust(hspace=0.3)
    if write_plot is True:
        write_plot = os.path.join(rd, 'Na_back.png')
    if isinstance(write_plot, str):
        savefig_overwrite(write_plot, transparent=True)
    if show:
        plt.show()
    plt.close()

    # Problem discussed in  https://mail.python.org/pipermail/tkinter-discuss/2019-December/004153.html
    gc.collect()

    return {'date': just_date,
            'jd': np.floor(df['jd'].iloc[0]),
            'biweight_back': biweight_back,
            'mad_std_back': mad_std_back,
            'na_back_list': na_back_list}

def na_back_tree(data_root=RAW_DATA_ROOT,
                 start=None,
                 stop=None,
                 calibration=None,
                 photometry=None,
                 show=False,
                 ccddata_cls=FwRedCorData, # less verbose
                 **kwargs):
    dirs_dates = get_dirs_dates(data_root, start=start, stop=stop)
    dirs, _ = zip(*dirs_dates)
    if len(dirs) == 0:
        log.warning('No directories found')
        return
    if calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = Photometry(precalc=True, **kwargs)

    to_plot_list = []
    na_back_list = []
    for d in dirs:
        nb = na_back_directory(d,
                               calibration=calibration,
                               photometry=photometry,
                               ccddata_cls=ccddata_cls,
                               **kwargs)
        if nb == {}:
            continue
        na_back_list.extend(nb['na_back_list'])
        del nb['na_back_list']
        to_plot_list.append(nb)

    # --> write to_plot_list
    df = pd.DataFrame(to_plot_list)
    plot_dates = julian2num(df['jd'])
    f = plt.figure()#figsize=[8.5, 11])
    plt.suptitle(f"Na background {df['date'].iloc[0]} -- {df['date'].iloc[-1]}")
    plt.plot_date(plot_dates, df['biweight_back'], 'k.')
    plt.plot_date(plot_dates, df['biweight_back']+df['mad_std_back'], 'kv')
    plt.plot_date(plot_dates, df['biweight_back']-df['mad_std_back'], 'k^')
    #plt.errorbar(plot_dates, df['biweight_back'])
    plt.ylim([0, 0.07])
    plt.ylabel('electron/s/pix^2')
    plt.gcf().autofmt_xdate()  # orient date labels at a slant
    if show:
        plt.show()
    return to_plot_list

log.setLevel('DEBUG')

#pout = na_back_test_cache('/data/io/IoIO/raw/20210617')
#pout = na_back_pipeline('/data/io/IoIO/raw/20210617')#, show=True)
#pout = na_back_directory('/data/io/IoIO/raw/20210617')#, show=True)
#pout = na_back_tree(start='2021-06-17', stop='2021-06-18', show=True)
#pout = na_back_tree(start='2021-01-01', show=True)
pout = na_back_tree(show=True)
print(pout)
#print(pair_list)
