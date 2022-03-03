"""Find telluric component for Na image"""

import gc
import os
from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import julian2num

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

from bigmultipipe import no_outfile, cached_pout, prune_pout

from IoIO.utils import (reduced_dir, get_dirs_dates, closest_in_time,
                        valid_long_exposure, im_med_min_max,
                        add_history, cached_csv, iter_polyfit, 
                        simple_show, savefig_overwrite)
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (IoIO_ROOT, RAW_DATA_ROOT,
                               CorMultiPipeBase, 
                               nd_filter_mask,
                               multi_filter_proc, combine_masks)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.standard_star import (extinction_correct,
                                StandardStar, SSArgparseHandler)
from IoIO.photometry import Photometry

NA_BACK_ROOT = os.path.join(IoIO_ROOT, 'Na_back')

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

def sun_angle(ccd,
              bmp_meta=None,
              **kwargs):
    # Put time and sun angle into bmp_meta
    date_obs = ccd.meta.get('DATE-AVG') or ccd.meta.get('DATE-OBS')
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
    bmp_meta['sun_angle'] = sun_angle.value
    bmp_meta['sun_angle_unit'] = sun_angle.unit
    return ccd

def na_back_process(data,
                    in_name=None,
                    bmp_meta=None,
                    calibration=None,
                    photometry=None,
                    n_back_boxes=N_BACK_BOXES,
                    show=False,
                    off_on_ratio=None,
                    **kwargs):
    """post-processing routine that processes a *pair* of ccd images
    in the order on-band, off-band"""
    if bmp_meta is None:
        bmp_meta = {}
    if off_on_ratio is None and calibration is None:
        calibration = Calibration(reduce=True)
    if photometry is None:
        photometry = Photometry(precalc=True,
                                n_back_boxes=n_back_boxes,
                                **kwargs)
    if off_on_ratio is None:
        off_on_ratio, _ = calibration.flat_ratio('Na')
    ccd = data[0]
    date_obs = ccd.meta.get('DATE-AVG') or ccd.meta.get('DATE-OBS')
    tm = Time(date_obs, format='fits')
    just_date, _ = date_obs.split('T')
    objctra = Angle(ccd.meta['OBJCTRA'])
    objctdec = Angle(ccd.meta['OBJCTDEC'])
    objctalt = ccd.meta['OBJCTALT']
    objctaz = ccd.meta['OBJCTAZ']
    raoff0 = ccd.meta.get('RAOFF') or 0
    decoff0 = ccd.meta.get('DECOFF') or 0
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
    best_back_std = np.std(background)

    # Mesosphere is above the stratosphere, where the density of the
    # atmosphere diminishes to very small values.  So all attenuation
    # has already happened by the time we get up to the mesospheric
    # sodium layer
    # https://en.wikipedia.org/wiki/Atmosphere_of_Earth#/media/File:Comparison_US_standard_atmosphere_1962.svg
    airmass = data[0].meta.get('AIRMASS')

    # We are going to turn this into a Pandas dataframe, which does
    # not do well with units, so just return everything
    tmeta = {'best_back': best_back.value,
             'best_back_std': best_back_std.value,
             'back_unit': best_back.unit.to_string(),
             'date': just_date,
             'date_obs': date_obs,
             'plot_date': tm.plot_date,
             'raoff': raoff0,
             'decoff': decoff0,
             'ra': objctra.value,
             'dec': objctdec.value,
             'alt': objctalt,
             'az': objctaz,
             'airmass': airmass}
    # Add sun angle
    _ = sun_angle(data[0], bmp_meta=tmeta, **kwargs)
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

def na_back_pipeline(directory=None, # raw directory
                     glob_include='Jupiter*',
                     calibration=None,
                     photometry=None,
                     n_back_boxes=N_BACK_BOXES,
                     num_processes=None,
                     outdir=None,
                     outdir_root=NA_BACK_ROOT,
                     create_outdir=True,
                     **kwargs):

    outdir = outdir or reduced_dir(directory, outdir_root, create=False)
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
    cmp = CorMultiPipeBase(
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
                      pout=None,
                      read_pout=True,
                      write_pout=True,
                      write_plot=True,
                      outdir=None,
                      create_outdir=True,
                      show=False,
                      **kwargs):
    
    poutname = os.path.join(outdir, 'Na_back.pout')
    pout = pout or cached_pout(na_back_pipeline,
                               poutname=poutname,
                               read_pout=read_pout,
                               write_pout=write_pout,
                               directory=directory,
                               outdir=outdir,
                               create_outdir=create_outdir,
                               **kwargs)
    if len(pout) == 0:
        #log.debug(f'no Na background measurements found in {directory}')
        return {}

    _ , pipe_meta = zip(*pout)
    na_back_list = [pm['Na_back'] for pm in pipe_meta]
    df = pd.DataFrame(na_back_list)
    just_date = df['date'].iloc[0]

    bunit = u.Unit(df['back_unit'].iloc[0])
    instr_mag = u.Magnitude(df['best_back']*bunit/u.pix**2)
    sun_angle = df['sun_angle']
    
    #df.sort_values('sun_angle')
    #plt.errorbar(df['sun_angle'], df['best_back'],
    #             yerr=df['best_back_std'], fmt='k.')
    #plt.show()

    return na_back_list

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
                 read_csvs=True,
                 write_csvs=True,
                 create_outdir=True,                       
                 show=False,
                 ccddata_cls=CorDataBase,
                 outdir_root=NA_BACK_ROOT,                 
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

    #to_plot_list = []
    na_back_list = []
    for directory in dirs:
        rd = reduced_dir(directory, outdir_root, create=False)
        nb = na_back_directory(directory,
                               outdir=rd,
                               create_outdir=create_outdir,
                               calibration=calibration,
                               photometry=photometry,
                               ccddata_cls=ccddata_cls,
                               **kwargs)
        if nb == {}:
            continue
        #na_back_list.extend(nb['na_back_list'])
        na_back_list.extend(nb)
        #del nb['na_back_list']
        #to_plot_list.append(nb)

    # --> Change me!
    return na_back_list


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


#on_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_on.fts'
#off_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_off.fts'
#on = CorDataBase.read(on_fname)
#off = CorDataBase.read(off_fname)
#bmp_meta = {}
#ccd = na_back_process([on, off], in_name=[on_fname, off_fname],
#                      bmp_meta=bmp_meta)#, show=True)
#pout = na_back_pipeline('/data/IoIO/raw/20210617')
#pout = na_back_directory('/data/IoIO/raw/20210617', outdir='/tmp', read_pout=True, fits_fixed_ignore=True)#, show=True)
na_back_list = na_back_tree(fits_fixed_ignore=True)#, read_pout=False)

df = pd.DataFrame(na_back_list)
df.sort_values('sun_angle')

angle_unit = df['sun_angle_unit'].iloc[0]
back_unit = df['back_unit'].iloc[0]
instr_mag = u.Magnitude(df['best_back']*u.Unit(back_unit))
instr_mag_err = u.Magnitude(df['best_back_std']*u.Unit(back_unit))

ss = StandardStar()
ext_coef, ext_coef_err = ss.extinction_coef('Na_on')
airmass = df['airmass'].to_numpy()
#ext_corrected_mag = instr_mag - (ext_coef.value*airmass)*ext_coef.unit
ext_corrected_mag = ss.extinction_correct(instr_mag, airmass, 'Na_on')
airmass_poly = iter_polyfit(df['airmass'],
                            ext_corrected_mag, deg=1, max_resid=1)
na_column_poly = airmass_poly.deriv()
na_column = na_column_poly(0)*ext_corrected_mag.unit
#volumetric_mag = extinction_correct(ext_corrected_mag, df['airmass'], na_column)
volumetric_mag = extinction_correct(ext_corrected_mag.value,
                                    df['airmass'], na_column.value)

sun_angle_poly = iter_polyfit(np.sin(np.radians(df['sun_angle'])),
                              volumetric_mag, deg=1, max_resid=1)
sun_stimulation_poly = sun_angle_poly.deriv()
sun_stimulation = sun_stimulation_poly(0)
sun_stim_mag =  extinction_correct(volumetric_mag,
                                   np.sin(np.radians(df['sun_angle'])),
                                   sun_stimulation)


chemilum_delay = 90 # deg
chemilum_rad = np.radians(df['sun_angle'] - chemilum_delay)
chemilum_poly = iter_polyfit(np.sin(chemilum_rad),
                             sun_stim_mag, deg=1, max_resid=2)
chemilum_poly_deriv = chemilum_poly.deriv()
chemilum_deriv = chemilum_poly_deriv(0)
chemilum_mag =  extinction_correct(sun_stim_mag,
                                   np.sin(chemilum_rad),
                                   chemilum_deriv)

## # Shows that we take data on both sides of the sky relative to the
## # sun, but most away from the sun
## ax = plt.subplot()
## plt.plot(df['sun_angle'], df['airmass'], 'k.')
## plt.xlabel(f'sun_angle ({angle_unit})')
## plt.ylabel(f'airmass')
## plt.show()
## 
## ax = plt.subplot()
## plt.errorbar(df['sun_angle'], df['best_back'],
##              yerr=df['best_back_std'], fmt='k.')
## plt.xlabel(f'sun_angle ({angle_unit})')
## plt.ylabel(f'best_back ({back_unit})')
## ax.set_ylim([0, 0.15])
## plt.show()
## 
## # Shows correlation that high airmasses make *more* background.  Must
## # be higher column through emission region
## ax = plt.subplot()
## plt.errorbar(df['airmass'], df['best_back'],
##              yerr=df['best_back_std'], fmt='k.')
## plt.xlabel(f'airmass')
## plt.ylabel(f'best_back ({back_unit})')
## ax.set_ylim([0, 0.15])
## plt.show()
## 
## ax = plt.subplot()
## #plt.errorbar(df['airmass'], instr_mag.value,
## #             yerr=instr_mag_err.value, fmt='k.')
## plt.plot(df['airmass'], instr_mag, 'k.')
## plt.xlabel(f'airmass')
## plt.ylabel(f'best_back ({back_unit})')
## plt.gca().invert_yaxis()
## #ax.set_ylim([0, 0.15])
## plt.show()
## 
## ax = plt.subplot()
## #plt.errorbar(df['airmass'], instr_mag.value,
## #             yerr=instr_mag_err.value, fmt='k.')
## plt.plot(df['airmass'], ext_corrected_mag, 'k.')
## plt.plot(df['airmass'], airmass_poly(df['airmass']), 'r-')
## plt.xlabel(f'airmass')
## plt.ylabel(f'extinction corrected best_back ({ext_corrected_mag.unit})')
## plt.gca().invert_yaxis()
## #ax.set_ylim([0, 0.15])
## plt.show()
## 
## ax = plt.subplot()
## #plt.errorbar(df['airmass'], instr_mag.value,
## #             yerr=instr_mag_err.value, fmt='k.')
## plt.plot(df['airmass'], volumetric_mag, 'k.')
## plt.xlabel(f'airmass')
## plt.ylabel(f'volumetric Na emission')
## plt.gca().invert_yaxis()
## #ax.set_ylim([0, 0.15])
## plt.show()

ax = plt.subplot()
#plt.errorbar(df['airmass'], instr_mag.value,
#             yerr=instr_mag_err.value, fmt='k.')
plt.plot(df['sun_angle'], volumetric_mag, 'k.')
plt.plot(df['sun_angle'],
         sun_angle_poly(np.sin(np.radians(df['sun_angle']))), 'r.')
plt.xlabel(f'sun angle')
plt.ylabel(f'volumetric Na emission')
plt.gca().invert_yaxis()
#ax.set_ylim([0, 0.15])
plt.show()

#ax = plt.subplot()
##plt.errorbar(df['airmass'], instr_mag.value,
##             yerr=instr_mag_err.value, fmt='k.')
#plt.plot(np.sin(np.radians(df['sun_angle'])), sun_stim_mag, 'k.')
#plt.xlabel(f'sin(sun_angle)')
#plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
#plt.gca().invert_yaxis()
##ax.set_ylim([0, 0.15])
#plt.show()

ax = plt.subplot()
#plt.errorbar(df['airmass'], instr_mag.value,
#             yerr=instr_mag_err.value, fmt='k.')
plt.plot(df['sun_angle'], sun_stim_mag, 'k.')
plt.plot(df['sun_angle'],
         chemilum_poly(np.sin(chemilum_rad)), 'r.')
plt.xlabel(f'sun_angle')
plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
plt.gca().invert_yaxis()
#ax.set_ylim([0, 0.15])
plt.show()

ax = plt.subplot()
#plt.errorbar(df['airmass'], instr_mag.value,
#             yerr=instr_mag_err.value, fmt='k.')
plt.plot(df['sun_angle'], chemilum_mag, 'k.')
plt.xlabel(f'sun_angle')
plt.ylabel(f'volumetric Na emission corrected for sun stimulation and chemiluminescence')
plt.gca().invert_yaxis()
#ax.set_ylim([0, 0.15])
plt.show()

ax = plt.subplot()
#plt.errorbar(df['airmass'], instr_mag.value,
#             yerr=instr_mag_err.value, fmt='k.')
plt.plot(df['airmass'], chemilum_mag, 'k.')
plt.xlabel(f'airmass')
plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
plt.gca().invert_yaxis()
#ax.set_ylim([0, 0.15])
plt.show()

