"""Construct model of telluric Na emission.  See `NaBack.best_back`"""

import gc
import os
from cycler import cycler

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.dates import julian2num

import pandas as pd

from astropy import log
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.nddata import CCDData
from astropy.stats import mad_std, biweight_location
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.modeling import models, fitting

from ccdproc import ImageFileCollection

from bigmultipipe import no_outfile, cached_pout, prune_pout

from precisionguide import pgproperty

from IoIO.ioio_globals import IoIO_ROOT, RAW_DATA_ROOT
from IoIO.utils import (Lockfile, reduced_dir, get_dirs_dates,
                        closest_in_time, valid_long_exposure, im_med_min_max,
                        add_history, cached_csv, iter_polyfit, 
                        savefig_overwrite)
from IoIO.simple_show import simple_show
from IoIO.cordata_base import CorDataBase
from IoIO.cormultipipe import (CorMultiPipeBase, angle_to_major_body,
                               nd_filter_mask, combine_masks)
from IoIO.calibration import Calibration, CalArgparseHandler
from IoIO.standard_star import (extinction_correct,
                                StandardStar, SSArgparseHandler)
# --> Consider making this CorPhotometry and allowing astrometric and
# --> photometric information to be siphoned off
from IoIO.photometry import Photometry

NA_BACK_ROOT = os.path.join(IoIO_ROOT, 'Na_back')
LOCKFILE = '/tmp/na_back_reduce.lock'

# For Photometry -- number of boxes for background calculation
N_BACK_BOXES = 20

def to_numpy(series_or_numpy):
    """Return input as non-pandas object (e.g. numpy.array)"""
    if isinstance(series_or_numpy, pd.Series):
        return series_or_numpy.to_numpy()
    return series_or_numpy    

def sun_angle(ccd,
              bmp_meta=None,
              **kwargs):
    """cormultipipe post-processing routine that inserts angle between
    pointing direction and sun"""
    sa = angle_to_major_body(ccd, 'sun')
    # --> Eventually just have this be a Quantity for a QTable
    bmp_meta['sun_angle'] = sa
    bmp_meta['sun_angle_unit'] = sa.unit.to_string()
    ccd.meta['HIERARCH SUN_ANGLE'] = (sa.value, f'[{sa.unit}]')
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
    # --> If we are going to use this for comets, what we really want
    # --> is just angular distance from Jupiter via the astropy ephemerides
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
        flux = photometry.background / exptime
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
    # --> I am eventually going to return a dictionary that can be
    # transformed into a QTable(rows==list_of_dict)
    dmeta = {'best_back': best_back.value,
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
    #tmeta = {'best_back': best_back,
    #         'best_back_std': best_back_std,
    #         'date_obs': tm,
    #         'raoff': raoff0*u.arcmin,
    #         'decoff': decoff0*u.arcmin,
    #         'ra': objctra,
    #         'dec': objctdec,
    #         'alt': objctalt*u.deg,
    #         'az': objctaz*u.deg,
    #         'airmass': airmass,
    #         'sun_angle': angle_to_major_body(ccd, 'sun')}
    # Add sun angle
    _ = sun_angle(data[0], bmp_meta=dmeta, **kwargs)
    bmp_meta['Na_back'] = dmeta
    #bmp_meta['Na_back_table'] = tmeta

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
    try:
        raoffs = collection.values('raoff', unique=True)
        decoffs = collection.values('decoff', unique=True)
    except Exception as e:
        log.debug(f'Skipping {directory} because of problem with RAOFF/DECOFF: {e}')
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
        
    # --> We are going to want add_ephemeris here with a CorPhotometry
    # --> to build up astormetric solutions
    cmp = CorMultiPipeBase(
        auto=True,
        calibration=calibration,
        photometry=photometry,
        create_outdir=create_outdir,
        post_process_list=[nd_filter_mask,
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

# Keep this stuff around just in case I want to do individual day
# stuff, though I have the dataset as a whole in NaBack to play with

#    #tdf = df.loc[df['airmass'] < 2.0]
#    tdf = df.loc[df['airmass'] < 2.5]
#    mean_back = np.mean(tdf['best_back'])
#    std_back = np.std(tdf['best_back'])
#    biweight_back = biweight_location(tdf['best_back'])
#    mad_std_back = mad_std(tdf['best_back'])
#
#
#    # https://stackoverflow.com/questions/20664980/pandas-iterate-over-unique-values-of-a-column-that-is-already-in-sorted-order
#    # and
#    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
#
#    #https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
#    offset_cycler = cycler(color=['r', 'g', 'b', 'y'])
#    plt.rc('axes', prop_cycle=offset_cycler)
#
#    f = plt.figure(figsize=[8.5, 11])
#    plt.suptitle(f"Na background {just_date}")
#    offset_groups = df.groupby(['raoff', 'decoff']).groups
#    ax = plt.subplot(3, 1, 1)
#    for offset_idx in offset_groups:
#        gidx = offset_groups[offset_idx]
#        gdf = df.iloc[gidx]
#        plot_dates = julian2num(gdf['jd'])
#        plt.plot_date(plot_dates, gdf['best_back'],
#                      label=f"dRA {gdf.iloc[0]['raoff']} "
#                      f"dDEC {gdf.iloc[0]['decoff']} armin")
#        plt.axhline(y=biweight_back, color='red')
#        plt.axhline(y=biweight_back+mad_std_back,
#                    linestyle='--', color='k', linewidth=1)
#        plt.axhline(y=biweight_back-mad_std_back,
#                    linestyle='--', color='k', linewidth=1)
#        plt.text(0.5, biweight_back + 0.1*mad_std_back, 
#                 f'{biweight_back:.4f} +/- {mad_std_back:.4f}',
#                 ha='center', transform=ax.get_yaxis_transform())
#        plt.xlabel('date')
#        plt.ylabel('electron/s')
#    ax.legend()
#
#    ax = plt.subplot(3, 1, 2)
#    for offset_idx in offset_groups:
#        gidx = offset_groups[offset_idx]
#        gdf = df.iloc[gidx]
#        plt.plot(gdf['airmass'], gdf['instr_mag'], '.')
#        #plt.axhline(y=biweight_back, color='red')
#        #plt.axhline(y=biweight_back+mad_std_back,
#        #            linestyle='--', color='k', linewidth=1)
#        #plt.axhline(y=biweight_back-mad_std_back,
#        #            linestyle='--', color='k', linewidth=1)
#        plt.xlabel('Airmass')
#        plt.ylabel('mag (electron/s/pix^2')
#
#    ax = plt.subplot(3, 1, 3)
#    for offset_idx in offset_groups:
#        gidx = offset_groups[offset_idx]
#        gdf = df.iloc[gidx]
#        plt.plot(gdf['alt'], gdf['best_back'], '.')
#        plt.axhline(y=biweight_back, color='red')
#        plt.axhline(y=biweight_back+mad_std_back,
#                    linestyle='--', color='k', linewidth=1)
#        plt.axhline(y=biweight_back-mad_std_back,
#                    linestyle='--', color='k', linewidth=1)
#        plt.xlabel('Alt')
#        plt.ylabel('electron/s')
#
#    f.subplots_adjust(hspace=0.3)
#    if write_plot is True:
#        write_plot = os.path.join(rd, 'Na_back.png')
#    if isinstance(write_plot, str):
#        savefig_overwrite(write_plot, transparent=True)
#    if show:
#        plt.show()
#    plt.close()
#
#    # Problem discussed in  https://mail.python.org/pipermail/tkinter-discuss/2019-December/004153.html
#    gc.collect()
#
#    return {'date': just_date,
#            'jd': np.floor(df['jd'].iloc[0]),
#            'biweight_back': biweight_back,
#            'mad_std_back': mad_std_back,
#            'na_back_list': na_back_list}

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

class NaBack():
    def __init__(self,
                 reduce=False,
                 raw_data_root=RAW_DATA_ROOT,
                 start=None,
                 stop=None,
                 calibration=None,
                 photometry=None,
                 standard_star_obj=None,
                 read_csvs=True,
                 write_csvs=True,
                 read_pout=True,
                 write_pout=True,
                 create_outdir=True,                       
                 show=False,
                 ccddata_cls=CorDataBase,
                 outdir_root=NA_BACK_ROOT,
                 chemilum_delay=150, # minimized biweight of daily mad_stds
                 write_summary_plots=False,
                 lockfile=LOCKFILE,
                 **kwargs):
        self.raw_data_root = raw_data_root
        self.start = start
        self.stop = stop
        self.calibration = calibration
        self.photometry = photometry
        self.standard_star_obj = standard_star_obj
        self.read_csvs = read_csvs
        self.write_csvs = write_csvs
        self.read_pout = read_pout
        self.write_pout = write_pout
        self.create_outdir = create_outdir
        self.show = show
        self.ccddata_cls = ccddata_cls
        self.outdir_root = outdir_root
        self.chemilum_delay = chemilum_delay
        self.write_summary_plots = write_summary_plots
        self._lockfile = lockfile
        self._kwargs = kwargs
        if reduce:
            # --> This may need improvement
            self.reduction_products

    @pgproperty
    def calibration(self):
        # If the user has opinions about the time range over which
        # calibration should be done, they should be expressed by
        # creating the calibration object externally and passing it in
        # at instantiation time
        return Calibration(reduce=True)

    @pgproperty
    def standard_star_obj(self):
        return StandardStar(calibration=self.calibration, reduce=True)
    
    @pgproperty
    def chemilum_delay(self):
        pass
    
    @chemilum_delay.setter
    def chemilum_delay(self, value):
        """Assumes non-`~astropy.units.Quantity` reads in degrees"""
        if not isinstance(value, u.Quantity):
            value *= u.deg
        return value

    @pgproperty
    def reduction_products(self):
        lock = Lockfile(self._lockfile)
        lock.create()

        rp = na_back_tree(raw_data_root=self.raw_data_root,
                          start=self.start,
                          stop=self.stop,
                          calibration=self.calibration,
                          photometry=self.photometry,
                          read_csvs=self.read_csvs,
                          write_csvs=self.write_csvs,
                          read_pout=self.read_pout,
                          write_pout=self.write_pout,
                          create_outdir=self.create_outdir,
                          show=self.show,
                          ccddata_cls=self.ccddata_cls,
                          outdir_root=self.outdir_root,
                          **self._kwargs)
        lock.clear()
        return rp

    @pgproperty
    def df(self):
        """Returns Pandas dataframe of all Na background fields"""
        return pd.DataFrame(self.reduction_products)

    @pgproperty
    def angle_unit(self):
        ustr = self.df['sun_angle_unit'].iloc[0]
        return u.Unit(ustr)

    @pgproperty
    def back_unit(self):
        ustr = self.df['back_unit'].iloc[0]
        return u.Unit(ustr)

    def instr_mag(self, best_back):
        """Returns best_back as `~astropy.units.Magnitude`. Uses
        `NaBack.back_unit` to convert non-`~astropy.Quantity` inputs
        to `~astropy.Quantity`

        Parameters
        ----------
        best_back : float, numpy.array, astropy.Quantity, or Pandas.DataFrame

        """
        if isinstance(best_back, u.Quantity):
            return u.Magnitude(best_back)
        # This converts best_back into a proper numpy array with astropy unit 
        return u.Magnitude(to_numpy(best_back)*self.back_unit)

    @pgproperty
    def instr_mag_unit(self):
        q = self.instr_mag(1)
        return q.unit

    @pgproperty
    def ext_coef(self):
        ext_coef, self.ext_coef_err = \
            self.standard_star_obj.extinction_coef('Na_on')
        return ext_coef

    @pgproperty
    def ext_coef_err(self):
        self.ext_coef

    def angle_quantity(self, angle):
        if isinstance(angle, u.Quantity):
            return angle
        return to_numpy(angle)*self.angle_unit
        
    def angle_sin(self, angle):
        angle_sin = np.sin(self.angle_quantity(angle).to(u.rad))
        return angle_sin.value

    def instr_mag_to_meso(self, instr_mag, airmass, inverse=False):
        """Convert detected instr_mag to base of mesosphere emission"""
        # --> This might eventually get a date on it
        instr_mag = to_numpy(instr_mag)
        airmass =  to_numpy(airmass)
        return extinction_correct(instr_mag,
                                  airmass=airmass,
                                  ext_coef=self.ext_coef,
                                  inverse=inverse)

    @pgproperty
    def all_instr_mag(self):
        return self.instr_mag(self.df['best_back'])

    @pgproperty
    def all_meso_mag(self):
        return self.instr_mag_to_meso(self.all_instr_mag,
                                      self.df['airmass'])

    @pgproperty
    def all_sun_angle(self):
        return self.angle_quantity(self.df['sun_angle'])

    @pgproperty
    def all_sun_angle_sin(self):
        return self.angle_sin(self.all_sun_angle)

    @pgproperty
    def meso_airmass_poly(self):
        """Returns order 1 polynomial fit to observed mesophereic emission in
        magnitudes vs tropospheric airmass.  All datapoints are used.
        This could be refinded geometrically to get a better path
        length through the mesophere

        """
        return iter_polyfit(self.df['airmass'], self.all_meso_mag,
                            deg=1, max_resid=1)

    @pgproperty
    def meso_vol_per_airmass(self):
        """Uses the dataset as a whole to calculate mesopheric volumetric
        density per (tropospheric) airmass.  Used in meso_vol

        """
        meso_vol_poly = self.meso_airmass_poly.deriv()
        return meso_vol_poly(0)*self.instr_mag_unit

    def meso_mag_to_vol(self, meso_mag, airmass, inverse=False):
        """Returns predicted mesospheric volumetric density in magnitudes

        This could be improved geometrically so pathlength in
        mesosphere is used rather than airmass in troposphere.  In
        principle, this function has a seasonal dependence as the
        parameters of the sodium layer, like thickness and centroid,
        change.  See Dunker et al 2015

        Parameters
        ----------
        meso_mag : Quantity
            Mesospheric emission in magnitudes (observed instr_mag
            corrected for tropospheric airmass)

        airmass : float-like
            Airmass (ideally in troposphere)

        inverse : bool
            Invert the model

        """
        meso_mag = to_numpy(meso_mag)
        airmass = to_numpy(airmass)
        return extinction_correct(meso_mag,
                                  airmass=airmass,
                                  ext_coef=self.meso_vol_per_airmass,
                                  inverse=inverse)
        
    @pgproperty
    def all_meso_vol(self):
        return self.meso_mag_to_vol(self.all_meso_mag, self.df['airmass'])

    @pgproperty
    def meso_sun_angle_poly(self):
        """Returns order 1 polynomial fit to mesospheric volumetric density in
        magnitudes vs sin(sun angle).  All datapoints are used.

        """
        #return iter_polyfit(self.all_sun_angle_sin, self.all_meso_vol,
        #                    deg=1, max_resid=1)
        return iter_polyfit(self.all_sun_angle_sin,
                            self.all_meso_vol,
                            deg=1, max_resid=1)

    @pgproperty
    def sun_stim_per_sin_sun_angle(self):
        """Uses the dataset as a whole to calculate direct photon stimulation
        and re-emission by mesospheric sodium (in magnitude) per sin
        sun angle.  See meso_sun_angle_poly

        """
        sun_stimulation_poly = self.meso_sun_angle_poly.deriv()
        return sun_stimulation_poly(0)*self.all_instr_mag.unit

    def meso_vol_sun_stim_correct(self, meso_vol, sun_angle, inverse=False):
        """Returns predicted mesospheric volumetric density in
        magnitudes corrected for solar illumination"""
        return extinction_correct(meso_vol,
                                  airmass=self.angle_sin(sun_angle),
                                  ext_coef=self.sun_stim_per_sin_sun_angle,
                                  inverse=inverse)        

    @pgproperty
    def all_sun_stim_corrected(self):
        return self.meso_vol_sun_stim_correct(self.all_meso_vol,
                                              self.all_sun_angle)

    def chemilum_sin(self, sun_angle):
        sun_angle = self.angle_quantity(sun_angle)
        return self.angle_sin(sun_angle - self.chemilum_delay)

    @pgproperty
    def chemilum_poly(self):
        """Returns order 1 polynomial fit to sun-stim-corrected mesospheric
        volumetric density in magnitudes vs sin(sun angle).  All
        datapoints are used.

        """
        return iter_polyfit(self.chemilum_sin(self.all_sun_angle),
                            self.all_sun_stim_corrected,
                            deg=1, max_resid=1)

    @pgproperty
    def chemilum_per_sin_sun_angle(self):
        """Uses the dataset as a whole to calculate sun-stim and
        chemiluminescent-corrected (in magnitude) per sin sun angle.
        See chemilum_poly

        """
        chemilum_per_sin_sun_angle_poly = self.chemilum_poly.deriv()
        return chemilum_per_sin_sun_angle_poly(0)*self.all_instr_mag.unit

    def meso_vol_sun_stim_chemilum_correct(self,
                                           sun_stim_corrected,
                                           sun_angle,
                                           inverse=False):
        """Returns predicted mesospheric volumetric density in magnitudes
        corrected for solar illumination and chemiluminescence

        """
        return extinction_correct(sun_stim_corrected,
                                  airmass=self.chemilum_sin(sun_angle),
                                  ext_coef=self.chemilum_per_sin_sun_angle,
                                  inverse=inverse)

    @pgproperty
    def all_meso_vol_sun_stim_chemilum_corrected(self):
        return self.meso_vol_sun_stim_chemilum_correct(
            self.all_sun_stim_corrected,
            self.all_sun_angle)

    @pgproperty
    # NO NOT USE THIS ONE
    def meso_vol_corrected_sin_mag(self):
        # This doesn't do any fit.  I have just tweaked the parameters
        # by hand
        fit = fitting.LevMarLSQFitter()
        sin_init = (models.Sine1D(amplitude=1,
                                  frequency=1/u.year,
                                  phase=(94*u.deg).to(u.rad))
                    + models.Const1D(amplitude=6))
        # So this fit is actually bogus and the units are not really
        # right.  They should be .value and then magically the
        # evaluated function should be converted back into a Magnitude
        return fit(sin_init,
                   to_numpy(self.df['plot_date'])*u.day,
                   self.all_meso_vol_sun_stim_chemilum_corrected.physical)

    @pgproperty
    def meso_vol_corrected_sin_physical(self):
        # nominally better-looking fits were achieved with the mag. version
        # This doesn't do any fit.  I have just tweaked the parameters
        # by hand
        # --! Aim for the low side to see if that helps make sure
        # background is not over-estimated.  Const1D=0.004 might be better
        fit = fitting.LevMarLSQFitter()
        sin_init = (models.Sine1D(amplitude=0.0030,
                                  frequency=1/u.year,
                                  phase=(181*u.deg).to(u.rad))
                    + models.Const1D(amplitude=0.0035))
        return fit(sin_init,
                   to_numpy(self.df['plot_date'])*u.day,
                   self.all_meso_vol_sun_stim_chemilum_corrected.physical)

    def back_rate_to_meso_vol_corrected(self, back_rate, airmass, sun_angle):
        """Returns detector count rate after correction for modeled
        mesopheric effects.  After such corretions, time variation is
        the only major systematic effect"""
        instr_mag = self.instr_mag(back_rate)
        meso_mag = self.instr_mag_to_meso(instr_mag, airmass)
        meso_vol = self.meso_mag_to_vol(meso_mag, airmass)
        meso_vol_sun_stim_corrected = \
            self.meso_vol_sun_stim_correct(meso_vol, sun_angle)
        meso_vol_sun_stim_chemilum_corrected = \
            self.meso_vol_sun_stim_chemilum_correct(
                meso_vol_sun_stim_corrected, sun_angle)
        return meso_vol_sun_stim_chemilum_corrected
            
    def corrected_meso_vol_to_back_rate(self, corrected, airmass, sun_angle):
        """Returns observed instr_mag given a volumetric mag corrected for all
        modeled effects except time variation"""
        sun_stim = self.meso_vol_sun_stim_chemilum_correct(
            corrected, sun_angle, inverse=True)
        meso_vol = self.meso_vol_sun_stim_correct(
            sun_stim, sun_angle, inverse=True)
        meso_mag = self.meso_mag_to_vol(meso_vol, airmass, inverse=True)
        instr_mag = self.instr_mag_to_meso(meso_mag, airmass, inverse=True)
        return instr_mag.physical 

    @pgproperty
    def meso_vol_corrected_table(self):
        self.df['iplot_date'] = to_numpy(self.df['plot_date']).astype(int)
        uidays = list(set(self.df['iplot_date']))
        iday_plot_dates = np.full(len(uidays), np.nan)
        vol_corrected_biweight = np.full(len(uidays), np.nan)*self.back_unit
        vol_corrected_mad_std = vol_corrected_biweight.copy()
        #self.df['vol_corrected_biweight'] = np.nan
        #self.df['vol_corrected_mad_std'] = np.nan
        for i, iday in enumerate(uidays):
            idx = np.flatnonzero(self.df['iplot_date'] == iday)
            tcorrected_vol = self.all_meso_vol_sun_stim_chemilum_corrected[idx]
            tcorrected_vol = tcorrected_vol.physical
            tbiweight = biweight_location(tcorrected_vol)
            tstd = mad_std(tcorrected_vol)
            iday_plot_dates[i] = iday
            vol_corrected_biweight[i] = tbiweight
            vol_corrected_mad_std[i] = tstd
            #self.df.iloc[idx, 'vol_corrected_biweight'] = tbiweight
            #self.df.iloc[idx, 'vol_corrected_mad_std'] = tstd
        qt = QTable([iday_plot_dates,
                     vol_corrected_biweight,
                     vol_corrected_mad_std],
                    names = ('iplot_date',
                             'vol_corrected_biweight',
                             'vol_corrected_mad_std'))
        return qt.group_by('iplot_date')

    @pgproperty
    def meso_vol_corrected_err(self):
        """Biweight location of measured meso_vol_corrected mad_stds

        """
        return biweight_location(
            self.meso_vol_corrected_table['vol_corrected_mad_std'],
            ignore_nan=True)

    def best_back(self, date_obs, airmass, sun_angle):
        """Returns the best-estimate Na background for a given observation

        Uses all Na background meausrements (--> need to add
        long-duration comet exposures as well) to construct an
        empirical, time-dependent model of the volumetric emission of
        Na in the mesosphere.  The effects of resonant scattering and
        chemiluminescence have been considered, as well as a simple
        sin-wave seasonal dependence.  This routine first looks to see
        if background observations were recorded on the date of
        observation and uses the biweight distribution of those 

        Parameters
        ----------
        date_obs : str or `~astropy.time.Time`
            Time of observation (DATE-AVG prefered in cor_processed data)

        airmass : float
            Airmass of observation look direction

        sun angle : float or Quantity
            Angle between observation look direction and sun.  If
            float, assumed to be in degrees

        Returns
        -------
        best_back, best_back_err : tuple of Quantity
            best-estimate Na background rate in electron/s

        """
        if isinstance(date_obs, Time):
            tm = date_obs
        else:
            tm = Time(date_obs, format='fits')
        iplot_date = int(tm.plot_date)
        t = self.meso_vol_corrected_table
        mask = t.groups.keys['iplot_date'] == iplot_date
        meso_vol_corrected = t['vol_corrected_biweight'][mask]
        meso_vol_corrected_err = t['vol_corrected_mad_std'][mask]
        if (len(meso_vol_corrected) == 0
            or not np.isfinite(meso_vol_corrected)):
            # --> Not sure where the missing entries are coming from
            # --> Might want to separate this so I can capture the
            # method used to get the rate
            meso_vol_corrected = \
                self.meso_vol_corrected_sin_physical(iplot_date*u.day)
            meso_vol_corrected_err = self.meso_vol_corrected_err
        else:
            meso_vol_corrected = meso_vol_corrected[0]
            meso_vol_corrected_err = meso_vol_corrected_err[0]            

        #print(f'meso_vol_corrected {meso_vol_corrected}')
        bb = self.corrected_meso_vol_to_back_rate(
            u.Magnitude(meso_vol_corrected), airmass, sun_angle)
        return bb, meso_vol_corrected_err

    def plots(self):
        f = plt.figure(figsize=[11, 8.5])
        ax = plt.subplot(3, 3, 1)
        plt.plot(self.df['sun_angle'], self.df['airmass'], 'k.')
        plt.xlabel(f'sun_angle ({self.angle_unit})')
        plt.ylabel(f'airmass')

        ax = plt.subplot(3, 3, 2)
        plt.errorbar(self.df['sun_angle'], self.df['best_back'],
                     yerr=self.df['best_back_std'], fmt='k.')
        plt.xlabel(f'sun_angle ({self.angle_unit})')
        plt.ylabel(f'best_back ({self.back_unit})')
        ax.set_ylim([0, 0.15])
        
        # Shows correlation that high airmasses make *more* background.  Must
        # be higher column through emission region
        ax = plt.subplot(3, 3, 3)
        plt.errorbar(self.df['airmass'], self.df['best_back'],
                     yerr=self.df['best_back_std'], fmt='k.')
        plt.xlabel(f'airmass')
        plt.ylabel(f'best_back ({self.back_unit})')
        ax.set_ylim([0, 0.15])
        
        ax = plt.subplot(3, 3, 4)
        #plt.errorbar(df['airmass'], instr_mag.value,
        #             yerr=instr_mag_err.value, fmt='k.')
        plt.plot(self.df['airmass'], self.all_meso_mag, 'k.')
        plt.plot(self.df['airmass'],
                 self.meso_airmass_poly(self.df['airmass']), 'r.')
        plt.xlabel(f'airmass')
        plt.ylabel(f'extinction corrected back ({self.back_unit})')
        plt.gca().invert_yaxis()
        
        #ax = plt.subplot(3, 3, 5)
        ##plt.errorbar(df['airmass'], instr_mag.value,
        ##             yerr=instr_mag_err.value, fmt='k.')
        #plt.plot(self.df['airmass'], self.all_meso_vol, 'k.')
        #plt.xlabel(f'airmass')
        #plt.ylabel(f'volumetric Na emission')
        #plt.gca().invert_yaxis()
        ##ax.set_ylim([0, 0.15])

        ax = plt.subplot(3, 3, 5)
        #plt.errorbar(df['airmass'], instr_mag.value,
        #             yerr=instr_mag_err.value, fmt='k.')
        plt.plot(self.df['sun_angle'], self.all_meso_vol, 'k.')
        plt.plot(self.df['sun_angle'],
                 self.meso_sun_angle_poly(self.all_sun_angle_sin), 'r.')
        plt.xlabel(f'sun_angle ({self.angle_unit})')
        plt.ylabel(f'volumetric Na emission')
        plt.gca().invert_yaxis()
        #ax.set_ylim([0, 0.15])

        ax = plt.subplot(3, 3, 6)
        #plt.errorbar(df['airmass'], instr_mag.value,
        #             yerr=instr_mag_err.value, fmt='k.')
        plt.plot(self.all_sun_angle, self.all_sun_stim_corrected, 'k.')
        plt.plot(self.df['sun_angle'],
                 self.chemilum_poly(self.all_sun_angle_sin), 'r.')
        plt.xlabel(f'sun_angle ({self.angle_unit})')
        plt.ylabel(f'sun stim-corrected volumetric Na emission')
        plt.gca().invert_yaxis()
        #ax.set_ylim([0, 0.15])

        ax = plt.subplot(3, 3, 7)
        plt.plot_date(self.df['plot_date'],
                      self.all_meso_vol_sun_stim_chemilum_corrected.physical,
                      'k.')
        days = np.arange(np.min(self.df['plot_date']),
                         np.max(self.df['plot_date']))
        plt.xlabel(f'Date')
        plt.ylabel(f'best_back ({self.back_unit})')
        plt.plot_date(days,
                      self.meso_vol_corrected_sin_physical(days*u.day), 'r.')
        ax.set_ylim([0, 0.02])
        ax.tick_params(axis='x', labelrotation = 45)
        
        ax = plt.subplot(3, 3, 8)
        plt.plot_date(self.df['plot_date'],
                      self.all_meso_vol_sun_stim_chemilum_corrected,
                      'k.')
        days = np.arange(np.min(self.df['plot_date']),
                         np.max(self.df['plot_date']))
        plt.xlabel(f'Date')
        plt.ylabel(f'best_back ({self.back_unit})')
        plt.plot_date(days,
                      self.meso_vol_corrected_sin_mag(days*u.day), 'r.')
        plt.gca().invert_yaxis()
        #ax.set_ylim([0, 0.02])
        ax.tick_params(axis='x', labelrotation = 45)
        
        #ax = plt.subplot(3, 3, 8)
        ##plt.errorbar(df['airmass'], instr_mag.value,
        ##             yerr=instr_mag_err.value, fmt='k.')
        #plt.plot(self.all_sun_angle,
        #         self.all_meso_vol_sun_stim_chemilum_corrected, 'k.')
        #plt.xlabel(f'sun_angle ({self.angle_unit})')
        #plt.ylabel(f'sun stim, chemilum-corrected volumetric Na emission')
        #plt.gca().invert_yaxis()
        ##ax.set_ylim([0, 0.15])
        #
        #ax = plt.subplot(3, 3, 9)
        ##plt.errorbar(df['airmass'], instr_mag.value,
        ##             yerr=instr_mag_err.value, fmt='k.')
        #plt.plot(self.df['airmass'],
        #         self.all_meso_vol_sun_stim_chemilum_corrected, 'k.')
        #plt.xlabel(f'airmass')
        #plt.ylabel(f'sun stim, chemilum-corrected volumetric Na emission')
        #plt.gca().invert_yaxis()
        ##ax.set_ylim([0, 0.15])

        # In vol units
        ax = plt.subplot(3, 3, 9)
        plt.plot_date(self.df['plot_date'],
                      self.all_meso_vol_sun_stim_chemilum_corrected.physical,
                      'k.')
        t = self.meso_vol_corrected_table
        plt.plot_date(t['iplot_date']+0.5,
                      t['vol_corrected_biweight'], 'r.')
        plt.errorbar(t['iplot_date']+0.5,
                     t['vol_corrected_biweight'].value,
                     yerr=t['vol_corrected_mad_std'].value,
                     fmt='r.')
        plt.xlabel(f'Date')
        plt.ylabel(f'best_back daily ({self.back_unit})')
        ax.set_ylim([0, 0.02])
        ax.tick_params(axis='x', labelrotation = 45)
        plt.tight_layout()
        plt.show()

        #t = self.meso_vol_corrected_table
        #obs_rate = self.corrected_meso_vol_to_back_rate(
        #    self.all_meso_vol_sun_stim_chemilum_corrected.physical)
        #day_rate = self.corrected_meso_vol_to_back_rate(
        #    t['vol_corrected_biweight'])
        #day_err =  self.corrected_meso_vol_to_back_rate(
        #    t['vol_corrected_mad_std'])
        #ax = plt.subplot(3, 3, 9)
        #plt.plot_date(self.df['plot_date'], rate, 'k.')
        #plt.plot_date(t['iplot_date']+0.5,
        #              day_rate, 'r.')
        #plt.errorbar(t['iplot_date']+0.5,
        #             day_rate.value,
        #             yerr=day_err.value,
        #             fmt='r.')
        #plt.xlabel(f'Date')
        #plt.ylabel(f'best_back daily ({self.back_unit})')
        #ax.set_ylim([0, 0.02])
        #ax.tick_params(axis='x', labelrotation = 45)
        #plt.tight_layout()
        #plt.show()

   #def vs_time(self):
   #    fit = fitting.LevMarLSQFitter()
   #    sin_init = (models.Sine1D(amplitude=0.0025,
   #                              frequency=1/u.year,
   #                              phase=(182*u.deg).to(u.rad))
   #                + models.Const1D(amplitude=0.003))
   #    fitted_sin = fit(sin_init,
   #                     to_numpy(self.df['plot_date'])*u.day,
   #                     self.all_meso_vol_sun_stim_chemilum_corrected.physical)
   #    print(sin_init)
   #    print(fitted_sin)
   #    print('did')
   #    #return fitted_sin
   #    f = plt.figure(figsize=[8.5, 11])
   #    ax = plt.subplot(2, 1, 1)
   #    days = np.arange(np.min(self.df['plot_date']),
   #                     np.max(self.df['plot_date']))
   #    plt.plot_date(self.df['plot_date'],
   #                  self.all_meso_vol_sun_stim_chemilum_corrected.physical,
   #                  'k.')
   #    plt.plot_date(days,
   #                  #fitted_sin(days*u.day),
   #                  sin_init(to_numpy(self.df['plot_date'])*u.day),#*self.back_unit,
   #                  'r.')
   #    plt.ylabel(f'corrected volumetric Na emission ({self.back_unit})')
   #    ax.set_ylim([0, 0.02])
   #    plt.gcf().autofmt_xdate()
   #
   #    ax = plt.subplot(2, 1, 2)
   #    plt.plot_date(self.df['plot_date'],
   #                  self.all_meso_vol_sun_stim_chemilum_corrected,
   #                  'k.')
   #    plt.ylabel(f'corrected volumetric Na emission ({self.back_unit})')
   #    plt.gca().invert_yaxis()
   #    plt.gcf().autofmt_xdate()
   #
   #
   #    plt.tight_layout()
   #    plt.show()

    
def na_meso_sub(ccd_in, bmp_meta=None, na_meso_obj=None, **kwargs):
    """cormultipipe post-processing routine to subtract mesospheric
    emission

    Parameters
    ----------
    na_meso_obj : NaBack object
    """
    assert ccd_in.meta['FILTER'] == 'Na_on'
    bmp_meta = {}
    ccd = ccd_in.copy()
    meso, meso_err = na_meso_obj.best_back(ccd.meta['DATE-AVG'],
                                           ccd.meta['AIRMASS'],
                                           ccd.meta['SUN_ANGLE'])
    # --> may want to mask the ND filter and put it back in again,
    # --> since this over-subtracts
    ccd = ccd.subtract(meso, handle_meta='first_found')

    ccd.meta['MESO'] = (
        meso.value,
        f'model mesospheric Na emission ({meso.unit})')
    ccd.meta['MESO_ERR'] = (
        meso_err.value,
        f'model mesospheric error ({meso_err.unit})')
    
    bmp_meta['meso'] = meso
    bmp_meta['meso_err'] = meso_err
    add_history(ccd.meta, 'Subtracted MESO')
    return ccd

#log.setLevel('DEBUG')

## We can close the loop!
#nab = NaBack()
#observed = nab.corrected_meso_vol_to_back_rate(
#    nab.all_meso_vol_sun_stim_chemilum_corrected,
#    nab.df['airmass'],
#    nab.df['sun_angle'])
#
#corrected = nab.back_rate_to_meso_vol_corrected(
#    nab.df['best_back'], nab.df['airmass'], nab.df['sun_angle'])
#
#print(corrected - nab.all_meso_vol_sun_stim_chemilum_corrected)
#
#print(observed - to_numpy(nab.df['best_back'])*nab.back_unit)
#
#print(observed / to_numpy(nab.df['best_back'])*nab.back_unit)

#print(nab.best_back('2020-01-01', 3, 90))
#print(nab.best_back('2020-06-16', 3, 90))
#nab.plots()


#### ######
#### # TIME DEPENDENT STUFF
#### #######
#### nab.df['iplot_date'] = to_numpy(nab.df['plot_date']).astype(int)
#### uidays = list(set(nab.df['iplot_date']))
#### iday_plot_dates = np.full(len(uidays), np.nan)
#### vol_corrected_biweight = np.full(len(uidays), np.nan)*nab.back_unit
#### vol_corrected_mad_std = vol_corrected_biweight.copy()
#### #nab.df['vol_corrected_biweight'] = np.nan
#### #nab.df['vol_corrected_mad_std'] = np.nan
#### for i, iday in enumerate(uidays):
####     idx = np.flatnonzero(nab.df['iplot_date'] == iday)
####     tcorrected_vol = nab.all_meso_vol_sun_stim_chemilum_corrected[idx]
####     tcorrected_vol = tcorrected_vol.physical
####     tbiweight = biweight_location(tcorrected_vol)
####     tstd = mad_std(tcorrected_vol)
####     iday_plot_dates[i] = iday
####     vol_corrected_biweight[i] = tbiweight
####     vol_corrected_mad_std[i] = tstd
####     #nab.df.iloc[idx, 'vol_corrected_biweight'] = tbiweight
####     #nab.df.iloc[idx, 'vol_corrected_mad_std'] = tstd
#### 
#### ax = plt.subplot()
#### plt.plot_date(nab.df['plot_date'],
####               nab.all_meso_vol_sun_stim_chemilum_corrected.physical, 'k.')
#### plt.plot_date(iday_plot_dates+0.5, vol_corrected_biweight, 'r.')
#### plt.errorbar(iday_plot_dates+0.5, vol_corrected_biweight.value,
####              yerr=vol_corrected_mad_std.value, fmt='r.')
#### ax.set_ylim([0, 0.02])
#### ax.tick_params(axis='x', labelrotation = 45)
#### plt.show()

#a = nab.vs_time()

    #@pgproperty
    #def model(self):
    #
    #    """Returns `dict` of model calculations with tag
    #    'corrected_volumetric_mag' being the current best model
    #
    #    """
    #    
    #    ext_corrected_mag = self.standard_star_obj(
    #        extinction_correct(self.instr_mag, self.airmass, 'Na_on'))
    #
    #    airmass_poly = iter_polyfit(self.airmass, ext_corrected_mag,
    #                                deg=1, max_resid=1)
    #    na_column_poly = airmass_poly.deriv()
    #    na_column = na_column_poly(0)*self.ext_corrected_mag.unit
    #    volumetric_mag = extinction_correct(ext_corrected_mag, airmass,
    #                                        na_column)
    #    
    #    sun_angle_poly = iter_polyfit(self.sun_angle_sin, volumetric_mag,
    #                                  deg=1, max_resid=1)
    #    sun_stimulation_poly = sun_angle_poly.deriv()
    #    sun_stimulation = sun_stimulation_poly(0)*volumetric_mag.unit
    #    sun_stim_mag = extinction_correct(volumetric_mag, self.sun_angle_sin,
    #                                      sun_stimulation)
    #    
    #    chemilum_rad = np.radians(self.sun_angle - self.chemilum_delay)
    #    chemilum_sin = np.sin(chemilum_rad)
    #    chemilum_sin = chemilum_sin.value
    #    chemilum_poly = iter_polyfit(chemilum_sin, sun_stim_mag,
    #                                 deg=1, max_resid=2)
    #    chemilum_poly_deriv = chemilum_poly.deriv()
    #    chemilum_deriv = chemilum_poly_deriv(0)*sun_stim_mag.unit
    #    chemilum_mag =  extinction_correct(sun_stim_mag, chemilum_sin,
    #                                       chemilum_deriv)
    #
    #    m = {'uncorrected_volumetric_mag': volumetric_mag,
    #         'sun_stim_corrected_mag': sun_stim_mag,
    #         'sun_stim_chemilum_corrected_mag': chemilum_mag,
    #         'corrected_volumetric_mag': chemilum_mag}
    #    return m
    #
    #@pgproperty
    #def tropospheric_ext(self):
    #    return self.standard_star_obj(
    #        extinction_correct(self.instr_mag, self.airmass, 'Na_on'))

    #@pgproperty
    #def ext_corrected_mag(self):
    #    return self.standard_star_obj(
    #        extinction_correct(self.instr_mag, self.airmass, 'Na_on'))
    #
    #@pgproperty
    #def volumetric_mag(self):
    #    airmass_poly = iter_polyfit(self.airmass, ext_corrected_mag,
    #                                deg=1, max_resid=1)
    #    na_column_poly = airmass_poly.deriv()
    #    na_column = na_column_poly(0)*self.ext_corrected_mag.unit
    #    return extinction_correct(ext_corrected_mag, airmass, na_column)
    #    
    #@pgproperty
    #def sun_stim_mag(self):
    #    """Volumetric mag corrected for solar illumination"""
    #    sun_angle_poly = iter_polyfit(self.sun_angle_sin, self.volumetric_mag,
    #                                  deg=1, max_resid=1)
    #    sun_stimulation_poly = sun_angle_poly.deriv()
    #    sun_stimulation = sun_stimulation_poly(0)*self.volumetric_mag.unit
    #    return extinction_correct(self.volumetric_mag, self.sun_angle_sin,
    #                              sun_stimulation)

    


#on_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_on.fts'
#off_fname = '/data/IoIO/raw/20210617/Jupiter-S007-R001-C001-Na_off.fts'
#on = CorDataBase.read(on_fname)
#off = CorDataBase.read(off_fname)
#bmp_meta = {}
#ccd = na_back_process([on, off], in_name=[on_fname, off_fname],
#                      bmp_meta=bmp_meta)#, show=True)
#pout = na_back_pipeline('/data/IoIO/raw/20210617')
#pout = na_back_directory('/data/IoIO/raw/20210617', outdir='/tmp', read_pout=True, fits_fixed_ignore=True)#, show=True)
#na_back_list = na_back_tree(fits_fixed_ignore=True)#, read_pout=False)

#df = pd.DataFrame(na_back_list)
#df.sort_values('sun_angle')
#
#angle_unit = df['sun_angle_unit'].iloc[0]
#back_unit = df['back_unit'].iloc[0]
## This converts best_back into a proper numpy array with astropy unit 
#instr_mag = u.Magnitude(df['best_back']*u.Unit(back_unit))
#instr_mag_err = u.Magnitude(df['best_back_std']*u.Unit(back_unit))
#
#ss = StandardStar()
#ext_coef, ext_coef_err = ss.extinction_coef('Na_on')
## We need to export things from the dataframe to numpy arrays since
## Pandas series don't play nice with astropy units
#airmass = df['airmass'].to_numpy()
#sun_angle = df['sun_angle'].to_numpy()
#sun_angle = sun_angle*u.Unit(angle_unit)
#sun_angle_sin = np.sin(sun_angle.to(u.rad))
#sun_angle_sin = sun_angle_sin.value
#ext_corrected_mag = ss.extinction_correct(instr_mag, airmass, 'Na_on')
#airmass_poly = iter_polyfit(df['airmass'], ext_corrected_mag,
#                            deg=1, max_resid=1)
#na_column_poly = airmass_poly.deriv()
#na_column = na_column_poly(0)*ext_corrected_mag.unit
#volumetric_mag = extinction_correct(ext_corrected_mag, airmass, na_column)
#
#sun_angle_poly = iter_polyfit(sun_angle_sin, volumetric_mag,
#                              deg=1, max_resid=1)
#sun_stimulation_poly = sun_angle_poly.deriv()
#sun_stimulation = sun_stimulation_poly(0)*volumetric_mag.unit
#sun_stim_mag =  extinction_correct(volumetric_mag, sun_angle_sin,
#                                   sun_stimulation)
#
#chemilum_delay = 90*u.deg
#chemilum_rad = np.radians(sun_angle - chemilum_delay)
#chemilum_sin = np.sin(chemilum_rad)
#chemilum_sin = chemilum_sin.value
#chemilum_poly = iter_polyfit(chemilum_sin, sun_stim_mag,
#                             deg=1, max_resid=2)
#chemilum_poly_deriv = chemilum_poly.deriv()
#chemilum_deriv = chemilum_poly_deriv(0)*sun_stim_mag.unit
#chemilum_mag =  extinction_correct(sun_stim_mag, chemilum_sin,
#                                   chemilum_deriv)
#
### # Shows that we take data on both sides of the sky relative to the
### # sun, but most away from the sun
### ax = plt.subplot()
### plt.plot(df['sun_angle'], df['airmass'], 'k.')
### plt.xlabel(f'sun_angle ({angle_unit})')
### plt.ylabel(f'airmass')
### plt.show()
### 
### ax = plt.subplot()
### plt.errorbar(df['sun_angle'], df['best_back'],
###              yerr=df['best_back_std'], fmt='k.')
### plt.xlabel(f'sun_angle ({angle_unit})')
### plt.ylabel(f'best_back ({back_unit})')
### ax.set_ylim([0, 0.15])
### plt.show()
### 
### # Shows correlation that high airmasses make *more* background.  Must
### # be higher column through emission region
### ax = plt.subplot()
### plt.errorbar(df['airmass'], df['best_back'],
###              yerr=df['best_back_std'], fmt='k.')
### plt.xlabel(f'airmass')
### plt.ylabel(f'best_back ({back_unit})')
### ax.set_ylim([0, 0.15])
### plt.show()
### 
### ax = plt.subplot()
### #plt.errorbar(df['airmass'], instr_mag.value,
### #             yerr=instr_mag_err.value, fmt='k.')
### plt.plot(df['airmass'], instr_mag, 'k.')
### plt.xlabel(f'airmass')
### plt.ylabel(f'best_back ({back_unit})')
### plt.gca().invert_yaxis()
### #ax.set_ylim([0, 0.15])
### plt.show()
### 
### ax = plt.subplot()
### #plt.errorbar(df['airmass'], instr_mag.value,
### #             yerr=instr_mag_err.value, fmt='k.')
### plt.plot(df['airmass'], ext_corrected_mag, 'k.')
### plt.plot(df['airmass'], airmass_poly(df['airmass']), 'r-')
### plt.xlabel(f'airmass')
### plt.ylabel(f'extinction corrected best_back ({ext_corrected_mag.unit})')
### plt.gca().invert_yaxis()
### #ax.set_ylim([0, 0.15])
### plt.show()
### 
### ax = plt.subplot()
### #plt.errorbar(df['airmass'], instr_mag.value,
### #             yerr=instr_mag_err.value, fmt='k.')
### plt.plot(df['airmass'], volumetric_mag, 'k.')
### plt.xlabel(f'airmass')
### plt.ylabel(f'volumetric Na emission')
### plt.gca().invert_yaxis()
### #ax.set_ylim([0, 0.15])
### plt.show()
#
#ax = plt.subplot()
##plt.errorbar(df['airmass'], instr_mag.value,
##             yerr=instr_mag_err.value, fmt='k.')
#plt.plot(df['sun_angle'], volumetric_mag, 'k.')
#plt.plot(df['sun_angle'],
#         sun_angle_poly(np.sin(np.radians(df['sun_angle']))), 'r.')
#plt.xlabel(f'sun angle ({angle_unit})')
#plt.ylabel(f'volumetric Na emission')
#plt.gca().invert_yaxis()
##ax.set_ylim([0, 0.15])
#plt.show()
#
##ax = plt.subplot()
###plt.errorbar(df['airmass'], instr_mag.value,
###             yerr=instr_mag_err.value, fmt='k.')
##plt.plot(np.sin(np.radians(df['sun_angle'])), sun_stim_mag, 'k.')
##plt.xlabel(f'sin(sun_angle)')
##plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
##plt.gca().invert_yaxis()
###ax.set_ylim([0, 0.15])
##plt.show()
#
#ax = plt.subplot()
##plt.errorbar(df['airmass'], instr_mag.value,
##             yerr=instr_mag_err.value, fmt='k.')
#plt.plot(df['sun_angle'], sun_stim_mag, 'k.')
#plt.plot(df['sun_angle'],
#         chemilum_poly(np.sin(chemilum_rad)), 'r.')
#plt.xlabel(f'sun_angle ({angle_unit})')
#plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
#plt.gca().invert_yaxis()
##ax.set_ylim([0, 0.15])
#plt.show()
#
#ax = plt.subplot()
##plt.errorbar(df['airmass'], instr_mag.value,
##             yerr=instr_mag_err.value, fmt='k.')
#plt.plot(df['sun_angle'], chemilum_mag, 'k.')
#plt.xlabel(f'sun_angle ({angle_unit})')
#plt.ylabel(f'volumetric Na emission corrected for sun stimulation and chemiluminescence')
#plt.gca().invert_yaxis()
##ax.set_ylim([0, 0.15])
#plt.show()
#
#ax = plt.subplot()
##plt.errorbar(df['airmass'], instr_mag.value,
##             yerr=instr_mag_err.value, fmt='k.')
#plt.plot(df['airmass'], chemilum_mag, 'k.')
#plt.xlabel(f'airmass')
#plt.ylabel(f'volumetric Na emission corrected for sun stimulation')
#plt.gca().invert_yaxis()
##ax.set_ylim([0, 0.15])
#plt.show()
#
#
#err_bar = df['best_back_std'].to_numpy()
#err_bar *= chemilum_mag.unit
#ax = plt.subplot()
#plt.plot_date(df['plot_date'], chemilum_mag.physical, 'k.')
##plt.errorbar(df['plot_date'], corrected_volumetric.value,
##             yerr=err_bar.value, fmt='k.')
#plt.ylabel(f'corrected volumetric Na emission ({back_unit})')
#plt.gcf().autofmt_xdate()
#ax.set_ylim([0, 0.02])
#plt.show()
#
#err_bar = df['best_back_std'].to_numpy()
#err_bar *= chemilum_mag.unit
#ax = plt.subplot()
#plt.plot_date(df['plot_date'], chemilum_mag, 'k.')
##plt.errorbar(df['plot_date'], corrected_volumetric.value,
##             yerr=err_bar.value, fmt='k.')
#plt.ylabel(f'corrected volumetric Na emission ({back_unit})')
#plt.gcf().autofmt_xdate()
##ax.set_ylim([0, 0.02])
#plt.show()
