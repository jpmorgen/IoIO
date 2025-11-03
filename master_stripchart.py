import os
from datetime import date

import matplotlib.pyplot as plt

import pandas as pd

from astropy import log
from astropy.table import QTable

from IoIO.utils import savefig_overwrite
from IoIO.na_nebula import (BASE as NA_NEBULA_BASE, NA_NEBULA_ROOT,
                            plot_na_nebula_surf_brights)
from IoIO.torus import (BASE as TORUS_BASE, TORUS_ROOT,
                        plot_ansa_surf_brights, plot_torus_epsilons,
                        plot_torus_ansa_pos, plot_ansa_r_cont,
                        plot_ansa_r_slope)
from IoIO.mme import MODEL, plot_mme, plot_mme_corr
from IoIO.juno import juno_pj_axis

def read_tables(t_na_nebula=None,
                na_nebula_day_table=None,
                t_torus=None,
                torus_day_table=None,
                df_mme=None,
                torus_mme=None):
    if t_na_nebula is None:
        t_na_nebula = os.path.join(NA_NEBULA_ROOT, NA_NEBULA_BASE+'_cleaned.ecsv')
    if isinstance(t_na_nebula, str):
        t_na_nebula = QTable.read(t_na_nebula)
    if na_nebula_day_table is None:
        na_nebula_day_table = os.path.join(NA_NEBULA_ROOT,
                                           NA_NEBULA_BASE+'_day_table.ecsv')
    if isinstance(na_nebula_day_table, str):
        na_nebula_day_table = QTable.read(na_nebula_day_table)
    if t_torus is None:
        t_torus = os.path.join(TORUS_ROOT, TORUS_BASE+'_cleaned.ecsv')
    if isinstance(t_torus, str):
        t_torus = QTable.read(t_torus)
    if torus_day_table is None:
        torus_day_table = os.path.join(TORUS_ROOT,
                                           TORUS_BASE+'_day_table.ecsv')
    if isinstance(torus_day_table, str):
        torus_day_table = QTable.read(torus_day_table)
    if df_mme is None:
        df_mme = os.path.join(TORUS_ROOT,
                              f'Jupiter_mme_{MODEL}.csv')
    if isinstance(df_mme, str):
            df_mme = pd.read_csv(df_mme, index_col=[0])
            df_mme.index = pd.to_datetime(df_mme.index)
    if torus_mme is None:
        torus_mme = os.path.join(TORUS_ROOT,
                                 f'{TORUS_BASE}_mme_{MODEL}.csv')
    if isinstance(torus_mme, str):
        torus_mme = pd.read_csv(torus_mme, index_col='tavg')
        torus_mme.index = pd.to_datetime(torus_mme.index)

    table_dict = {'t_na_nebula' : t_na_nebula,
                  'na_nebula_day_table' : na_nebula_day_table,
                  't_torus' : t_torus,
                  'torus_day_table' : torus_day_table,
                  'df_mme' : df_mme,
                  'torus_mme' : torus_mme}
    return table_dict
                  
def master_stripchart(
        t_na_nebula=None,
        na_nebula_day_table=None,
        t_torus=None,
        torus_day_table=None,
        df_mme=None,
        torus_mme=None,
        plots=['na_nebula', 'ansa_brights'],# 'epsilons', 'ansa_pos'],
        #plots=['epsilons', 'mme', 'mme_corr'],
        plot_mme_corr_kwargs_list=[None],
        start=None,
        stop=None,
        outname=None,
        days_per_in=None,
        figsize=[12, 10],
        **kwargs):

    if start is None:
        start = date.fromisoformat('2017-01-01')
    if stop is None:
        stop = date.today()
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(stop, str):
        stop = date.fromisoformat(stop)

    if days_per_in is not None:
        dt = stop-start
        figsize[0] = dt.days/days_per_in        

    nplots = len(plots)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nplots, hspace=0)
    axs = gs.subplots(sharex=True)

    iplot = 0
    for plotname in plots:
        if nplots == 1:
            ax = axs
        else:
            ax = axs[iplot]
        top_axis = iplot == 0
        if plotname == 'na_nebula':
            plot_na_nebula_surf_brights(
                t_na_nebula,
                na_nebula_day_table,
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)            
        elif plotname == 'ansa_brights':    
            plot_ansa_surf_brights(
                t_torus, torus_day_table,
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'epsilons':    
            plot_torus_epsilons(
                t_torus, torus_day_table, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'ansa_pos':
            plot_torus_ansa_pos(
                t_torus, torus_day_table,
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'ansa_dawn_r_stddev':
            plot_dawn_r_stddev(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'ansa_dusk_r_stddev':
            plot_dusk_r_stddev(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'plot_dawn_slope':
            plot_dawn_slope(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'plot_dusk_slope':
            plot_dusk_slope(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'plot_ansa_r_cont':
            plot_ansa_r_cont(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
        elif plotname == 'plot_ansa_r_slope':
            plot_ansa_r_slope(
                t_torus, 
                fig=fig, ax=ax,
                tlim=(start, stop),
                **kwargs)
                               
        elif plotname == 'mme':
            plot_mme(
                df_mme, torus_mme,
                fig=fig, ax=ax,
                tlim=(start, stop),
                ylim=(0, 0.16), # --> remove this
                **kwargs)
        elif plotname == 'mme_corr':
            for corr_kwargs in plot_mme_corr_kwargs_list:
                if corr_kwargs is None:
                    corr_kwargs = kwargs
                else:
                    corr_kwargs.update(kwargs)
                plot_mme_corr(
                    torus_mme,
                    fig=fig, ax=ax,
                    tlim=(start, stop),
                    **corr_kwargs)
                if iplot == 0:
                    juno_pj_axis(ax)
                iplot += 1
            iplot -= 1
        else:
            # --> Write code for the rest as needed
            log.error(f'Unknown plot name: {plotname}')

        if iplot == 0:
            juno_pj_axis(ax)
        iplot += 1
        if nplots > 1:
            for ax in axs:
                ax.label_outer()

    plt.tight_layout()
    if outname is None:
        plt.show()
    else:
        savefig_overwrite(outname)
        plt.close()

#master_stripchart()

