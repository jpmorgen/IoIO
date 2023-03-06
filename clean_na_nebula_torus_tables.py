"""Clean up tables for Zenodo"""
from astropy.table import QTable

t_na = QTable.read('/data/IoIO/Na_nebula/Na_nebula_cleaned.ecsv')
t_na = t_na[~t_na['mask']]
t_torus = QTable.read('/data/IoIO/Torus/Torus_cleaned.ecsv')
t_torus = t_torus[~t_torus['mask']]

# Incomplete until reprocess.  Shutter time uncertainties < 0.5s
t_na.remove_column('tavg_uncertainty')
t_na.write('/home/jpmorgen/Papers/io/IoIO_2017--2023_GRL/Na_nebula_Morgenthaler_etal_2023.ecsv')

remove_cols = ['tavg_uncertainty', 'ansa_left_r_peak', 'ansa_left_r_peak_err', 'ansa_left_r_stddev', 'ansa_left_r_stddev_err', 'ansa_left_r_amplitude', 'ansa_left_r_amplitude_err', 'ansa_left_y_peak', 'ansa_left_y_peak_err', 'ansa_left_y_stddev', 'ansa_left_y_stddev_err', 'ansa_right_r_peak', 'ansa_right_r_peak_err', 'ansa_right_r_stddev', 'ansa_right_r_stddev_err', 'ansa_right_r_amplitude', 'ansa_right_r_amplitude_err', 'ansa_right_y_peak', 'ansa_right_y_peak_err', 'ansa_right_y_stddev', 'ansa_right_y_stddev_err']

t_torus.remove_columns(remove_cols)
t_torus.write('/home/jpmorgen/Papers/io/IoIO_2017--2023_GRL/Torus_Morgenthaler_etal_2023.ecsv')

