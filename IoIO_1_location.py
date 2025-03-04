# Got IoIO alt from a combination of Google Earth and USGS topo
# maps.  See IoIO.notebk Sun Apr 28 12:50:37 2019 EDT.  My
# conclusion was that the altitude was compatible with WGS84,
# though Google Maps apparently uses EGM96 ellipsoid for heights
# and WGS84 for lat-lon.  But EarthLocation does not support
# EGM96, so use WGS84, which is a best-fit ellipsoid
IOIO_1_LOCATION = EarthLocation.from_geodetic(
    '110 15 25.13 W', '31 56 28.30 N', 1095.143, 'WGS84')
IOIO_1_LOCATION.info.name = 'IoIO_1'
IOIO_1_LOCATION.info.meta = {
    'longname': 'Io Input/Output observatory Benson AZ USA'}

