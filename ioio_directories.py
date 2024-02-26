"""Define the top-level IoIO directories in this file to avoid
circular imports"""

import os

IoIO_ROOT = '/data/IoIO'
RAW_DATA_ROOT = os.path.join(IoIO_ROOT, 'raw')
