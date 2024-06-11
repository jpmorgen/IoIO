"""Basic globals used throughout the IoIO reduction and analysis system"""

import os

fsroot = os.path.abspath(os.sep)
IoIO_ROOT = os.path.join(fsroot, 'data', 'IoIO')
RAW_DATA_ROOT = os.path.join(IoIO_ROOT, 'raw')




# --> Should have all of the directories for all of the components?
# --> Probably so that those modules don't need to be imported when
# --> getting the file info 

# They could be functions somewhere, that take IoIO_ROOT as the
# argument, or something

