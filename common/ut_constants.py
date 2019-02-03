### Constants ###

#Identify the platform
from sys import platform

#Path manipulations
from pathlib import Path

#Dataset constants for Windows
is_windows = True if platform.startswith('win') else False

#Unittest store
UT_DATA_STORE = Path('tests\\store' if is_windows else 'tests/store')

#Training data path
UT_TRAIN_STORE = Path(UT_DATA_STORE) / 'train'

#Unitest logging class
UT_LOGGING_CLASS = 'unittest'

#Dropbox parameters
UT_DROPBOX_STORE = Path(UT_DATA_STORE) / 'dropbox'