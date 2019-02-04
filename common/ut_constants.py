### Constants ###

#Identify the platform
from sys import platform

#Path manipulations
from pathlib import Path

#Dataset constants for Windows
is_windows = True if platform.startswith('win') else False

#Unittest store
DATA_STORE = Path('tests\\store' if is_windows else 'tests/store')

#Training data path
TRAIN_STORE = Path(DATA_STORE) / 'train'

#Unitest logging class
LOGGING_CLASS = 'unittest'

#Dropbox parameters
DROPBOX_STORE = Path(DATA_STORE) / 'dropbox'

#Tuples Dataframe
LABEL_DATAFRAME_PATH = Path(DATA_STORE) / 'label_df.csv'
LABEL_DATAFRAME_LABEL_COL = 'Id'
LABEL_DATAFRAME_IMAGE_COL = 'Image'
TUPLE_DATAFRAME_COLS = ['Anchor', 'Sample', 'Label']