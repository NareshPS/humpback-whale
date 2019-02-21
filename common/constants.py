### Constants ###

#Identify the platform
from sys import platform

#Path manipulations
from pathlib import Path

#Logging constants
LOG_CONFIG_PATH = 'configuration'
LOG_CONFIGS = {
                'siamese_input_tuples.py' : 'siamese_input_tuples.yml',
                'model_generation.py' : 'model_generation.yml',
                'insights.py' : 'insights.yml',
                'siamese_train.py' : 'siamese_train.yml',
                'unittest' : 'unittest.yml',
                'evaluate.py' : 'evaluate.yml',
                'evaluate_inputs.py' : 'evaluate_inputs.yml',
                'predict.py' : 'predict.yml',
                'augment.py' : 'augment.yml',
                'classify_labels.py' : 'classify_labels.yml',
                'rebalance.py' : 'rebalance.yml'
            }

#Dropbox connection configuration
DROPBOX_ENDPOINT = "https://content.dropboxapi.com/2/files/upload"
DROPBOX_APP_PATH_PREFIX = Path('/run_data/')
DROPBOX_CHUNK_SIZE = 4 * 1024 * 1024 #4 megabytes

#Pandas aggregation column
PANDAS_COUNT_AGG_COLUMN = 'Count'
PANDAS_PCT_AGG_COLUMN = 'Percentage'
PANDAS_COUNT_HIST_COLUMN = 'Histogram Count'
PANDAS_COUNT_BIN_COLUMN = 'Bin Count'
PANDAS_PREDICTION_COLUMN = 'Prediction'
PANDAS_MATCH_COLUMN = 'Match'

#PIL Image constants
PIL_IMAGE_RGB_COLOR_MODE = 'RGB'
