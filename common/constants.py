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
                'train.py' : 'train.yml',
                'unittest' : 'unittest.yml',
                'evaluate.py' : 'evaluate.yml',
                'evaluate_inputs.py' : 'evaluate_inputs.yml',
                'predict.py' : 'predict.yml',
                'augment.py' : 'augment.yml',
                'classify_labels.py' : 'classify_labels.yml',
                'rebalance.py' : 'rebalance.yml',
                'consolidate_result.py' : 'consolidate_result.yml'
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

#Input files
INPUT_RESULT_FILE_PREFIX = 'result'
PREDICTION_INPUT_DATA_FILE_NAME_GUIDANCE = 'prediction_input_data'
PREDICTION_RESULT_FILE_NAME_GUIDANCE = 'prediction_result'

#Output files
CONSOLIDATED_RESULT_FILE = Path('consolidated_result.dmp')
