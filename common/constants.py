### Constants ###

#Identify the platform
from sys import platform

#Path manipulations
from pathlib import Path

#Dataset constants for Windows
if platform.startswith("win"):
    DATASET_MAPPINGS = {
        "train" : "..\\Humpback Whale\\dataset\\train",
        "test" : "..\\Humpback Whale\\dataset\\test",
        "train_preprocessed" : "..\\Humpback Whale\\dataset\\train_preprocessed",
        "test_preprocessed" : "..\\Humpback Whale\\dataset\\test_preprocessed"
    }

    TENSORBOARD_LOGS_LOC = "logs\\"
else:
    DATASET_MAPPINGS = {
        "train" : "dataset/train",
        "test" : "dataset/test",
        "train_preprocessed" : "dataset/train_preprocessed",
        "test_preprocessed" : "dataset/test_preprocessed"
    }

    TENSORBOARD_LOGS_LOC = "logs/"

DATASET_NAMES = ["train", "test", "train_preprocessed", "test_preprocessed"]

#Siamese tuples
TRAIN_TUPLE_HEADERS = ['Anchor', 'Sample', 'Label']
TUPLE_FILE_PREFIX = 'input_tuples'
TUPLE_FILE_EXTENSION = "tuples"
TRAIN_TUPLE_LABEL_COL = TRAIN_TUPLE_HEADERS[2]

#Input image shape
INPUT_SHAPE = (224, 224, 3)

#Label mapping input source headers
IMAGE_HEADER_NAME = "Image"
LABEL_HEADER_NAME = "Id"

#Feature model options
FEATURE_MODELS = ["resnet", "inceptionv3"]

#Logging constants
LOG_CONFIG_PATH = 'configuration'
LOG_CONFIGS = {
                'siamese_input_tuples.py' : 'siamese_input_tuples.yml',
                'model_generation.py' : 'model_generation.yml',
                'insights.py' : 'insights.yml',
                'train.py' : 'train.yml',
                'unittest' : 'unittest.yml',
                'evaluate.py' : 'evaluate.yml',
                'evaluate_inputs.py' : 'evaluate_inputs.yml'
            }

#Dropbox connection configuration
DROPBOX_ENDPOINT = "https://content.dropboxapi.com/2/files/upload"
DROPBOX_APP_PATH_PREFIX = Path('/run_data/')
DROPBOX_CHUNK_SIZE = 4 * 1024 * 1024 #4 megabytes