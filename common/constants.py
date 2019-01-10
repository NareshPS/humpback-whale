### Constants ###

#Identify the platform
from sys import platform

#Dataset constants for Windows
if platform.startswith("win"):
    DATASET_MAPPINGS = {
        "train" : "..\\Humpback Whale\\dataset\\train",
        "test" : "..\\Humpback Whale\\dataset\\test",
        "train_preprocessed" : "..\\Humpback Whale\\dataset\\train_preprocessed",
        "test_preprocessed" : "..\\Humpback Whale\\dataset\\test_preprocessed",
        "labels" : "..\\Humpback Whale\\dataset\\train.csv",
        "train_tuples" : "..\\Humpback Whale\\dataset\\train_tuples.bin"
    }

    TENSORBOARD_LOGS_LOC = "logs\\"
else:
    DATASET_MAPPINGS = {
        "train" : "dataset/train",
        "test" : "dataset/test",
        "train_preprocessed" : "dataset/train_preprocessed",
        "test_preprocessed" : "dataset/test_preprocessed",
        "labels" : "dataset/train.csv",
        "train_tuples" : "dataset/train_tuples.bin"
    }

    TENSORBOARD_LOGS_LOC = "logs/"

DATASET_NAMES = ["train", "test", "train_preprocessed", "test_preprocessed"]
TRAIN_TUPLE_HEADERS = ['Anchor', 'Sample', 'Label', 'Similar']

#Input image shape
INPUT_SHAPE = (224, 224, 3)

#Label mapping input source headers
IMAGE_HEADER_NAME = "Image"
LABEL_HEADER_NAME = "Id"

#Feature vector dimensions
FEATURE_VECTOR_DIMS = 300

#Feature model options
FEATURE_MODELS = ["resnet", "inceptionv3"]

#Logging constants
LOG_CONFIG_PATH = 'configuration'
LOG_CONFIGS = {
                'siamese_input_tuples.py': 'siamese_input_tuples.yml',
                'siamese_network.py': 'siamese_network.yml'
            }