### Constants ###

#Identify the platform
from sys import platform

#Dataset constants for Windows
if platform.startswith("win"):
    RAW_DATASET_MAPPINGS = {
        "train" : "..\\Humpback Whale\\dataset\\train",
        "test" : "..\\Humpback Whale\\dataset\\test",
        "labels" : "..\\Humpback Whale\\dataset\\train.csv"
    }

    PROCESSED_DATASET_MAPPINGS = {
        "train" : "..\\Humpback Whale\\dataset\\train_preprocessed",
        "test" : "..\\Humpback Whale\\dataset\\test_preprocessed",
        "labels" : "..\\Humpback Whale\\dataset\\train.csv",
        "train_tuples" : "..\\Humpback Whale\\dataset\\train_tuples.bin"
    }

    TENSORBOARD_LOGS_LOC = "logs\\"
else:
    RAW_DATASET_MAPPINGS = {
        "train" : "dataset/train",
        "test" : "dataset/test",
        "labels" : "dataset/train.csv"
    }

    PROCESSED_DATASET_MAPPINGS = {
        "train" : "dataset/train_preprocessed",
        "test" : "dataset/test_preprocessed",
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