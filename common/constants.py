from sys import platform

### Constants ###

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
        "labels" : "..\\Humpback Whale\\dataset\\train.csv"
    }
else:
    RAW_DATASET_MAPPINGS = {
        "train" : "dataset/train",
        "test" : "dataset/test",
        "labels" : "dataset/train.csv"
    }

    PROCESSED_DATASET_MAPPINGS = {
        "train" : "dataset/train_preprocessed",
        "test" : "dataset/test_preprocessed",
        "labels" : "dataset/train.csv"
    }

DATASET_NAMES = ["train", "test", "train_preprocessed", "test_preprocessed"]

TARGET_IMG_SHAPE = (400, 700)