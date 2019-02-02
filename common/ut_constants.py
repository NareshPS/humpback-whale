### Constants ###

#Identify the platform
from sys import platform

#Path manipulations
from os import path

#Dataset constants for Windows
if platform.startswith("win"):
    UT_DATA_STORE = "tests\\store"
else:
    UT_DATA_STORE = "tests/store"

#Training data path
UT_TRAIN_STORE = path.join(UT_DATA_STORE, "train")

#Unitest logging class
UT_LOGGING_CLASS = 'unittest'