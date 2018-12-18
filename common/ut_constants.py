### Constants ###

#Identify the platform
from sys import platform

#Dataset constants for Windows
if platform.startswith("win"):
    UT_DATA_STORE = "tests\\store"
else:
    UT_DATA_STORE = "tests/store"