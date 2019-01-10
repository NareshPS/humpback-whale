#Python logging
import logging.config as base_logging_config
import logging as base_logging

#Path manipulations
from os import path

#Yaml parsing
import yaml

#Constants
from common import constants

def initialize(entry_point):
    """It initializes logging configuration.
    
    Arguments:
        module_name {string} -- A string representing the name of the entry point file.
    """
    #Config file path
    filename = path.join(constants.LOG_CONFIG_PATH, constants.LOG_CONFIGS[entry_point])

    if path.exists(filename):
        with open(filename, 'rt') as handle:
            config = yaml.safe_load(handle.read())
        base_logging_config.dictConfig(config)
    else:
        raise ValueError("Log config: {} is missing".format(filename))

def get_logger(module_name):
    """It gets a logger given a module name.
    
    Arguments:
        module_name {string} -- A string representing the name of the module
    """
    return base_logging.getLogger(module_name)
