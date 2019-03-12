#Python logging
import logging.config as base_logging_config
import logging as base_logging

#Path manipulations
from os import path

#Yaml parsing
import yaml

#Constants
from common import constants

def initialize(entry_point, log_to_console = False, no_logging = False):
    """It initializes logging configuration.
    
    Arguments:
        module_name {string} -- A string representing the name of the entry point file.
        log_to_console {boolean} -- A boolean flag to enable console logging.
        no_logging {boolean} -- A boolean flag to disable logging.
    """
    #Config file path
    filename = path.join(constants.LOG_CONFIG_PATH, constants.LOG_CONFIGS[entry_point])

    if path.exists(filename):
        with open(filename, 'rt') as handle:
            config = yaml.safe_load(handle.read())

            #Apply no logging flag
            config = _apply_no_logging(config, no_logging)
            
            #Apply console logging flag
            config = _apply_console_logging(config, log_to_console)
        base_logging_config.dictConfig(config)
    else:
        raise ValueError("Log config: {} is missing".format(filename))

def get_logger(module_name):
    """It gets a logger given a module name.
    
    Arguments:
        module_name {string} -- A string representing the name of the module

    Returns:
        A Logger object.
    """
    return base_logging.getLogger(module_name)

def _apply_console_logging(config, log_to_console):
    """If console logging is enabled, it updates the configuration to that effect.
    
    Arguments:
        config {dict} -- YAML configuration
        log_to_console {boolean} -- A boolean flag to enable console logging.
    """
    if log_to_console:
        config['root']['handlers'].append('console')

        #Overwrite the trace level with console trace level
        config['root']['level'] = config['handlers']['console']['level']

    return config

def _apply_no_logging(config, no_logging):
    """If no_logging is enabled, the handler list is set to empty.
    
    Arguments:
        config {dict} -- YAML configuration
        no_logging {boolean} -- A boolean flag to disable logging.
    """
    if no_logging:
        config['root']['handlers'] = []

    return config
