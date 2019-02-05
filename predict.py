"""It evaluates a model for the fidelity.
"""
#Dropbox
from client.dropbox import DropboxConnection

#Load objects from the disk
from pandas import read_csv

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It produces insights on the dataset and the model.')
    parser.add_argument(
        '-m', '--model_name',
        required = True,
        help = 'It specifies the name of the model to evaluate.')
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to download the checkpoints.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args.model_name, args.dropbox_parameters, args.log_to_console

if __name__ == "__main__":
    #Parse commandline arguments
    model_name, dropbox_parameters, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Dropbox
    dropbox = None

    if dropbox_parameters:
        dropbox_auth_token = dropbox_parameters[0]
        dropbox_remote_dir = constants.DROPBOX_APP_PATH_PREFIX / dropbox_parameters[1]

        dropbox = DropboxConnection(dropbox_auth_token, dropbox_remote_dir)

        logger.info('Dropbox parameters:: dropbox_remote_dir: %s', dropbox_remote_dir)

    #Log input parameters
    logger.info(
                'Running with parameters model_name: %s log_to_console: %s',
                model_name,
                log_to_console)

    #Model file
    model_file = Path("{}.h5".format(model_name))

    #Local file does not exist. Verify if dropbox parameters are provided to enable download.
    if not model_file.exists() and not dropbox:
        raise ValueError("File: {} does not exist locally. Please specify dropbox parameters to download.".format(model_file))

    #Download the file from the dropbox
    if not model_file.exists():
        dropbox.download(model_file)

    #Model file is successfully downloaded
    if not model_file.exists():
        raise ValueError("Model file: {} is not found.".format(model_file))

    

    