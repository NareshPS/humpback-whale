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
    parser = ArgumentParser(description = 'It generates training samples for the siamese network model.')
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

    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Dropbox
    dropbox = None

    if dropbox_parameters:
        dropbox_auth_token = dropbox_parameters[0]
        dropbox_remote_dir = constants.DROPBOX_APP_PATH_PREFIX / dropbox_parameters[1]

        dropbox = DropboxConnection(dropbox_auth_token, dropbox_remote_dir)

        logger.info('Dropbox parameters:: dropbox_remote_dir: %s', dropbox_remote_dir)

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)

    #Log input parameters
    logger.info(
                'Running with parameters model_name: %s log_to_console: %s',
                model_name,
                log_to_console)

    #Required parameters
    train_tuples_df = read_csv(constants.DATASET_MAPPINGS['train_tuples'])

    #Model name
    model_file = model_name + ".h5"

    local_file = Path(model_file).exists()

    #Local file does not exist. Verify if dropbox parameters are provided to enable download.
    if not local_file and not dropbox:
        raise ValueError("File: {} does not exist locally. Please specify dropbox parameters to download.".format(model_file))

    #Download the file from the dropbox
    if not local_file:
        dropbox.download(model_file)