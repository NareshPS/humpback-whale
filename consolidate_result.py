"""It consolidates the run results.
"""
#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Dropbox store
from client.dropbox import DropboxConnection

#Input files
from iofiles.input_file import InputFiles, ResultFile

#Progress bar
from tqdm import tqdm

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It consolidates the run results.')

    parser.add_argument(
        '-e', '--epoch_data_dirs',
        type = Path,
        nargs = '+', required = True,
        help = 'It specifies the list of directories containing the results.')
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to download the results.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args.epoch_data_dirs, args.dropbox_parameters, args.log_to_console

if __name__ == "__main__":
    #Parse commandline arguments
    epoch_data_dirs, dropbox_parameters, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Running with parameters epoch_data_dirs: %s log_to_console: %d', epoch_data_dirs, log_to_console)

    #Dropbox connection
    dropbox = DropboxConnection.get_client_from_params(dropbox_parameters)

    #Prepare input files
    input_files_client = InputFiles(dropbox)

    #Epoch data files placeholder
    input_files = []

    #Iterate over input epoch stores and enumerate their result files.
    for epoch_store in epoch_data_dirs:
        #Fetch the remote epoch data
        epoch_data = dropbox.list(epoch_store, constants.INPUT_RESULT_FILE_PREFIX)

        #Extract file paths from epoch data
        input_files.extend([file_path for file_path in epoch_data[0]])

    #Create local epoch store locations
    _ = [Path(epoch_store).mkdir(parents = True, exist_ok = True) for epoch_store in epoch_data_dirs]

    #Prepare the required files
    input_files = input_files_client.get_all(input_files)
