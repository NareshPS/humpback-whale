#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from pathlib import Path

#Logging
from common import logging

#Dropbox client
from client.dropbox import DropboxConnection

#Input files
from iofiles.input_file import ResultFile, InputFiles

#Pickle
from pickle import load as pickle_load

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')

    parser.add_argument(
        '-f', '--file_prefix',
        required = True,
        help = 'It specifies the prefix of files to be combined.')
    parser.add_argument(
       '-e', '--number_of_epochs',
       default = 1, type = int,
       help = 'The number of epochs')
    parser.add_argument(
       '-b', '--number_of_batches',
       default = 1, type = int,
       help = 'The number of batches')  
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to upload the checkpoints.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #Parse commandline arguments
    args = parse_args()

    #Extract parameters
    file_prefix = args.file_prefix
    number_of_batches = args.number_of_batches
    number_of_epochs = args.number_of_epochs
    dropbox_parameters = args.dropbox_parameters
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Result file
    result_file = ResultFile()

    #Dropbox client placeholder
    dropbox = None
    
    if dropbox_parameters:
        dropbox = DropboxConnection.get_client(dropbox_parameters[0], dropbox_parameters[1])

    #Input files placeholder
    input_files = []

    for epoch_id in range(number_of_epochs):
        for batch_id in range(number_of_batches):
            input_files.append(result_file.file_name(batch_id, epoch_id))

    #Log the required pickled object file names
    logger.info('Pickled input files: %s', input_files)

    #Get the local files, and download the remote files
    input_files_client = InputFiles(dropbox)
    available_file_paths = input_files_client.get_all(input_files)

    for file_path in input_files:
        #Composed file path for local or remote files.
        available_file_path = available_file_paths[file_path]

        #Write the name of the pickled object file
        logger.info('Picked available file path: %s', available_file_path)

        #Load the pickled objects from the disk
        pickled_object = pickle_load(available_file_path.open(mode = 'rb'))

        print(type(pickled_object))