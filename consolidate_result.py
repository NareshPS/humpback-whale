"""It consolidates the run results.
"""
#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Dropbox store
from client.dropbox import DropboxConnection

#Input files
from iofiles.input_file import InputFiles

#Result object
from model.result import EpochResponse

#Progress bar
from tqdm import tqdm

#Load pickled objects
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from pickle import Unpickler

#Default valued dictionary
from collections import defaultdict

#Response object

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

class RenameUnpickler(Unpickler):
    def find_class(self, module, name):
        if module == 'model.response':
            module = 'model.result'

        return super().find_class(module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

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

    ####################################### Prepare input files [Start] ############################################
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
    ####################################### Prepare input files [End] ############################################

    ####################################### Pull metric names [Start] ############################################
    #Metric names placeholder
    metric_names = None

    #Get a candidate file to pull metric names
    candidate_item = next(iter(input_files))

    with candidate_item.open(mode = 'rb') as handle:
        result = renamed_load(handle)

        #Pull metric names
        metric_names = result.metric_names

    logger.info('Metric names: %s', metric_names)
    ####################################### Pull metric names [End] ############################################

    ####################################### Consolidate results [Start] ############################################
    #Result storage
    epoch_metric_results = defaultdict(dict)

    for file_path, _ in input_files.items():
        with file_path.open(mode = 'rb') as handle:
            result = renamed_load(handle)

            #Pull the epoch id from the result object
            epoch_id = result.epoch_id

            for metric_name in metric_names:
                #Pull the batch results of the epoch
                batch_results = epoch_metric_results[epoch_id, metric_name]

                #Zip batch_ids and metric values to simplify iteration
                batch_metric_values = zip(result.batch_ids, result.get(metric_name))

                #Pull the metric values and map them to the batch ids
                for batch_id, metric_value in batch_metric_values:
                    batch_results[batch_id] = metric_value

    #Consolidated results placeholder
    consolidated_results = defaultdict(list)

    for key, batch_results in epoch_metric_results.items():
        #Get the batch end
        max_batch_id = max(batch_results.keys())

        #Get the result object for the metric and epoch
        consolidated_results[key] = list(map(lambda x : 0.0 if batch_results.get(x) is None else batch_results[x], range(max_batch_id)))

    #Save the consolidated results
    with constants.CONSOLIDATED_RESULT_FILE.open(mode = 'wb') as handle:
        pickle_dump(consolidated_results, handle)
    ####################################### Consolidate results [End] ############################################
