#Commandline arguments
from argparse import ArgumentParser

#File operations
from pathlib import Path

#Operations on the input data
from common.pandas import csv_to_dataframe, dataframe_to_csv
from operation.rebalancing import Rebalancing

#Input files
from iofiles.input_file import InputFiles

#Dropbox store
from client.dropbox import DropboxConnection

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It rebalances the input data to improve training performance.')

    parser.add_argument(
        '-i', '--input_data',
        required = True, type = Path,
        help = 'It specifies the path to the input data file.')
    parser.add_argument(
        '--label_col',
        required = True,
        help = 'It specifies the names of the label column.')
    parser.add_argument(
        '--output_file',
        required = True, type = Path,
        help = 'It specifies the location of the output data in csv format.')
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to upload the rebalanced data file.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args.input_data, args.label_col, args.output_file, args.dropbox_parameters, args.log_to_console

if __name__ == '__main__':
    #Parse commandline arguments
    input_data, label_col, output_file, dropbox_parameters, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    logger.info(
                'Input parameters:: input_data: %s label_col: %s output_file: %s log_to_console: %s',
                input_data,
                label_col,
                output_file,
                log_to_console)

    #Dropbox connection placeholder
    dropbox = None

    if dropbox_parameters:
        dropbox_params = DropboxConnection.Parameters(dropbox_parameters[0], dropbox_parameters[1])
        dropbox = DropboxConnection(dropbox_params)

        logger.info('Dropbox parameters:: dropbox_params: %s', dropbox_params)

    ####################################### Prepare the input dataset [Start] ############################################
    #Prepare input files
    input_files_client = InputFiles(dropbox)
    input_data = input_files_client.get_all([input_data])[input_data]

    #Input data as pandas data frame
    input_data = csv_to_dataframe(input_data)
    ####################################### Prepare the input dataset [End] ############################################

    ####################################### Rebalance the dataset [Start] ############################################
    #Rebalance the data and obtain the statistics
    rebalancer = Rebalancing(input_data, label_col)
    result, pre_stats, post_stats = rebalancer.rebalance(statistics = True)
    ####################################### Rebalance the dataset [End] ############################################

    #Output to a file
    dataframe_to_csv(result, output_file)

    ####################################### Process output [Start] ############################################
    #Upload the output to dropbox
    if dropbox:
        input_files_client.put_all([output_file])

    print_summary = """
                        Pre-balancing statistics: {}
                        Results: {}
                        Post-balancing statistics: {}
                    """.format(
                            pre_stats,
                            result,
                            post_stats)

    print(print_summary)
    ####################################### Process output [End] ############################################
