#Path manipulations
from pathlib import Path

#Tuple generation
from siamese.tuple import TupleGeneration

#Load and save objects to disk
from pandas import read_csv

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It generates training samples for the siamese network model.')

    parser.add_argument(
        '-i', '--input_labels',
        required = True, type = Path,
        help = 'It specifies the location of the input labels in csv format.')
    parser.add_argument(
        '-c', '--input_cols',
        required = True, nargs = 2,
        help = 'It specifies the tuple of the names of image and label column in the dataframe.')
    parser.add_argument(
        '-p', '--num_positive_samples',
        default = 5, type = int,
        help = 'It defines the number of positive samples per image.')
    parser.add_argument(
        '-n', '--num_negative_samples',
        default = 5, type = int,
        help = 'It defines the number of negative samples per image.')
    parser.add_argument(
        '-o', '--overwrite',
        action = 'store_true', default=False,
        help = 'If specified, overwrites the existing tuple file.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    input_labels = args.input_labels
    input_cols = args.input_cols
    num_positive_samples = args.num_positive_samples
    num_negative_samples = args.num_negative_samples
    overwrite = args.overwrite
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters num_positive_samples: %d num_negative_samples: %d',
                num_positive_samples, 
                num_negative_samples)

    logger.info(
                'Additional parameters input_labels: %s overwrite: %s log_to_console: %s',
                input_labels,
                overwrite, 
                log_to_console)

    #Required inputs
    label_df = read_csv(input_labels)
    train_tuples_columns = constants.INPUT_TUPLE_HEADERS
    output_file_name = Path(
                        "{}_p{}_n{}.{}".format(
                                            constants.INPUT_TUPLE_FILE_PREFIX,
                                            num_positive_samples, 
                                            num_negative_samples,
                                            constants.INPUT_TUPLE_FILE_EXTENSION))
    image_col, label_col = input_cols

    if not output_file_name.exists() or overwrite:
        #Generation object
        tuple_generation = TupleGeneration(label_df, image_col, label_col, train_tuples_columns)

        #Tuple DataFrame
        train_tuple_df = tuple_generation.get_tuples(num_positive_samples, num_negative_samples)

        #Write to disk
        train_tuple_df.to_csv(output_file_name)
    else:
        raise ValueError("File: {} already exists. Please specify overwrite flag to regenerate.".format(output_file_name))