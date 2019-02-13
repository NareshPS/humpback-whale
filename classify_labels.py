#Commandline arguments
from argparse import ArgumentParser

#Read csv
from pandas import read_csv

#Dataframe
from pandas import DataFrame

#Path manipulations
from pathlib import Path

#Logging
from common import logging

#Bidirectional dictionary
from bidict import frozenbidict

#Pickle
from pickle import dump as pickle_dump

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')

    parser.add_argument(
        '-i', '--input_data',
        required = True, type = Path,
        help = 'It specifies the path to the input data file.')
    parser.add_argument(
        '--label_col',
        required = True,
        help = 'It specifies the names of the label column.')
    parser.add_argument(
        '--mapping_keys',
        required = True, type = Path,
        help = 'It specifies the file name to store label name to class mappings.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #Parse commandline arguments
    args = parse_args()

    #Extract logging parameters
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Parameters
    input_data = args.input_data
    label_col = args.label_col
    mapping_keys = args.mapping_keys

    #Input dataframe
    input_data_df = read_csv(input_data, index_col = 0)
    logger.info('Loaded input data: %s', input_data)

    #Get all labels
    label_names = set(input_data_df[label_col].tolist())
    label_names_size = len(label_names)
    logger.info('Loaded %d label names', label_names_size)

    #Generate label classes
    label_classes = list(range(len(label_names)))
    label_classes_size = len(label_classes)
    logger.info('Created %d label classes', label_classes_size)

    #Output dataframe
    output_df = DataFrame(columns = list(input_data_df))

    print(list(input_data_df))

    #Create names to class mappings
    mappings = frozenbidict(zip(label_names, label_classes))

    for _, row in input_data_df.iterrows():
        label_name = row[label_col]
        label_class = mappings[label_name]

        #Update the row
        row[label_col] = label_class

    #Save the updated dataframe
    input_data_df.to_csv(input_data)

    #Save mappings
    with mapping_keys.open(mode = 'wb') as handle:
        pickle_dump(mappings, handle)

    print_summary = """
                        Input labels: {}
                        Number of classes: {}
                        Deviation: {}
                        Output file: {}
                        Mapping file: {}
                    """.format(
                            label_names_size,
                            label_classes_size,
                            label_classes_size - label_names_size,
                            input_data,
                            mapping_keys)

    print(print_summary)