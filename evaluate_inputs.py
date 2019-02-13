"""It compute statistics on the input tuples.
"""
#Load objects from the disk
from pandas import read_csv

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Siamese tuples
from siamese.tuple import TupleGeneration
from model.evaluation import LabelEvaluation

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It produces insights on the input tuples.')

    parser.add_argument(
        '-i', '--input_labels',
        required = True, type = Path,
        help = 'It specifies the location of the input labels in csv format.')
    parser.add_argument(
        '-c', '--input_cols',
        required = True, nargs = 2,
        help = 'It specifies the tuple of the names of image and label column in the dataframe.')
    parser.add_argument(
        '--output_cols',
        required = True, nargs = '+',
        help = 'It specifies the column names of the output dataframe.')
    parser.add_argument(
        '-p', '--num_positive_samples',
        required = True, type = int,
        help = 'It defines the number of positive samples per image.')
    parser.add_argument(
        '-n', '--num_negative_samples',
        required = True, type = int,
        help = 'It defines the number of negative samples per image.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args.input_labels, args.input_cols, args.output_cols, args.num_positive_samples, args.num_negative_samples, args.log_to_console

if __name__ == "__main__":
    #Parse commandline arguments
    input_labels, input_cols, output_cols, num_positive_samples, num_negative_samples, log_to_console = parse_args()
    
    #Required parameters
    image_col, label_col = input_cols

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters input_labels: %s input_cols: %s num_positive_samples: %d num_negative_samples: %d',
                input_labels,
                input_cols,
                num_positive_samples,
                num_negative_samples)

    logger.info('Parameters:: output_cols: %s', output_cols)

    #Additional parameters
    logger.info('Additional parameters log_to_console: %s', log_to_console)

    #Validation
    if not input_labels.exists():
        raise ValueError('Input labels file: {} does not exist'.format(input_labels))

    #Label dataframe
    label_df = read_csv(input_labels)

    #Tuple generation
    generation = TupleGeneration(label_df, image_col, label_col, output_cols)
    input_tuples = generation.get_tuples(num_positive_samples, num_negative_samples)

    #Label evaluation
    label_evaluation = LabelEvaluation(input_tuples)
    print(label_evaluation.evaluate(output_cols[-1]))