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
        '-i', '--input_data',
        required = True, type = Path,
        help = 'It specifies the location of the input data in csv format.')
    parser.add_argument(
        '--label_col',
        required = True,
        help = 'It specifies the column names of the output dataframe.')
    parser.add_argument(
        '--number_of_bins',
        default = 5,
        help = 'It specifies the number of bins of the label counts.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()
    
    return args.input_data, args.label_col, args.number_of_bins, args.log_to_console

if __name__ == "__main__":
    #Parse commandline arguments
    input_data, label_col, number_of_bins, log_to_console = parse_args()

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Running with parameters input_data: %s label_col: %s number_of_bins: %d', input_data, label_col, number_of_bins)

    #Additional parameters
    logger.info('Additional parameters log_to_console: %s', log_to_console)

    #Input dataframe
    input_data = read_csv(input_data)

    #Label evaluation
    label_evaluation = LabelEvaluation(input_data)
    label_statistics = label_evaluation.distribution(label_col)
    label_count_bins = label_evaluation.bin(label_col, number_of_bins)
    label_count_histogram = label_evaluation.histogram(label_col)

    #Print summary
    print_summary = """
                        Label statistics: {}
                        Binned counts: {}
                        Histogram: {}
                    """.format(
                            label_statistics,
                            label_count_bins,
                            label_count_histogram)

    print(print_summary)
