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
        '-i', '--input_file',
        required = True, type = Path,
        help = 'It specifies the location of the input labels in csv format.')
    parser.add_argument(
        '-o', '--output_file',
        required = True, type = Path,
        help = 'It specifies the location of the output file.')
    parser.add_argument(
        '-c', '--input_cols',
        required = True, nargs = 2,
        help = 'It specifies the tuple of the names of image and label column in the dataframe.')
    parser.add_argument(
        '--output_cols',
        required = True, nargs = 3,
        help = 'It specifies the column headers of the output dataframe.')
    parser.add_argument(
        '-s', '--num_samples',
        default = 10, type = int,
        help = 'It defines the number of samples per example image.')
    parser.add_argument(
        '-n', '--num_inputs',
        default = 200, type = int,
        help = 'It defines the number of inputs to process')
    parser.add_argument(
        '--triplets',
        action = 'store_true', default = False,
        help = 'It instructs to generate input triplets.')
    parser.add_argument(
        '-f', '--force_write',
        action = 'store_true', default = False,
        help = 'If specified, overwrites the output file.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    input_file = args.input_file
    output_file = args.output_file
    input_cols = args.input_cols
    output_cols = args.output_cols
    num_samples = args.num_samples
    num_inputs = args.num_inputs
    triplets = args.triplets
    force_write = args.force_write
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Parameters:: input_file: %s output_file: %s', input_file, output_file)
    logger.info('Parameters:: input_cols: %s output_cols: %s', input_cols, output_cols)
    logger.info('Parameters:: num_samples: %d num_inputs: %d', num_samples, num_inputs)
    logger.info('Parameters:: triplets: %s', triplets)
    logger.info('Additional parameters force_write: %s log_to_console: %s', force_write, log_to_console)

    #Required inputs
    input_data = read_csv(input_file)
    input_data = input_data if num_inputs == -1 else input_data[:num_inputs]

    image_col, label_col = input_cols

    if output_file.exists() and not force_write:
        raise ValueError("File: {} already exists. Please specify overwrite flag to regenerate.".format(output_file))

    #Generation object
    tuple_generation = TupleGeneration(input_data, image_col, label_col, output_cols)

    #Output data placeholder
    output_data = None

    #Generate tuples with an anchor, a positive, and a negative sample
    if triplets:
        output_data = tuple_generation.get_triplets(num_samples)
    #Generate tuples with an anchor, a sample, and a label
    else:
        num_positive_samples = int(num_samples / 2)
        num_negative_samples = int(num_samples / 2)
        output_data = tuple_generation.get_tuples(num_positive_samples, num_negative_samples)

    #Shuffle the tuple
    output_data = output_data.sample(frac = 1).reset_index(drop = True)

    #Write to disk
    output_data.to_csv(output_file)

    #Statistics
    input_size = len(input_data)
    output_size = len(output_data)
    expected_output_size = num_samples * input_size
    deviation = expected_output_size - output_size

    completion_statistics = """
                                    Tuple Generation Summary
                                    ====================
                                    Input size: {}
                                    Output size: {}
                                    Expected output size: {}
                                    Error: {}
                                    """.format(
                                            input_size,
                                            output_size,
                                            expected_output_size,
                                            deviation)

    #Print the statistics
    print(completion_statistics)
