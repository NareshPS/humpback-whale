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
        '-p', '--num_positive_samples',
        default = 5, type = int,
        help = 'It defines the number of positive samples per image.')
    parser.add_argument(
        '-n', '--num_negative_samples',
        default = 5, type = int,
        help = 'It defines the number of negative samples per image.')
    parser.add_argument(
        '-f', '--force_write',
        action = 'store_true', default=False,
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
    num_positive_samples = args.num_positive_samples
    num_negative_samples = args.num_negative_samples
    force_write = args.force_write
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Parameters:: input_file: %s output_file: %s', input_file, output_file)
    logger.info('Parameters:: input_cols: %s output_cols: %s', input_cols, output_cols)
    logger.info('Running with parameters num_positive_samples: %d num_negative_samples: %d', num_positive_samples, num_negative_samples)
    logger.info('Additional parameters force_write: %s log_to_console: %s', force_write, log_to_console)

    #Required inputs
    input_df = read_csv(input_file)

    #Output file
    output_file_stem = output_file.stem
    output_file_extension = output_file.suffix
    output_file_path = Path(
                        "{}_p{}_n{}{}".format(
                                            output_file_stem,
                                            num_positive_samples, 
                                            num_negative_samples,
                                            output_file_extension))
    image_col, label_col = input_cols

    if output_file_path.exists() and not force_write:
        raise ValueError("File: {} already exists. Please specify overwrite flag to regenerate.".format(output_file_path))

    #Generation object
    tuple_generation = TupleGeneration(input_df, image_col, label_col, output_cols)

    #Tuple DataFrame
    output_df = tuple_generation.get_tuples(num_positive_samples, num_negative_samples)

    #Shuffle the tuple
    output_df = output_df.sample(frac = 1).reset_index(drop = True)

    #Write to disk
    output_df.to_csv(output_file_path)

    #Statistics
    input_size = len(input_df)
    output_size = len(output_df)
    expected_output_size = (num_positive_samples + num_negative_samples)*input_size
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
