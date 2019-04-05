"""It evaluates a model for the fidelity.
"""
#Dropbox
from client.dropbox import DropboxConnection

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed

#Load objects from the disk
from pandas import read_csv
from keras.models import load_model

#Image data generation
from operation.image import ImageDataGeneration
from operation.input import InputParameters, TrainingParameters, ImageGenerationParameters, update_params
from iofiles.input_file import InputDataFile, InputFiles, ModelInput

#Predictions
from operation.prediction import Prediction

#Ceiling roundoff
from math import ceil

#Useful constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It produces insights on the dataset and the model.')
    parser.add_argument(
        '-m', '--model_name',
        required = True,
        help = 'It specifies the name of the model to evaluate.')
    parser.add_argument(
        '-d', '--dataset_location',
        required = True, type = Path,
        help = 'It specifies the input dataset location.')
    parser.add_argument(
        '--image_cols',
        required = True, nargs = '+',
        help = 'It specifies the names of the image column in the dataframe.')
    parser.add_argument(
        '--image_transform_cols',
        nargs = '+',
        help = 'It specifies the names of the image column in the dataframe that are to be transformed.')
    parser.add_argument(
        '--label_col',
        required = True,
        help = 'It specifies the names of the label column.')
    parser.add_argument(
        '--input_shape',
        default = [224, 224, 3],
        type = int, nargs = 3,
        help = 'It specifies the shape of the image input.')
    parser.add_argument(
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to download the checkpoints.')
    parser.add_argument(
        '-b', '--batch_size',
        default = 32, type = int,
        help = 'It specifies the prediction batch size.')
    parser.add_argument(
        '-c', '--image_cache_size',
        default = 32, type = int,
        help = 'It specifies the image cache size.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    #Required params
    input_params = InputParameters(args)
    image_generation_params = ImageGenerationParameters(args)

    dropbox_parameters = args.dropbox_parameters
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info('Running with parameters input_params: %s', input_params)
    logger.info('Additional parameters: image_generation_params: %s log_to_console: %s', image_generation_params, log_to_console)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)

    #Dropbox
    dropbox = None

    if dropbox_parameters:
        dropbox_params = DropboxConnection.Parameters(dropbox_parameters[0], dropbox_parameters[1])
        dropbox = DropboxConnection(dropbox_params)

        logger.info('Dropbox parameters:: dropbox_params: %s', dropbox_params)

    #Model file
    model_file = ModelInput(input_params.model_name)
    model_file_name = model_file.file_name(0, 0)

    #Input data file
    input_data_file = InputDataFile(constants.PREDICTION_INPUT_DATA_FILE_NAME_GUIDANCE)
    input_data_file_name = input_data_file.file_name(0, 0)

    #Prepare input files
    input_files_client = InputFiles(dropbox)
    input_files = input_files_client.get_all([input_data_file_name, model_file_name])

    #Assign input files
    input_data_file_path = input_files[input_data_file_name]
    model_file_path = input_files[model_file_name]

    #Load model
    model = load_model(str(model_file_path))

    #Input data frame
    input_data = read_csv(input_data_file_path, index_col = 0)

    #Update input data parameters
    num_classes = len(getattr(input_data, image_generation_params.label_col).unique())
    image_generation_params_update = dict(num_classes = num_classes)
    update_params(image_generation_params, **image_generation_params_update)
    logger.info('Updated image generation parameters: %s', image_generation_params)

    #Compute predictions
    num_prediction_steps = ceil(len(input_data) / image_generation_params.batch_size)
    predictor = Prediction(model, input_params, image_generation_params)
    predicted_data = predictor.predict(input_data, num_prediction_steps)

    #Compute accuracy
    num_matches = (predicted_data[constants.PANDAS_MATCH_COLUMN].to_numpy().nonzero())[0].shape[0]
    num_mismatches = len(predicted_data[constants.PANDAS_MATCH_COLUMN]) - num_matches
    accuracy = (num_matches/len(predicted_data[constants.PANDAS_MATCH_COLUMN])) * 100.

    print_summary = """
                        Result Dataframe: {}
                        Total predictions: {}
                        Correct predictions: {}
                        Wrong predictions: {}
                        Accuracy: {}
                    """.format(
                            predicted_data,
                            len(predicted_data),
                            num_matches,
                            num_mismatches,
                            accuracy)

    #Print summary
    print(print_summary)
