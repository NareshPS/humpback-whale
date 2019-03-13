#Keras imports
from keras import backend as K

#Load models from the disk
from keras.models import load_model

#Image data generation, transformation, and prediction
from operation.image import ImageDataGeneration
from operation.transform import ImageDataTransformation
from operation.prediction import Prediction
from operation.input import TrainingParameters, InputParameters, ImageGenerationParameters, update_params
from model.callback import BatchTrainStateCheckpoint

#Training
from operation.training import ImageTraining

#Data operations
import numpy as np
import pandas as pd

#Load and save objects to disk
from pandas import read_csv
from pickle import dump as pickle_dump

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed
from imgaug import seed as imgaug_seed

#Dropbox store
from client.dropbox import DropboxConnection

#Constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser
from common.parse import kv_str_to_tuple

#Input files
from iofiles.input_file import InputFiles, ModelInput, InputDataFile, ResultFile

#Path manipulations
from pathlib import Path

#Logging
from common import logging

#Rounding off
from math import ceil

#Metric recording
from common.metric import Metric

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')
    parser.add_argument(
        '-m', '--model_name',
        required = True,
        help = 'It specifies the name of the model to train.')
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
        '-b', '--batch_size',
        default = 128, type = int,
        help = 'It specifies the training batch size.')
    parser.add_argument(
        '-c', '--image_cache_size',
        default = 512, type = int,
        help = 'It specifies the image cache size.')
    parser.add_argument(
        '-s', '--validation_split',
        type = float,
        help = 'It specifies the validation split on the training dataset. It must be a float between 0 and 1')
    parser.add_argument(
        '-r', '--learning_rate',
        type = float,
        help = 'It specifies the learning rate of the optimization algorithm. It must be a float between 0 and 1')
    parser.add_argument(
        '-t', '--transformations',
        nargs = '+', default = [],
        type = kv_str_to_tuple,
        help = 'It specifies transformation parameters. Options: {}'
                    .format(ImageDataTransformation.Parameters().__dict__.keys()))
    parser.add_argument(
        '-x', '--num_fit_images',
        default = 1000, type = int,
        help = 'It specifies the number of images to send to fit()')
    parser.add_argument(
        '--epoch_id',
        default = 0, type = int,
        help = 'It specifies the start epoch id.')
    parser.add_argument(
        '--batch_id',
        default = 0, type = int,
        help = 'It specifies the start batch id.')
    parser.add_argument(
        '-e', '--number_of_epochs',
        type = int, default = 1,
        help = 'It specifies the number of epochs to train per input set.')
    parser.add_argument(
        '--input_shape',
        default = [224, 224, 3],
        type = int, nargs = 3,
        help = 'It specifies the shape of the image input.')
    parser.add_argument(
        '--number_prediction_steps', default = 2,
        type = int,
        help = 'It specifies the number of prediction steps to evaluate trained model.')
    parser.add_argument(
        '--checkpoint_batch_interval', default = 1,
        type = int,
        help = 'It specifies the number of batches after which to take a checkpoint.')
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

def batch_train_state_callback(model_name, checkpoint_batch_interval, dropbox):
    """It creates the state checkpoint callback that provides callbacks to training events.

    Arguments:
        model_name {string} -- The name of the model.
        checkpoint_batch_interval {int} -- It specifies the number of batches after which to take a checkpoint
        dropbox {client.dropbox.DropboxConnection} -- The dropbox client.
    """ 
    #Initialize input files
    model_input = ModelInput(model_name)
    input_data_file = InputDataFile()
    result_file = ResultFile()

    state_checkpoint_callback = BatchTrainStateCheckpoint(
                                        batch_input_files = [model_input, result_file],
                                        checkpoint_batch_interval = checkpoint_batch_interval,
                                        epoch_input_files = [input_data_file, result_file],
                                        dropbox = dropbox)

    return state_checkpoint_callback

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    #Extract required parameters
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Input data parameters
    input_params = InputParameters(args)
    logger.info('Input parameters: %s', input_params)

    #Image generation paramters
    image_generation_params = ImageGenerationParameters(args)
    logger.info('Image generation parameters: %s', image_generation_params)

    #Training parameters
    training_params = TrainingParameters(args)
    logger.info('Training parameters: %s', training_params)

    #Transformation parameters
    transformation_params = ImageDataTransformation.Parameters.parse(dict(args.transformations))
    logger.info('Transformation parameters: %s', transformation_params)

    #Dropbox parameters
    dropbox_parameters = args.dropbox_parameters

    #Dropbox connection placeholder
    dropbox = None

    if dropbox_parameters:
        dropbox_params = DropboxConnection.Parameters(dropbox_parameters[0], dropbox_parameters[1])
        dropbox = DropboxConnection(dropbox_params)

        logger.info('Dropbox parameters:: dropbox_params: %s', dropbox_params)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)
    imgaug_seed(seed)

    #Input data file
    input_data_file = InputDataFile()
    input_data_file_name = input_data_file.file_name(0, training_params.epoch_id)

    #Prepare input files
    input_files_client = InputFiles(dropbox)
    input_data_file_path = input_files_client.get_all([input_data_file_name])[input_data_file_name]

    #Input data frame
    input_data = read_csv(input_data_file_path, index_col = 0)

    #Update input data parameters
    num_classes = max(getattr(input_data, image_generation_params.label_col)) + 1
    image_generation_params_update = dict(num_classes = num_classes)
    update_params(image_generation_params, **image_generation_params_update)

    logger.info('Updated input data parameters: %s', input_params)
 
    #Model input
    model_input = ModelInput(input_params.model_name)
    model_file = model_input.file_name(training_params.batch_id, training_params.epoch_id)

    #Add to the list of input files
    input_files = input_files_client.get_all([model_file])
    
    #Model file
    model_file = input_files[model_file]
    model = load_model(str(model_file))

    logger.info("Loaded model from: {}".format(model_file))

    #Checkpoint callback
    checkpoint_callback = batch_train_state_callback(
                                    input_params.model_name,
                                    training_params.checkpoint_batch_interval,
                                    dropbox)

    #Training setup
    trainer = ImageTraining(
                    input_params,
                    training_params,
                    image_generation_params,
                    transformation_params,
                    checkpoint_callback)

    #Train
    #model, result = trainer.batch_train(model, input_data)
    model, result = trainer.train(model, input_data)

    #Compute accuracy
    num_matches = (result[constants.PANDAS_MATCH_COLUMN].to_numpy().nonzero())[0].shape[0]
    num_mismatches = len(result[constants.PANDAS_MATCH_COLUMN]) - num_matches
    accuracy = (num_matches/len(result[constants.PANDAS_MATCH_COLUMN])) * 100.

    summary = """
                Result Dataframe: {}
                Total predictions: {}
                Correct predictions: {}
                Wrong predictions: {}
                Accuracy: {}
            """.format(
                    result,
                    len(result),
                    num_matches,
                    num_mismatches,
                    accuracy)

    print(summary)

    Metric.save(Path('metric_data.metric'))
