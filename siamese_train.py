#Keras imports
from keras import backend as K

#Load models from the disk
from keras.models import load_model

#Data processing
from operation.image import ImageDataGeneration
from operation.transform import ImageDataTransformation
from operation.input import TrainingParameters, InputDataParameters
import numpy as np
import pandas as pd

#Load and save objects to disk
from pandas import read_csv
from pickle import dump as pickle_dump

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed
from imgaug import seed as imgaug_seed

#Save checkpoints
from model.callback import ModelDropboxCheckpoint

#Dropbox store
from client.dropbox import DropboxConnection

#Constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser
from common.parse import kv_str_to_tuple

#Input files
from operation.input_file import InputFiles, ModelInput

#Path manipulations
from pathlib import Path

#Logging
from common import logging

#Rounding off
from math import ceil

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
        '-i', '--input_data',
        required = True, type = Path,
        help = 'It specifies the path to the input data file.')
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
        '--session_id',
        default = 1, type = int,
        help = 'It specifies an identifier of the run.')
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
        '-e', '--number_of_epochs',
        type = int, default = 1,
        help = 'It specifies the number of epochs to train per input set.')
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
        '-p', '--dropbox_parameters',
        nargs = 2,
        help = 'It specifies dropbox parameters required to upload the checkpoints.')
    parser.add_argument(
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')
    parser.add_argument(
        '--num_prediction_steps',
        default = 2, type = int,
        help = 'The number of steps to use for verification by prediction')
    parser.add_argument(
        '--input_data_training_set_size',
        default = 300000, type = int,
        help = 'It specifies the the size of input data set that will be trained in one go.')
    parser.add_argument(
        '--input_data_training_set_id',
        default = 1, type = int,
        help = 'It specifies the set if of the input data set to start training.')
    parser.add_argument(
        '--input_shape',
        default = [224, 224, 3],
        type = int, nargs = 3,
        help = 'It specifies the shape of the image input.')

    args = parser.parse_args()

    return args

def train(
        model,
        set_id,
        input_data_df,
        input_data_params,
        training_params,
        transformation_params,
        dropbox_auth,
        dropbox_dir,
        n_epochs):
    """It executes the training.
    
    Arguments:
        model {A keras model object} -- The keras model object.
        set_id {int} -- The set id of the current training data frame.
        input_data_df {A pandas DataFrame} -- The input data.
        input_data_params {A InputDataParameters object} -- It contains the input parameters.
        training_params {A TrainingParameters object} -- It contains training parameters.
        dropbox_auth {string} -- The authentication token to access dropbox store.
        dropbox_dir {string} -- The dropbox directory to store the generated data.
        n_epochs {int} -- The number of epochs to train the model.
    """
    #Logging
    logger = logging.get_logger(__name__)

    #Transformer
    transformer = ImageDataTransformation(parameters = transformation_params)
    
    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                    input_data_params.dataset_location, input_data_df,
                    input_data_params.input_shape[:2], training_params.batch_size, 
                    input_data_params.num_classes,
                    input_data_params.image_cols, input_data_params.label_col,
                    transform_x_cols = input_data_params.image_transform_cols,
                    validation_split = training_params.validation_split,
                    cache_size = training_params.image_cache_size,
                    transformer = transformer)

    #Fit the data generator
    datagen.fit(n_images = training_params.num_fit_images)

    #Training flow
    train_gen = datagen.flow(subset = 'training')

    #Validation flow
    validation_gen = datagen.flow(subset = 'validation') if training_params.validation_split else None

    if training_params.learning_rate:
        #Update the learning rate
        logger.info("Switching learning rate from: {} to: {}".format(
                                                                K.get_value(model.optimizer.lr),
                                                                training_params.learning_rate))

        K.set_value(model.optimizer.lr, training_params.learning_rate)
        
    #Training callbacks
    dropbox_callback = ModelDropboxCheckpoint(
                                input_data_params.model_name,
                                input_data_params.session_id,
                                set_id,
                                input_data_params.num_df_sets,
                                dropbox_auth = dropbox_auth,
                                dropbox_dir = dropbox_dir)

    #Fit the model the input.
    model.fit_generator(
                    generator = train_gen,
                    validation_data = validation_gen,
                    epochs = n_epochs,
                    callbacks = [dropbox_callback])

    logger.info('Training finished. Trained: %d epochs', n_epochs)

    return model

def verify(model, input_tuples_df, input_data_params, training_params):
    """[summary]
    
    Arguments:
        model {A Keras Model object} -- The model to use for verification
        input_tuples_df {A pandas DataFrame} -- The input tuples.
        input_data_params {A InputDataParameters object} -- It contains the input parameters.
        training_params {A TrainingParameters object} -- It contains training parameters.
    """
    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                        input_data_params.dataset_location,
                        input_tuples_df, 
                        input_data_params.input_shape[:2],
                        training_params.batch_size,
                        input_data_params.num_classes,
                        input_data_params.image_cols,
                        input_data_params.label_col,
                        cache_size = training_params.image_cache_size,
                        randomize = False)

    #Training flow
    predict_gen = datagen.flow(subset = 'prediction')

    #Fit the model the input.
    predictions = model.predict_generator(
                            generator = predict_gen,
                            steps = training_params.num_prediction_steps)

    print('Input tuples:: ')
    print(input_tuples_df)
    print('Predictions:: ')
    print(predictions)

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    #Extract required parameters
    log_to_console = args.log_to_console

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)
    
    #Input data parameters
    input_data_params = InputDataParameters(args)
    logger.info('Input data parameters: %s', input_data_params)

    #Training parameters
    training_params = TrainingParameters(args)
    logger.info('Training parameters: %s', training_params)

    #Transformation parameters
    transformation_params = ImageDataTransformation.Parameters.parse(dict(args.transformations))
    logger.info('Transformation parameters: %s', transformation_params)

    #Dropbox parameters
    dropbox_parameters = args.dropbox_parameters
    dropbox_auth = None
    dropbox_dir = None

    if dropbox_parameters:
        dropbox_auth = dropbox_parameters[0]
        dropbox_dir = constants.DROPBOX_APP_PATH_PREFIX / dropbox_parameters[1]

    #Log input parameters 
    logger.info('Additional parameters log_to_console: %s dropbox_dir: %s', log_to_console, dropbox_dir)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)
    imgaug_seed(seed)

    #Prepare input data file
    input_files_client = InputFiles(dropbox_auth, dropbox_dir)
    input_data_file_path = input_files_client.get_all([input_data_params.input_data])[input_data_params.input_data]

    #Input data frame
    input_data_df = read_csv(input_data_file_path)
    num_df_sets = ceil(len(input_data_df) / input_data_params.input_data_training_set_size)
    num_classes = len(set(input_data_df[input_data_params.label_col]))

    #Update input data parameters
    input_data_params_update = dict(
                                    num_classes = num_classes,
                                    num_df_sets = num_df_sets)
    input_data_params.update(**input_data_params_update)

    logger.info('Input data parameters: %s', input_data_params)
 
    #Model input
    model_input = ModelInput(
                        input_data_params.model_name,
                        input_data_params.session_id,
                        input_data_params.input_data_training_set_id,
                        num_df_sets)

    #Add to the list of input files
    input_files = input_files_client.get_all([
                                                model_input.last_saved_file_name()
                                            ])
    
    #Model file
    model_file = input_files[model_input.last_saved_file_name()]
    model = load_model(str(model_file))

    logger.info("Loaded model from: {}".format(model_file))

    #Iterate over requested batches
    for set_id in range(input_data_params.input_data_training_set_id, num_df_sets + 1):
        #DataFrame start and end location for the iteration
        start_df_idx = (set_id - 1) * input_data_params.input_data_training_set_size
        end_df_idx = set_id * input_data_params.input_data_training_set_size

        #DataFrame set
        input_data_df_set = input_data_df.loc[start_df_idx:end_df_idx, :].reset_index(drop = True)

        #Log training metadata
        logger.info('Training tuples:: set_id: %d start_df_idx: %d end_df_idx: %d', set_id, start_df_idx, end_df_idx)

        #Train
        model = train(
                    model,
                    set_id,
                    input_data_df_set,
                    input_data_params,
                    training_params,
                    transformation_params,
                    dropbox_auth,
                    dropbox_dir,
                    1)

        #Verification
        verify(model, input_data_df_set, input_data_params, training_params)