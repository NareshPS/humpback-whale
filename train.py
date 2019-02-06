#Keras imports
from keras import backend as K

#Load models from the disk
from keras.models import load_model

#Data processing
from operation.image import ImageDataGeneration
from operation.transform import ImageDataTransformation
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

#Constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser
from common.parse import kv_str_to_tuple

#Path manipulations
from pathlib import Path

#Logging
from common import logging

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')
    parser.add_argument(
        '-m', '--model_name',
        required = True,
        help = 'It specifies the name of the model to train.')
    parser.add_argument(
        '-d', '--dataset',
        default = 'train',
        choices = constants.DATASET_NAMES,
        help = 'It specifies the dataset to use for training.')
    parser.add_argument(
        '-i', '--input_tuples',
        required = True, type = Path,
        help = 'It specifies the path to the input tuples file.')
    parser.add_argument(
        '-n', '--num_inputs',
        type = int, nargs = '?',
        help = 'It specifies the number of inputs to use for training')
    parser.add_argument(
        '-e', '--epochs',
        default = 50, type = int,
        help = 'It specifies the number of training epochs.')
    parser.add_argument(
        '-b', '--batch_size',
        default = 32, type = int,
        help = 'It specifies the training batch size.')
    parser.add_argument(
        '-c', '--cache_size',
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

    args = parser.parse_args()

    return args

def train(model_name, dataset_loc, input_tuples_df,
            input_shape, transform_x_cols, validation_split,
            batch_size, cache_size, n_fit_images, learning_rate,
            dropbox_auth, dropbox_dir, n_epochs):
    """It executes the training.
    
    Arguments:
        model_name {string} -- The name of the model.
        dataset_loc {A Path object} -- The location of the image dataset.
        input_tuples_df {A pandas DataFrame} -- The input tuples.
        input_shape {(int, int)} -- HxW dimensions of the images.
        transform_x_cols {[string]} -- The list of column names containing the image names that will be transformed.
        validation_split {float} -- The fraction of the image set to be used for validation.
        batch_size {int} -- The number of images the data generator processes in each step.
        cache_size {int} -- The number of images to keep in the cache.
        n_fit_images {int} -- The number of images to use for computing transformation statistics.
        learning_rate {float} -- The learning rate to use for training the model.
        dropbox_auth {string} -- The authentication token to access dropbox store.
        dropbox_dir {string} -- The dropbox directory to store the generated data.
        n_epochs {int} -- The number of epochs to train the model.
    """
    #Logging
    logger = logging.get_logger(__name__)

    #Output files
    model_file = model_name + ".h5"
    history_file = model_name + ".history"

    #Input tuple columns
    image_cols = constants.INPUT_TUPLE_HEADERS[0:2]
    label_col = constants.INPUT_TUPLE_LABEL_COL

    #Output files
    logger.info(
                'Output files:: model_file: %s history_file: %s',
                model_file,
                history_file)

    #Transformer
    transformer = ImageDataTransformation(parameters = transformation_params)
    
    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                    dataset_loc, input_tuples_df, 
                    input_shape, batch_size,
                    image_cols, label_col,
                    transform_x_cols = image_cols,
                    validation_split = validation_split,
                    cache_size = cache_size,
                    transformer = transformer)

    #Fit the data generator
    datagen.fit(n_images = n_fit_images)

    #Training flow
    train_gen = datagen.flow(subset = 'training')

    #Validation flow
    validation_gen = datagen.flow(subset = 'validation') if validation_split else None

    #Load the model
    model = load_model(model_file)
    logger.info("Loaded model from: {}".format(model_file))

    if learning_rate:
        #Update the learning rate
        logger.info("Switching leraning rate from: {} to: {}".format(K.get_value(model.optimizer.lr), learning_rate))
        K.set_value(model.optimizer.lr, learning_rate)
        
    #Training callbacks
    dropbox_callback = ModelDropboxCheckpoint(
                                model_name,
                                dropbox_auth = dropbox_auth,
                                dropbox_dir = dropbox_dir)

    #Fit the model the input.
    history = model.fit_generator(
                        generator = train_gen,
                        validation_data = validation_gen,
                        epochs = n_epochs,
                        callbacks = [dropbox_callback])

    logger.info('Training finished. Trained: %d epochs', n_epochs)

    #Save training history
    with open(history_file, 'wb') as handle:
        pickle_dump(history, handle)

    logger.info('Wrote the history file: %s', history_file)

    return model

def verify(model,
            dataset_loc, input_tuples_df, 
            input_shape, batch_size, cache_size,
            num_prediction_steps):
    """[summary]
    
    Arguments:
        model_name {string} -- The name of the model.
        dataset_loc {A Path object} -- The location of the image dataset.
        input_tuples_df {A pandas DataFrame} -- The input tuples.
        input_shape {(int, int)} -- HxW dimensions of the images.
        batch_size {int} -- The number of images the data generator processes in each step.
        cache_size {int} -- The number of images to keep in the cache.
        num_prediction_steps {int} -- The number of prediction steps to use for verification.
    """
    #Input tuple columns
    image_cols = constants.INPUT_TUPLE_HEADERS[0:2]
    label_col = constants.INPUT_TUPLE_LABEL_COL

    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                    dataset_loc, input_tuples_df, 
                    input_shape, batch_size,
                    image_cols, label_col,
                    cache_size = cache_size,
                    randomize = False)

    #Training flow
    predict_gen = datagen.flow(subset = 'prediction')

    #Fit the model the input.
    predictions = model.predict_generator(
                            generator = predict_gen,
                            steps = num_prediction_steps)

    print('Input tuples:: ')
    print(input_tuples_df)
    print('Predictions:: ')
    print(predictions)

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    #Extract command line parameters
    model_name = args.model_name
    dataset = args.dataset
    n_inputs = args.num_inputs
    n_epochs = args.epochs
    batch_size = args.batch_size
    cache_size = args.cache_size
    log_to_console = args.log_to_console
    validation_split = args.validation_split
    learning_rate = args.learning_rate
    transformation_params = ImageDataTransformation.Parameters.parse(dict(args.transformations))
    n_fit_images = args.num_fit_images
    dropbox_parameters = args.dropbox_parameters
    input_tuples = args.input_tuples
    num_prediction_steps = args.num_prediction_steps

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters model_name: %s dataset: %s n_inputs: %s n_epochs: %d batch_size: %d cache_size: %d',
                model_name,
                dataset,
                n_inputs,
                n_epochs,
                batch_size,
                cache_size)

    #Additional parameters
    logger.info(
                'Additional parameters log_to_console: %s validation_split: %s learning_rate: %s input_tuples: %s',
                log_to_console,
                validation_split,
                learning_rate,
                input_tuples)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)
    imgaug_seed(seed)

    #Required parameters
    input_shape = constants.INPUT_SHAPE[:2]
    dataset_loc = constants.DATASET_MAPPINGS[dataset]

    #Validation
    if not input_tuples.exists():
        raise ValueError('The input tuples file: {} is not found'.format(input_tuples))

    input_tuples_df = read_csv(input_tuples)
    input_tuples_df = input_tuples_df.loc[:(n_inputs - 1), :] if n_inputs else input_tuples_df

    #Input tuple columns
    image_cols = constants.INPUT_TUPLE_HEADERS[0:2]
    label_col = constants.INPUT_TUPLE_LABEL_COL

    #Transformation parameters
    logger.info('Transformation parameters: %s', transformation_params)

    #Fit images parameter
    logger.info('Fit images:: n_fit_images: %d', n_fit_images)

    #Dropbox parameters
    dropbox_auth = None
    dropbox_dir = None

    if dropbox_parameters:
        dropbox_auth = dropbox_parameters[0]
        dropbox_dir = constants.DROPBOX_APP_PATH_PREFIX / dropbox_parameters[1]
    
    logger.info('Dropbox parameters:: dir: %s', dropbox_dir)

    #Log training metadata
    logger.info("Training set size: {} image_cols: {} label_col: {}".format(len(input_tuples_df), image_cols, label_col))

    #Train
    model = train(model_name, dataset_loc, input_tuples_df,
                    input_shape, image_cols, validation_split,
                    batch_size, cache_size, n_fit_images, learning_rate,
                    dropbox_auth, dropbox_dir, n_epochs)

    #Verification
    verify(model,
            dataset_loc, input_tuples_df, 
            input_shape, batch_size, cache_size,
            num_prediction_steps)