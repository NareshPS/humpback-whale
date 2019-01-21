#Keras imports
from keras.layers import Input, Dense, BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

#Load models from the disk
from keras.models import load_model

#Load/Save model states
from model.state import ModelState

#Feature models
from model.feature_models import feature_model

#Data processing
from image.generation import ImageDataGeneration
from image.transformation import ImageDataTransformation
from imgaug import seed as imgaug_seed
import numpy as np
import pandas as pd

#Load and save objects to disk
from pandas import read_csv
from pickle import dump as pickle_dump

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed

#Constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from os import path

#Logging
from common import logging

top_layer_name_prefix = 'siamese_network'

def layer_name(layer_type, param):
    return "{}_{}_{}".format(top_layer_name_prefix, layer_type, param)

def siamese_network_model(base_model, input_shape, feature_dims, learning_rate):
    """It creates a siamese network model using the input as a base model.
    
    Arguments:
        base_model {A Model object.} -- A base model to generate feature vector.
        input_shape {(int, int, int))} -- A tuple to indicate the shape of inputs.
        feature_dims {int} -- An integer indicating the dimensions of the feature vector.
        learning_rate {float} -- A float value to control speed of learning.
    
    Returns:
        {A Model object} -- A keras model.
    """

    anchor_input = Input(shape = input_shape, name = 'Anchor')
    sample_input = Input(shape = input_shape, name = 'Sample')

    anchor_features = feature_model(base_model, input_shape, feature_dims)(anchor_input)
    sample_features = feature_model(base_model, input_shape, feature_dims)(sample_input)

    X = Concatenate()([anchor_features, sample_features])
    X = Dense(16, activation = 'linear', name = layer_name('dense', 16))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(4, activation = 'linear', name = layer_name('dense', 4))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(1, activation = 'sigmoid', name = layer_name('dense', 1))(X)
    
    #Create an optimizer object
    adam_optimizer = Adam(lr = learning_rate)

    siamese_network = Model(inputs = [anchor_input, sample_input], outputs = [X], name = 'Siamese Model')
    siamese_network.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics = ['accuracy'])
    siamese_network.summary()

    return siamese_network

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')
    parser.add_argument(
        '-m', '--base_model',
        required = True,
        help = 'It specifies a base model to use for the siamese network.')
    parser.add_argument(
        '-d', '--dataset',
        default='train',
        choices = constants.DATASET_NAMES,
        help = 'It specifies the dataset to use for training.')
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
        '-l', '--log_to_console',
        action = 'store_true', default = False,
        help = 'It enables logging to console')
    parser.add_argument(
        '-s', '--validation_split',
        type = float,
        help = 'It specifies the validation split on the training dataset. It must be a float between 0 and 1')
    parser.add_argument(
        '-r', '--learning_rate',
        default = 0.0001, type = float,
        help = 'It specifies the learning rate of the optimization algorithm. It must be a float between 0 and 1')
    parser.add_argument(
        '-t', '--transformations',
        nargs = '*', default = [],
        choices = ImageDataTransformation.Parameters().__dict__.keys(),
        help = 'It specifies transformation parameters')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    #Parse commandline arguments
    args = parse_args()

    #Extract command line parameters
    base_model, = args.base_model,
    dataset, = args.dataset, 
    n_inputs, = args.num_inputs, 
    n_epochs, = args.epochs, 
    batch_size, = args.batch_size, 
    cache_size, = args.cache_size, 
    log_to_console, = args.log_to_console,
    validation_split, = args.validation_split,
    learning_rate, = args.learning_rate,
    transformation_params = ImageDataTransformation.Parameters.parse(args.transformations)

    #Initialize logging
    logging.initialize(__file__, log_to_console = log_to_console)
    logger = logging.get_logger(__name__)

    #Log input parameters
    logger.info(
                'Running with parameters base_model: %s dataset: %s n_inputs: %s n_epochs: %d batch_size: %d cache_size: %d',
                base_model, 
                dataset, 
                n_inputs, 
                n_epochs, 
                batch_size, 
                cache_size)

    #Additional parameters
    logger.info(
                'Additional parameters log_to_console: %s validation_split: %s learning_rate: %s',
                log_to_console,
                validation_split,
                learning_rate)

    #Predictable randomness
    seed = 3
    np_seed(seed)
    tf_seed(seed)
    imgaug_seed(seed)

    #Required parameters
    df_image_col = constants.IMAGE_HEADER_NAME
    df_class_col = constants.LABEL_HEADER_NAME
    input_shape = constants.INPUT_SHAPE
    feature_dims = constants.FEATURE_VECTOR_DIMS
    train_set_loc = constants.DATASET_MAPPINGS[dataset]
    anchor_field = constants.TRAIN_TUPLE_HEADERS[0]
    sample_field = constants.TRAIN_TUPLE_HEADERS[1]
    similar_field = constants.TRAIN_TUPLE_HEADERS[3]

    train_tuples_df = read_csv(constants.DATASET_MAPPINGS['train_tuples'])
    train_tuples_df = train_tuples_df.loc[:(n_inputs - 1), :] if n_inputs else train_tuples_df

    image_cols = constants.TRAIN_TUPLE_HEADERS[0:2]
    output_col = constants.TRAIN_TUPLE_HEADERS[-1]

    model_file = base_model + ".h5"
    model_state_file = base_model + ".model_state"
    history_file = base_model + ".history"

    #Output files
    logger.info(
                'Output files:: model_file: %s model_state_file: %s history_file: %s',
                model_file,
                model_state_file,
                history_file)

    #Transformation parameters
    logger.info('Transformation parameters: %s', transformation_params)
    
    #Log training metadata
    logger.info("Training set size: {} image_cols: {} output_col: {}".format(len(train_tuples_df), image_cols, output_col))

    #Transformer
    transformer = ImageDataTransformation(parameters = transformation_params)

    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGeneration(
                    train_set_loc, 
                    train_tuples_df, 
                    input_shape[:2], 
                    batch_size,
                    validation_split = validation_split,
                    cache_size = cache_size,
                    transformer = transformer)

    train_gen = datagen.flow(
                            image_cols, 
                            output_col,
                            transform_x_cols = image_cols,
                            subset = 'training')

    validation_gen = datagen.flow(
                                image_cols,
                                output_col,
                                transform_x_cols = image_cols,
                                subset = 'validation') if validation_split else None

    #Create a model placeholder to create or load a model.
    model = None

    if path.exists(model_file):
        #Try to load the trained model from the disk.
        model = load_model(model_file)
        logger.info("Loaded model from: {}".format(model_file))

        #Update the learning rate
        logger.info("Switching leraning rate from: {} to: {}".format(K.get_value(model.optimizer.lr), learning_rate))
        K.set_value(model.optimizer.lr, learning_rate)
    else:
        #Create siamese network model
        model = siamese_network_model(base_model, input_shape, feature_dims, learning_rate)
        logger.info("Created a new model using base_model: {}".format(base_model))

        #Write model trainable state
        model_state = ModelState(model)
        model_state.save(".", model_state_file)
        logger.info("Writing model state to: {}".format(model_state_file))

    #Fit the model the input.
    history = model.fit_generator(
                    generator = train_gen,
                    validation_data = validation_gen,
                    epochs = n_epochs)

    logger.info('Training finished. Trained: %d epochs', n_epochs)

    #Save the trained model.
    model.save(model_file)

    logger.info('Wrote the model object: %s', model_file)

    #Save training history
    with open(history_file, 'wb') as handle:
        pickle_dump(history, handle)

    logger.info('Wrote the history file: %s', history_file)
