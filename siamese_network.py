#Keras imports
from keras.layers import Input, Dense, BatchNormalization, Activation, Concatenate
from keras.models import Model

#Load models from the disk
from keras.models import load_model

#Feature model
from resnet_feature_model import resnet_feature_model as feature_model

#Data processing
from image.generator import ImageDataGenerator
import numpy as np
import pandas as pd

#Load and save objects to disk
from pandas import read_csv

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed

#Constants
from common import constants

#Commandline arguments
from argparse import ArgumentParser

#Path manipulations
from os import path

def siamese_network_model(input_shape, feature_dims):
    anchor_input = Input(shape = input_shape, name = 'Anchor')
    sample_input = Input(shape = input_shape, name = 'Sample')

    anchor_features = feature_model(input_shape, feature_dims)(anchor_input)
    sample_features = feature_model(input_shape, feature_dims)(sample_input)

    X = Concatenate()([anchor_features, sample_features])
    X = Dense(16, activation = 'linear')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(4, activation = 'linear')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Dense(1, activation = 'sigmoid')(X)

    siamese_network = Model(inputs = [anchor_input, sample_input], outputs = [X], name = 'Siamese Model')
    siamese_network.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['mae', 'accuracy'])
    siamese_network.summary()

    return siamese_network
"""
def make_datagen(batch_size, validation_split = None):
    idg_kwargs = dict(
                    rescale = 1./255, shear_range = 0.2,
                    rotation_range=10, width_shift_range=0.2,
                    height_shift_range=0.2, zoom_range = 0.2, horizontal_flip= True)

    datagen = None

    if validation_split:
        datagen = ImageDataGenerator(validation_split = validation_split, **idg_kwargs)
    else:
        datagen = ImageDataGenerator(**idg_kwargs)

    return datagen

def make_generators(datagen, source, train_tuples_df, x_col, y_col, batch_size, validation_split = None):
    input_shape_nc = (224, 224)
    val_gen = None

    flow_kwargs = dict(
                    x_col = x_col, y_col = y_col, target_size = input_shape_nc,
                    batch_size = batch_size)

    if validation_split:
        val_gen = datagen.flow_from_dataframe(train_tuples_df, source, subset = 'validation', **flow_kwargs)

    train_gen = datagen.flow_from_dataframe(train_tuples_df, source, subset = 'training', **flow_kwargs)

    return train_gen, val_gen
    """

def parse_args():
    parser = ArgumentParser(description = 'It trains a siamese network for whale identification.')
    parser.add_argument(
        '-r', '--r_name',
        required = True,
        help = 'It names the run. The name is used to generate output names.')
    parser.add_argument(
        '-d', '--dataset',
        default='train', choices = constants.DATASET_NAMES,
        help = 'It specifies the dataset to use for training.')
    parser.add_argument(
        '-n', '--num_inputs',
        type = int,
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

    args = parser.parse_args()

    return args.r_name, args.dataset, args.num_inputs, args.epochs, args.batch_size, args.cache_size

if __name__ == "__main__":
    #Parse commandline arguments
    r_name, dataset, n_inputs, n_epochs, batch_size, cache_size = parse_args()

    #Predictable randomness
    seed = 3
    np_seed(3)
    tf_seed(3)
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

    output_model_file = r_name + ".h5"

    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGenerator(train_set_loc, train_tuples_df, input_shape[:2], batch_size, cache_size)

    #Create a model placeholder to create or load a model.
    model = None

    if path.exists(output_model_file):
        #Try to load the trained model from the disk.
        model = load_model(output_model_file)
    else:
        #Create siamese network model
        model = siamese_network_model(input_shape, feature_dims)

    #Calculate number of steps per epoch based on the input and the batch sizes.
    steps_per_epoch = int((len(train_tuples_df) + batch_size - 1)/batch_size)

    #Fit the model the input.
    model.fit_generator(
        datagen.flow(image_cols, output_col),
        steps_per_epoch = steps_per_epoch,
        epochs = n_epochs)

    #Save the trained model.
    model.save(output_model_file)
