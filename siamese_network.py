#Keras imports
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, BatchNormalization, Activation, Concatenate
from keras.models import Model

#Feature model
from resnet_feature_model import resnet_feature_model as feature_model

#Data processing
import numpy as np
import pandas as pd

#Load and save objects to disk
from pandas import read_csv

#Allow reproducible results
from numpy.random import seed as np_seed
from tensorflow import set_random_seed as tf_seed

#Constants
from common import constants

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

if __name__ == "__main__":
    #Predictable randomness
    seed = 3
    np_seed(3)
    tf_seed(3)
    df_image_col = constants.IMAGE_HEADER_NAME
    df_class_col = constants.LABEL_HEADER_NAME
    input_shape = constants.INPUT_SHAPE
    feature_dims = constants.FEATURE_VECTOR_DIMS
    train_set_loc = constants.RAW_DATASET_MAPPINGS['train']
    anchor_field = constants.TRAIN_TUPLE_HEADERS[0]
    sample_field = constants.TRAIN_TUPLE_HEADERS[1]
    similar_field = constants.TRAIN_TUPLE_HEADERS[3]

    batch_size = 32
    n_epochs = 1

    train_tuples_df = read_csv(constants.PROCESSED_DATASET_MAPPINGS['train_tuples'])

    from image.generator import ImageDataGenerator

    image_cols = constants.TRAIN_TUPLE_HEADERS[0:2]
    output_col = constants.TRAIN_TUPLE_HEADERS[-1]

    #Create a data generator to be used for fitting the model.
    datagen = ImageDataGenerator(train_set_loc, train_tuples_df, input_shape[:2], batch_size)

    #Create siamese network model
    model = siamese_network_model(input_shape, feature_dims)

    #Calculate number of steps per epoch based on the input and the batch sizes.
    steps_per_epoch = int((len(train_tuples_df) + batch_size - 1)/batch_size)

    #Fit the model the input.
    model.fit_generator(
        datagen.flow(image_cols, output_col),
        steps_per_epoch = steps_per_epoch,
        epochs = n_epochs)
