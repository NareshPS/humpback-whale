"""It contains utility functions for the tests.
"""
#Constants
from common import ut_constants

#Unittest
from unittest.mock import PropertyMock

#Argument parsing
from argparse import ArgumentParser

#Dataframe operations
from pandas import read_csv
from pandas import DataFrame
import numpy as np

#Image loading
from operation.utils import imload

#Input parameters
from operation.input import TrainingParameters

#Keras
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Flatten

#Path manipulations
from pathlib import Path

#Basic parameters
model_file_name = 'cnn_model2d_1.h5'
input_shape = (200, 200, 3)
dimensions = 12
input_data_file = Path('input_data.csv')
input_data_path = ut_constants.DATA_STORE / input_data_file
dataset_location = ut_constants.TRAIN_STORE
input_tuples_file_name = 'input_tuples_p5_n5.tuples'
model_name = 'cnn_model2d_1'
image_cols = ['Anchor', 'Sample']

#Dataframe parameters
image_col = 'Image'
label_col = 'Id'
columns = [image_col, label_col]

#Image list
image_names = ['0000e88ab.jpg', '3889d6902.jpg']

def load_test_model():
    #Model file path
    model_file_path = ut_constants.DATA_STORE / model_file_name

    #Model object
    model = load_model(str(model_file_path))

    return model

def get_args(model_name = model_name, dataset_location = dataset_location, image_cols = image_cols):
    args = ArgumentParser()

    type(args).model_name = PropertyMock(return_value = model_name)
    type(args).dataset_location = PropertyMock(return_value = dataset_location)

    type(args).input_data = PropertyMock(return_value = 'input_data')
    type(args).image_cols = PropertyMock(return_value = image_cols)
    type(args).image_transform_cols = PropertyMock(return_value = image_cols)
    type(args).label_col = PropertyMock(return_value = 'Id')

    type(args).batch_size = PropertyMock(return_value = 32)
    type(args).image_cache_size = PropertyMock(return_value = 1024)

    type(args).validation_split = PropertyMock(return_value = 0.02)
    type(args).learning_rate = PropertyMock(return_value = 0.001)
    type(args).num_fit_images = PropertyMock(return_value = 20)
    type(args).number_of_epochs = PropertyMock(return_value = 1)
    type(args).checkpoint_batch_interval = PropertyMock(return_value = 1)

    type(args).epoch_id = PropertyMock(return_value = 0)
    type(args).batch_id = PropertyMock(return_value = 0)
    type(args).number_prediction_steps = PropertyMock(return_value = 2)
    type(args).input_shape = PropertyMock(return_value = [224, 224, 3])

    type(args).dropbox_parameters = PropertyMock(return_value = ['auth_token', 'dropbox_dir'])

    return args

def get_train_params():
    args = get_args()
    train_params = TrainingParameters(args)

    return train_params

def get_input_data():
    return read_csv(str(input_data_path), index_col = 0)

def create_dataframe_from_dict(data_dict):
    columns = list(data_dict.keys())

    #Prepare the first column values of the data frame
    results = data_dict[columns[0]]

    for column_name in columns[1:]:
        results = list(zip(results, data_dict[column_name]))

    dataframe = DataFrame(results, columns = columns)

    return dataframe

def create_dataframe():
    image_values = map(lambda x : 'image{}.jpg'.format(x), range(10))
    label_meta_values = [[0]*2, [1]*3, [2]*4]
    label_values = [item for sublist in label_meta_values for item in sublist]

    data = create_dataframe_from_dict({
                                            image_col : image_values,
                                            label_col : label_values
                                        })

    return data

def patch_imload(source, images, shape = None):
    return np.random.random((len(images), 400, 700, 1))

def create_model(input_shape = input_shape, num_classes = 2, metrics = None):
    model = Sequential()

    model.add(Dense(100, input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.compile('adam', loss = 'binary_crossentropy', metrics = metrics)

    return model

def create_model_input_data(input_shape):
    X = imload(dataset_location, image_names, shape = input_shape[:2])
    Y = np.ones((len(image_names), 1))

    return X, Y
