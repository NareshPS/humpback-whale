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

#Keras
from keras.models import load_model

#Path manipulations
from pathlib import Path

#Basic parameters
model_file_name = 'cnn_model2d_1.h5'
input_shape = (200, 200, 3)
dimensions = 12
input_data_file = Path('input_data.csv')
input_data_path = ut_constants.DATA_STORE / input_data_file

def load_test_model():
    #Model file path
    model_file_path = ut_constants.DATA_STORE / model_file_name

    #Model object
    model = load_model(str(model_file_path))

    return model

def get_args(model_name, dataset_location):
    args = ArgumentParser()

    type(args).model_name = PropertyMock(return_value = model_name)
    type(args).dataset_location = PropertyMock(return_value = dataset_location)

    type(args).input_data = PropertyMock(return_value = 'input_data')
    type(args).image_cols = PropertyMock(return_value = ['Anchor', 'Sample'])
    type(args).image_transform_cols = PropertyMock(return_value = ['Anchor', 'Sample'])
    type(args).label_col = PropertyMock(return_value = 'Id')

    type(args).session_id = PropertyMock(return_value = 0)
    type(args).batch_size = PropertyMock(return_value = 32)
    type(args).image_cache_size = PropertyMock(return_value = 1024)

    type(args).validation_split = PropertyMock(return_value = 0.02)
    type(args).learning_rate = PropertyMock(return_value = 0.001)
    type(args).num_fit_images = PropertyMock(return_value = 20)
    type(args).number_of_epochs = PropertyMock(return_value = 1)

    type(args).num_prediction_steps = PropertyMock(return_value = 2)
    type(args).input_data_training_set_size = PropertyMock(return_value = 100)
    type(args).input_data_training_set_id = PropertyMock(return_value = 0)
    type(args).input_shape = PropertyMock(return_value = [224, 224, 3])

    return args

def get_input_df():
    return read_csv(str(input_data_path), index_col = 0)
