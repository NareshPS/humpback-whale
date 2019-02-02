"""It contains utility functions for the tests.
"""
#Constants
from common import ut_constants

#Keras
from keras.models import load_model

#Path manipulations
from os import path

#Basic parameters
model_file_name = 'cnn_model2d_1.h5'
input_shape = (200, 200, 3)
dimensions = 12

def load_test_model():
    #Model file path
    model_file_path = path.join(ut_constants.UT_DATA_STORE, model_file_name)

    #Model object
    model = load_model(model_file_path)

    return model
