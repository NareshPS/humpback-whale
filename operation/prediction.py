"""It computes predictions for the model.
"""
#Image data generator
from operation.image import ImageDataGeneration

#Pandas operations
from pandas import Series

#Numpy operations
from numpy import argmax, where

#Math operations
from math import ceil

#Logging
from common import logging

#Constants
from common import constants

class Prediction:
    def __init__(self, model, input_params, image_generation_params):
        """It sets up the input parameters.
        
        Arguments:
            model {A keras model object} -- The keras model object to use for making predictions.
            input_params {A InputParameters object} -- The input parameters.
            image_generation_params {A Path object} -- The parameters required for the image data generation.
        """
        #Input parameters
        self._model = model
        self._input_params = input_params
        self._image_generation_params = image_generation_params

        #Validation
        if self._model is None:
            raise ValueError('The model object must be valid')

        #Logging
        self._logger = logging.get_logger(__file__)

    def predict(self, input_data, num_prediction_steps):
        """[summary]
    
        Arguments:
            input_data {A pandas DataFrame} -- The input dataframe.
            num_prediction_steps {int} -- The number of prediction steps.
        """
        #Create a data generator to be used for fitting the model.
        datagen = ImageDataGeneration(
                                input_data,
                                self._input_params,
                                self._image_generation_params,
                                transformer = None,
                                randomize = False)

        #Training flow
        predict_gen = datagen.flow(subset = 'prediction')

        #Cap the prediction steps for small input datasets
        num_inputs = len(input_data)
        num_prediction_steps = min(num_prediction_steps, ceil(num_inputs / self._image_generation_params.batch_size))

        #Fit the model the input.
        predictions = self._model.predict_generator(
                                        generator = predict_gen,
                                        steps = num_prediction_steps)

        #Result dataframe
        num_predictions = predictions.shape[0]
        input_data_predicted_slice = input_data[:num_predictions].reset_index(drop = True)
        input_data_predicted_slice[constants.PANDAS_PREDICTION_COLUMN] = Series(argmax(predictions, axis = 1), index = input_data_predicted_slice.index)
        match_series = where(
                            input_data_predicted_slice[self._image_generation_params.label_col] == input_data_predicted_slice[constants.PANDAS_PREDICTION_COLUMN],
                            1, #True Value
                            0)
        input_data_predicted_slice[constants.PANDAS_MATCH_COLUMN] = match_series

        return input_data_predicted_slice