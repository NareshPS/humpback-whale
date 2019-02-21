#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch
from unittest.mock import MagicMock

#Test utils
from tests.support.utils import load_test_model, get_args, get_input_df

#Input imports
from operation.input import InputParameters, ImageGenerationParameters, update_params

#Prediction
from operation.prediction import Prediction

#Mocking operations
from operation.utils import imload
import numpy as np

#Path manipulations
from pathlib import Path

#Constants
from common import ut_constants
from common import constants

#Common parameters
model_name = 'model_name'
num_prediction_steps = 1

def patch_imload(source, images, shape = None):
    return np.random.random((len(images), 400, 700, 1))

class TestPrediction(ut.TestCase):
    def test_init_invalid_args(self):
        #Arrange
        args = get_args(model_name, Path())
        input_params = InputParameters(args)
        image_generation_params = ImageGenerationParameters(args)

        with self.assertRaises(ValueError):
            _ = Prediction(None, input_params, image_generation_params)

    def test_init_valid_args(self):
        #Arrange
        model = load_test_model()
        args = get_args(model_name, Path())
        input_params = InputParameters(args)
        image_generation_params = ImageGenerationParameters(args)
        input_data_df = get_input_df()

        #Update input data parameters
        num_classes = len(set(input_data_df[image_generation_params.label_col]))
        image_generation_params_update = dict(num_classes = num_classes, image_cols = ['Image'])
        update_params(image_generation_params, **image_generation_params_update)

        #Act
        _ = Prediction(model, input_params, image_generation_params)

    def test_predict(self):
        #Arrange
        model = load_test_model()
        args = get_args(model_name, Path())
        input_params = InputParameters(args)
        image_generation_params = ImageGenerationParameters(args)
        input_data_df = get_input_df()

        #Update input data parameters
        num_classes = len(set(input_data_df[image_generation_params.label_col]))
        image_generation_params_update = dict(num_classes = num_classes, image_cols = ['Image'])
        update_params(image_generation_params, **image_generation_params_update)

        #Mocks
        prediction_results = np.zeros((image_generation_params.batch_size * num_prediction_steps, num_classes))
        for row_id in range(image_generation_params.batch_size * num_prediction_steps):
            prediction_results[row_id, row_id % num_classes] = 1

        model.predict_generator = MagicMock()
        model.predict_generator.return_value = prediction_results

        with mock_patch('operation.utils.imload', side_effect = patch_imload): 
            #Act
            predictor = Prediction(model, input_params, image_generation_params)
            predictions = predictor.predict(input_data_df, num_prediction_steps)
            num_results = image_generation_params.batch_size * num_prediction_steps

            self.assertTrue(len(predictions), num_results)
            self.assertCountEqual(
                    input_data_df.loc[:num_results - 1, image_generation_params.image_cols[0]].values,
                    predictions.loc[:num_results, image_generation_params.image_cols[0]].values)
            self.assertCountEqual(
                    list(predictions),
                    [constants.PANDAS_PREDICTION_COLUMN, constants.PANDAS_MATCH_COLUMN, image_generation_params.image_cols[0], image_generation_params.label_col])
            self.assertCountEqual(
                    [0],
                    predictions[constants.PANDAS_MATCH_COLUMN].unique())