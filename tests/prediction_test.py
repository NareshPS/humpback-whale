#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch
from unittest.mock import MagicMock

#Test utils
from tests.support.utils import load_test_model, get_args, get_input_data, patch_imload

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

class TestPrediction(ut.TestCase):
    def test_init_invalid_args(self):
        #Arrange
        args = get_args(model_name, Path())
        input_params = InputParameters(args)
        image_generation_params = ImageGenerationParameters(args)

        with self.assertRaises(ValueError):
            _ = Prediction(None, input_params, image_generation_params)

    @classmethod
    def get_inputs(cls, num_prediction_steps, num_results = None):
        #Arrange
        model = load_test_model()
        args = get_args(model_name, Path())
        input_params = InputParameters(args)
        image_generation_params = ImageGenerationParameters(args)
        input_data = get_input_data()

        #Update input data parameters
        num_classes = len(set(input_data[image_generation_params.label_col]))
        image_generation_params_update = dict(num_classes = num_classes, image_cols = ['Image'])
        update_params(image_generation_params, **image_generation_params_update)

        #Num results
        num_results = num_results or image_generation_params.batch_size * num_prediction_steps

        #Mocks
        prediction_results = np.zeros((num_results, num_classes))
        for row_id in range(num_results):
            prediction_results[row_id, row_id % num_classes] = 1

        model.predict_generator = MagicMock()
        model.predict_generator.return_value = prediction_results

        return model, input_data, input_params, image_generation_params, prediction_results

    def test_predict(self):
        #Arrange
        num_prediction_steps = 1
        model, input_data, input_params, image_generation_params, prediction_results = TestPrediction.get_inputs(num_prediction_steps)

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            predictor = Prediction(model, input_params, image_generation_params)
            predictions = predictor.predict(input_data, num_prediction_steps)
            num_results = image_generation_params.batch_size * num_prediction_steps

            self.assertTrue(len(predictions), num_results)
            self.assertCountEqual(
                    input_data.loc[:num_results - 1, image_generation_params.image_cols[0]].values,
                    predictions.loc[:num_results, image_generation_params.image_cols[0]].values)
            self.assertCountEqual(
                    list(predictions),
                    [constants.PANDAS_PREDICTION_COLUMN, constants.PANDAS_MATCH_COLUMN, image_generation_params.image_cols[0], image_generation_params.label_col])
            self.assertCountEqual(
                    [0],
                    predictions[constants.PANDAS_MATCH_COLUMN].unique())

    def test_predict_prediction_with_reduced_num_prediction_steps(self):
        #Arrange
        num_prediction_steps = 2
        num_results = 5
        model, input_data, input_params, image_generation_params, _ = TestPrediction.get_inputs(num_prediction_steps, num_results)

        with mock_patch('operation.utils.imload', side_effect = patch_imload): 
            #Act
            predictor = Prediction(model, input_params, image_generation_params)
            predictions = predictor.predict(input_data[:num_results], num_prediction_steps)

            #Assert
            model.predict_generator.assert_called_once()
            _, args = model.predict_generator.call_args_list[0]
            self.assertEqual(
                    1, #Modified prediction steps
                    args['steps'])

        