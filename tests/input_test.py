#Unittest
import unittest as ut
from unittest.mock import MagicMock, PropertyMock

#Constants
from common import ut_constants

#Argument parsing
from argparse import ArgumentParser

#Path parsing
from pathlib import Path

#Input parameters
from operation.input import InputDataParameters

model_name = 'model_name'
dataset_location = Path()

class TestInputDataParameters(ut.TestCase):
    @classmethod
    def get_args(cls, model_name, dataset_location):
        args = ArgumentParser()

        type(args).model_name = PropertyMock(return_value = model_name)
        type(args).dataset_location = PropertyMock(return_value = dataset_location)

        type(args).input_data = PropertyMock(return_value = 'input_data')
        type(args).image_cols = PropertyMock(return_value = ['Anchor', 'Sample'])
        type(args).image_transform_cols = PropertyMock(return_value = ['Anchor', 'Sample'])
        type(args).label_col = PropertyMock(return_value = 'label_col')

        type(args).session_id = PropertyMock(return_value = 0)
        type(args).batch_size = PropertyMock(return_value = 32)
        type(args).cache_size = PropertyMock(return_value = 1024)

        type(args).validation_split = PropertyMock(return_value = 0.02)
        type(args).learning_rate = PropertyMock(return_value = 0.001)
        type(args).num_fit_images = PropertyMock(return_value = 20)

        type(args).num_prediction_steps = PropertyMock(return_value = 2)
        type(args).input_data_training_set_size = PropertyMock(return_value = 100)
        type(args).input_data_training_set_id = PropertyMock(return_value = 0)
        type(args).input_shape = PropertyMock(return_value = [224, 224, 3])

        return args

    def test_init(self):
        with self.assertRaises(ValueError):
            args = TestInputDataParameters.get_args(None, dataset_location)
            _ = InputDataParameters(args)

        with self.assertRaises(ValueError):
            args = TestInputDataParameters.get_args(model_name, None)
            _ = InputDataParameters(args)

        args = TestInputDataParameters.get_args(model_name, dataset_location)
        _ = InputDataParameters(args)