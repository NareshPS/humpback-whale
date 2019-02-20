#Unittest
import unittest as ut
from unittest.mock import MagicMock

#Constants
from common import ut_constants

#Test utils
from tests.utils import get_args

#Path parsing
from pathlib import Path

#Input parameters
from operation.input import InputParameters, TrainingParameters, ImageGenerationParameters, update_params

model_name = 'model_name'
dataset_location = Path()

num_classes = 5

class TestUpdateParams(ut.TestCase):
    def test_update_params(self):
        #Arrange
        args = get_args(model_name, dataset_location)
        image_generation_params = ImageGenerationParameters(args)
        additional_kwargs = dict(num_classes = num_classes)

        #Act
        update_params(image_generation_params, **additional_kwargs)

        #Assert
        self.assertEqual(num_classes, image_generation_params.num_classes)

class TestImageGenerationParameters(ut.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            #Arrange
            bad_dataset_location = dataset_location / 'non_existent_location'
            args = get_args(None, bad_dataset_location)

            #Act
            _ = ImageGenerationParameters(args)

        args = get_args(model_name, dataset_location)
        _ = ImageGenerationParameters(args)

class TestInputDataParameters(ut.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            args = get_args(None, None)
            _ = InputParameters(args)

        args = get_args(model_name, dataset_location)
        _ = InputParameters(args)

class TestTrainingParameters(ut.TestCase):
    def test_init(self):
        #Arrange
        args = get_args(None, None)
        
        #Act
        _ = TrainingParameters(args)