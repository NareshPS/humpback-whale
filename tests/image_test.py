#Unittest
import unittest as ut
from unittest.mock import MagicMock, PropertyMock
from unittest.mock import patch as mock_patch

#Argument parsing
from argparse import ArgumentParser

#Constants
from common import ut_constants
from common import constants

#Numpy
import numpy as np

#Image data generation
from operation.image import ImageDataGeneration, ImageDataIterator
from operation.input import InputParameters, TrainingParameters, ImageGenerationParameters

#Data manipulations
from pandas import DataFrame
from pandas import read_csv

columns = ['Anchor', 'Sample', 'Label']
input_data_df = DataFrame(
                    [['0000e88ab.jpg', '0000e88ab.jpg', 1]],
                    columns = columns)
train_set_loc = ut_constants.TRAIN_STORE
input_tuples_file_name = 'input_tuples_p5_n5.tuples'
model_name = 'model_name'

input_shape = (224, 224, 3)
image_cols = columns[:2]
label_col = columns[-1]

def get_args():
    args = ArgumentParser()

    type(args).model_name = PropertyMock(return_value = model_name)
    type(args).dataset_location = PropertyMock(return_value = train_set_loc)

    type(args).input_data = PropertyMock(return_value = input_data_df)
    type(args).image_cols = PropertyMock(return_value = columns[:2])
    type(args).image_transform_cols = PropertyMock(return_value = columns[:2])
    type(args).label_col = PropertyMock(return_value = columns[-1])

    type(args).session_id = PropertyMock(return_value = 0)
    type(args).batch_size = PropertyMock(return_value = 1)
    type(args).image_cache_size = PropertyMock(return_value = 1024)
    type(args).number_of_epochs = PropertyMock(return_value = 1)

    type(args).validation_split = PropertyMock(return_value = 0.02)
    type(args).learning_rate = PropertyMock(return_value = 0.001)
    type(args).num_fit_images = PropertyMock(return_value = 20)

    type(args).num_prediction_steps = PropertyMock(return_value = 2)
    type(args).input_data_training_set_size = PropertyMock(return_value = 100)
    type(args).input_data_training_set_id = PropertyMock(return_value = 0)
    type(args).input_shape = PropertyMock(return_value = input_shape)

    return args

def get_input_and_image_generation_params():
    args = get_args()
    input_data_params = InputParameters(args)
    image_generation_params = ImageGenerationParameters(args)

    return input_data_params, image_generation_params

class TestImageDataGeneration(ut.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageDataGeneration, self).__init__(*args, **kwargs)

    def test_init(self):
        #Arrange
        input_data_params, image_generation_params = get_input_and_image_generation_params()

        #With None transform columns
        _ = ImageDataGeneration(
                        input_data_df,
                        input_data_params,
                        image_generation_params, 
                        transformer = None)

    def flow_transformer(self, transformer, image_transform_cols):
        #Arrange
        input_data_params, image_generation_params = get_input_and_image_generation_params()
        image_generation_params.image_transform_cols = image_transform_cols

        generator = ImageDataGeneration(
                        input_data_df,
                        input_data_params,
                        image_generation_params, 
                        transformer = transformer)

        iterator = generator.flow()

        _ = iterator.__getitem__(0)

    def test_flow_transformer_called(self):
        #Create a mock transformer
        image_data_transformation = ut.mock.Mock()

        #Action
        self.flow_transformer(image_data_transformation, [image_cols[0]])

        #Assert
        image_data_transformation.transform.assert_called_once()

    def test_flow_transformer_not_called(self):
        #Create a mock transformer
        image_data_transformation = ut.mock.Mock()

        #Action
        self.flow_transformer(image_data_transformation, [])

        #Assert
        image_data_transformation.transform.assert_not_called()

    @classmethod
    def get_image_objects(cls, image_name):
        #Initial image object
        image = np.ones(input_shape)

        return {image_name: image}

    def get_generator(self, transformer = None):
        #Arrange
        input_data_params, image_generation_params = get_input_and_image_generation_params()

        #Image generator
        generator = ImageDataGeneration(
                            input_data_df,
                            input_data_params,
                            image_generation_params,
                            transformer = transformer)

        return generator

    @mock_patch.object(ImageDataGeneration, '_get_image_objects')
    def test_fit(self, get_image_objects_mock):
        #Mock transformer
        transformer = ut.mock.Mock()

        #Override _get_image_objects return value
        image_name = input_data_df.loc[0, 'Anchor']
        img_objs_map = TestImageDataGeneration.get_image_objects(image_name)
        get_image_objects_mock.return_value = img_objs_map

        #Image generator
        generator = self.get_generator(transformer)

        #Fit data
        generator.fit(n_images = 2)

        #Assert
        img_objs = np.reshape(img_objs_map[image_name], (1, ) + input_shape)
        transformer.fit.assert_called_once()
        np.testing.assert_array_equal(img_objs, transformer.fit.call_args[0][0])

    def get_input_tuple_df(self):
        input_tuples_file_path = ut_constants.TRAIN_STORE / input_tuples_file_name

        return read_csv(input_tuples_file_path)

    def validate_fit_output(self, subset, is_tuple):
        #Arrange
        generator = self.get_generator()

        #Override _get_image_objects return value
        image_name = input_data_df.loc[0, 'Anchor']
        img_objs_map = TestImageDataGeneration.get_image_objects(image_name)
        generator._get_image_objects = MagicMock()
        generator._get_image_objects.return_value = img_objs_map

        #Act
        iterator = generator.flow(subset = subset)
        values = iterator.__getitem__(0)

        #Assert
        if is_tuple:
            self.assertTrue(type(values) is tuple)
            self.assertEqual(2, len(values))
        else:
            self.assertFalse(type(values) is tuple)

    def test_fit_training_output(self):
        self.validate_fit_output('training', True)

    def test_fit_prediction_output(self):
        self.validate_fit_output('prediction', False)

    def validate_fit_calls(self, subset, calls_prediction):
        #Arrange
        generator = self.get_generator()

        #Override _get_image_objects return value
        generator._load_predict_phase_slice = MagicMock()
        generator._load_train_phase_slice = MagicMock()

        #Act
        iterator = generator.flow(subset = subset)
        _ = iterator.__getitem__(0)

        #Assert
        if calls_prediction:
            generator._load_train_phase_slice.assert_not_called()
            generator._load_predict_phase_slice.assert_called_once()
        else:
            generator._load_train_phase_slice.assert_called_once()
            generator._load_predict_phase_slice.assert_not_called()

    def test_fit_training_call(self):
        self.validate_fit_calls('training', False)

    def test_fit_prediction_call(self):
        self.validate_fit_calls('prediction', True)