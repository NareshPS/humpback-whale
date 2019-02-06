#Unittest
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants
from common import constants

#Numpy
import numpy as np

#Transformation
from operation.image import ImageDataGeneration, ImageDataIterator

#Data manipulations
from pandas import DataFrame
from pandas import read_csv

columns = constants.INPUT_TUPLE_HEADERS
train_tuple_df = DataFrame(
                    [['0000e88ab.jpg', '0000e88ab.jpg', 1]],
                    columns = columns)
train_set_loc = ut_constants.TRAIN_STORE
input_tuples_file_name = 'input_tuples_p5_n5.tuples'

input_shape = constants.INPUT_SHAPE
image_cols = constants.INPUT_TUPLE_HEADERS[0:2]
label_col = constants.INPUT_TUPLE_LABEL_COL

class TestImageDataGeneration(ut.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageDataGeneration, self).__init__(*args, **kwargs)

    def flow_transformer(self, transformer, transform_x_cols):
        generator = ImageDataGeneration(
                        train_set_loc, train_tuple_df,
                        input_shape[:2], #Input shape
                        1, #Batch size
                        image_cols, label_col,
                        transform_x_cols = transform_x_cols,
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
        #Image generator
        generator = ImageDataGeneration(
                        train_set_loc, train_tuple_df,
                        input_shape[:2], #Input shape
                        1, #Batch size
                        image_cols, label_col,
                        transformer = transformer)

        return generator

    @mock_patch.object(ImageDataGeneration, '_get_image_objects')
    def test_fit(self, get_image_objects_mock):
        #Mock transformer
        transformer = ut.mock.Mock()

        #Override _get_image_objects return value
        image_name = train_tuple_df.loc[0, 'Anchor']
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
        image_name = train_tuple_df.loc[0, 'Anchor']
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
        image_name = train_tuple_df.loc[0, 'Anchor']
        generator._load_predict_phase_slice = MagicMock()
        generator._load_train_phase_slice = MagicMock()

        #Act
        iterator = generator.flow(subset = subset)
        values = iterator.__getitem__(0)

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