#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants
from common import constants

#Numpy
import numpy as np

#Transformation
from image.generation import ImageDataGeneration

#Data manipulations
from pandas import DataFrame

columns = constants.TRAIN_TUPLE_HEADERS
train_tuple_df = DataFrame(
                    [['0000e88ab.jpg', '0000e88ab.jpg', 1]],
                    columns = columns)
train_set_loc = ut_constants.TRAIN_STORE

input_shape = constants.INPUT_SHAPE
image_cols = constants.TRAIN_TUPLE_HEADERS[0:2]
label_col = constants.TRAIN_TUPLE_LABEL_COL

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

    @mock_patch.object(ImageDataGeneration, '_get_image_objects')
    def test_fit(self, get_image_objects_mock):
        #Mock transformer
        transformer = ut.mock.Mock()

        #Override _get_image_objects return value
        image_name = train_tuple_df.loc[0, 'Anchor']
        img_objs_map = TestImageDataGeneration.get_image_objects(image_name)
        get_image_objects_mock.return_value = img_objs_map

        #Image generator
        generator = ImageDataGeneration(
                        train_set_loc, train_tuple_df,
                        input_shape[:2], #Input shape
                        1, #Batch size
                        image_cols, label_col,
                        transformer = transformer)

        #Fit data
        generator.fit(n_images = 2)

        #Assert
        img_objs = np.reshape(img_objs_map[image_name], (1, ) + input_shape)
        transformer.fit.assert_called_once()
        np.testing.assert_array_equal(img_objs, transformer.fit.call_args[0][0])