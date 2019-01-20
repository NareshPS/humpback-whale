#Unittest
import unittest as ut

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
                    [['0000e88ab.jpg', '0000e88ab.jpg', 'xyz', 1]],
                    columns = columns)
train_set_loc = ut_constants.UT_TRAIN_STORE

input_shape = constants.INPUT_SHAPE
image_cols = constants.TRAIN_TUPLE_HEADERS[0:2]
output_col = constants.TRAIN_TUPLE_HEADERS[-1]

class TestImageDataGeneration(ut.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageDataGeneration, self).__init__(*args, **kwargs)

    def flow_transformer(self, transformer, transform_x_cols):
        generator = ImageDataGeneration(
                        train_set_loc,
                        train_tuple_df,
                        target_shape = input_shape[:2],
                        batch_size = 1,
                        transformer = transformer)

        iterator = generator.flow(
                                image_cols,
                                output_col,
                                transform_x_cols = transform_x_cols)

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