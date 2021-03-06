#Unittest
import unittest as ut
from unittest.mock import MagicMock

#Constants
from common import ut_constants
from common import constants

#Local pandas wrappers
from common.pandas import *

#Test support
from tests.support.utils import image_col, label_col, columns
from tests.support.utils import create_dataframe

input_data_filepath = ut_constants.DATA_STORE / 'input_data.csv'

class TestPandas(ut.TestCase):
    def test_csv_to_dataframe(self):
        #Act
        input_data = csv_to_dataframe(input_data_filepath)

        #Assert
        self.assertCountEqual(list(input_data), ['Image', 'Id'])

    def test_dataframe_to_csv(self):
        #Arrange
        input_data = csv_to_dataframe(input_data_filepath)
        input_data.to_csv = MagicMock()

        #Act
        dataframe_to_csv(input_data, input_data_filepath)

        #Assert
        input_data.to_csv.assert_called_once()

    def test_unique_items(self):
        #Arrange
        input_data = csv_to_dataframe(input_data_filepath)

        #Act
        values = unique_items(input_data, label_col)

        #Assert
        self.assertCountEqual(values, list(range(64)))

    def test_count_items(self):
        #Arrange
        input_data = csv_to_dataframe(input_data_filepath)

        #Act
        counts = count_items(input_data, label_col)

        #Assert
        self.assertEqual(len(input_data), sum_items(counts, constants.PANDAS_COUNT_AGG_COLUMN))

    def test_sum_items(self):
        #Arrange
        input_data = csv_to_dataframe(input_data_filepath)

        #Act
        value = sum_items(input_data, label_col)

        #Assert
        self.assertEqual(value, 3168)

    def test_min_freq(self):
        #Arrange
        input_data = csv_to_dataframe(input_data_filepath)

        #Act
        value = min_freq(input_data, label_col)

        #Assert
        self.assertEqual(1, value)

    def test_random_choice(self):
        #Arrange
        data = create_dataframe()
        unique_labels = unique_items(data, label_col)

        #Act
        result = random_choice(data, label_col, 1)

        #Assert
        self.assertEqual(len(unique_labels), len(result))

    def test_randomize(self):
        #Arrange
        data = create_dataframe()

        #Act
        result = randomize(data)

        #Assert
        self.assertEqual(len(data), len(result))
        self.assertCountEqual(unique_items(data, image_col), unique_items(result, image_col))

    def test_remove(self):
        #Arrange
        data = create_dataframe()
        value = 0
        value_count = len(data[data[label_col] == value])

        #Act
        result = remove(data, label_col, [value])

        #Assert
        self.assertEqual(len(result), len(data) - value_count)
        self.assertTrue(value not in unique_items(result, label_col))

    def test_group_aggregate_as_list(self):
        #Arrange
        candidate_label = 0
        data = create_dataframe()
        unique_labels = unique_items(data, label_col)
        candidate_label_images = data[data[label_col] == candidate_label][image_col].values

        #Act
        grouped = group_aggregate_as_list(data, label_col, image_col)

        #Assert
        self.assertCountEqual(unique_labels, grouped[label_col].values)
        self.assertCountEqual(candidate_label_images, grouped[grouped[label_col] == 0][image_col].values[0])

    def test_to_dict(self):
        #Arrange
        candidate_label = 0
        data = create_dataframe()
        unique_labels = unique_items(data, label_col)
        candidate_label_images = data[data[label_col] == candidate_label][image_col].values

        #Act
        labelled_images = to_dict(data, label_col, image_col)

        #Assert
        self.assertCountEqual(unique_labels, list(labelled_images.keys()))
        self.assertCountEqual(candidate_label_images, labelled_images[candidate_label])
