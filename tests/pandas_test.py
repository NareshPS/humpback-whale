#Unittest
import unittest as ut
from unittest.mock import MagicMock

#Constants
from common import ut_constants
from common import constants

#Local pandas wrappers
from common.pandas import csv_to_dataframe, dataframe_to_csv
from common.pandas import unique_items, count_items, sum_items, min_freq, random_choice, randomize

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
