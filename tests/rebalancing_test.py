#Unittest
import unittest as ut
from unittest.mock import MagicMock

#Constants
from common import constants
from common import ut_constants

#Test support
from tests.support.utils import image_col, label_col, columns
from tests.support.utils import get_input_data, create_dataframe

#Local Pandas operations
from common.pandas import min_freq, unique_items

#Rebalancing
from operation.rebalancing import Rebalancing

class TestRebalancing(ut.TestCase):
    def test_init(self):
        #Arrange
        data = get_input_data()

        #Act
        _ = Rebalancing(data, label_col)

    def test_rebalance(self):
        #Arrange
        data = create_dataframe()
        rebalancer = Rebalancing(data, label_col)
        min_freq_value = min_freq(data, label_col)
        label_values = unique_items(data, label_col)

        #Act
        result, _, _ = rebalancer.rebalance()

        #Assert
        self.assertCountEqual(label_values, unique_items(result, label_col))
        self.assertEqual(len(label_values) * min_freq_value, len(result))

    def test_rebalance_without_statistics(self):
        #Arrange
        data = create_dataframe()
        rebalancer = Rebalancing(data, label_col)

        #Act
        _, pre_stats, post_stats = rebalancer.rebalance()

        #Assert
        self.assertIsNone(pre_stats)
        self.assertIsNone(post_stats)

    def test_rebalance_with_statistics(self):
        #Arrange
        data = create_dataframe()
        rebalancer = Rebalancing(data, label_col)

        #Act
        _, pre_stats, post_stats = rebalancer.rebalance(statistics = True)

        #Assert
        self.assertIsNotNone(pre_stats)
        self.assertIsNotNone(post_stats)