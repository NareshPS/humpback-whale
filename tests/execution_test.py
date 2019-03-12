#Unittests
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Parallel or sequential execution
from common.execution import execute

#Common parameters
num_items = 5

def stub(iter_item):
    return iter_item

class TestExecution(ut.TestCase):
    def execute(self, parallel):
        #Arrange
        iterator = range(num_items)

        #Act
        results = execute(stub, iterator, num_items, parallel)

        #Assert
        self.assertEqual(num_items, len(results))
        self.assertCountEqual(list(range(num_items)), results)

    def test_execute_serial(self):
        #Arrange
        parallel = False

        #Act
        self.execute(parallel)

    def test_execute_parallel(self):
        #Arrange
        parallel = True

        #Act
        self.execute(parallel)