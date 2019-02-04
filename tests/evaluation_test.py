#Unittest
import unittest as ut

#Constants
from common import constants
from common import ut_constants

#Path manipulation
from pathlib import Path

#Load CSVs
from pandas import read_csv

#Tuple evaluation
from siamese.evaluation import TupleEvaluation

#Logging
from common import logging

class TestTupleEvaluation(ut.TestCase):
    @classmethod
    def get_tuples(cls):
        #Tuples file name
        input_tuples_file_name = 'input_tuples_p5_n5.tuples'

        #Tuple file path
        input_tuples_file_path = ut_constants.DATA_STORE / input_tuples_file_name

        return read_csv(input_tuples_file_path)

    def test_init(self):
        with self.assertRaises(ValueError):
            _ = TupleEvaluation(None)

        #Label  DataFrame
        input_tuples_df = TestTupleEvaluation.get_tuples()

        _ = TupleEvaluation(input_tuples_df)

    def test_evaluate(self):
        #Label  DataFrame
        input_tuples_df = TestTupleEvaluation.get_tuples()

        #Tuple evaluation object
        tuple_evaluation = TupleEvaluation(input_tuples_df)
        label_pcts = tuple_evaluation.evaluate()

        #Assert
        self.assertAlmostEqual(label_pcts[0], 66.84491, places = 2)
        self.assertAlmostEqual(label_pcts[1], 33.15508, places = 2)