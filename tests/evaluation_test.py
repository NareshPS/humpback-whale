#Unittest
import unittest as ut

#Constants
from common import constants
from common import ut_constants

#Path manipulation
from pathlib import Path

#Load CSVs
from pandas import read_csv

#Performance evaluation
from model.evaluation import LabelEvaluation
from model.evaluation import ModelEvaluation

#Image data generator
from operation.image import ImageDataGeneration

#Load models from the disk
from keras.models import load_model

#Logging
from common import logging

#Common parameters
label_col = 'Label'
number_of_bins = 5

def get_input_data():
    #Tuples file name
    input_data_file_name = 'input_tuples_p5_n5.tuples'

    #Tuple file path
    input_data_file_path = ut_constants.DATA_STORE / input_data_file_name

    return read_csv(input_data_file_path)

def get_model():
    #Model file name
    model_file_name = 'cnn_model2d_1.h5'

    #Model file path
    model_file_path = ut_constants.DATA_STORE / model_file_name

    return load_model(str(model_file_path))

class TestLabelEvaluation(ut.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            _ = LabelEvaluation(None)

        #Label  DataFrame
        input_data_df = get_input_data()

        _ = LabelEvaluation(input_data_df)

    def test_distribution(self):
        #Label  DataFrame
        input_data_df = get_input_data()

        #Tuple evaluation object
        label_evaluation = LabelEvaluation(input_data_df)
        label_pcts = label_evaluation.distribution(label_col)

        #Assert columns
        self.assertCountEqual(list(label_pcts), [label_col, constants.PANDAS_COUNT_AGG_COLUMN, constants.PANDAS_PCT_AGG_COLUMN])
        
        #Assert percentages
        self.assertAlmostEqual(label_pcts.loc[0, constants.PANDAS_PCT_AGG_COLUMN], 66.84491, places = 2)
        self.assertAlmostEqual(label_pcts.loc[1, constants.PANDAS_PCT_AGG_COLUMN], 33.15508, places = 2)

        #Assert counts
        self.assertEqual(label_pcts.loc[0, constants.PANDAS_COUNT_AGG_COLUMN], 500)
        self.assertEqual(label_pcts.loc[1, constants.PANDAS_COUNT_AGG_COLUMN], 248)

    def test_bin(self):
        #Label  DataFrame
        input_data_df = get_input_data()

        #Label evaluation object
        label_evaluation = LabelEvaluation(input_data_df)
        label_bins = label_evaluation.bin(label_col, number_of_bins)

        #Assert columns
        self.assertCountEqual(list(label_bins), [constants.PANDAS_COUNT_AGG_COLUMN, constants.PANDAS_COUNT_BIN_COLUMN])
        self.assertEqual(number_of_bins, len(label_bins))
        self.assertEqual(label_bins.loc[0, constants.PANDAS_COUNT_BIN_COLUMN], 0)
        self.assertEqual(label_bins.loc[4, constants.PANDAS_COUNT_BIN_COLUMN], 500)

    def test_histogram(self):
        #Label  DataFrame
        input_data_df = get_input_data()

        #Tuple evaluation object
        label_evaluation = LabelEvaluation(input_data_df)
        label_histogram = label_evaluation.histogram(label_col)

        #Assert columns
        self.assertCountEqual(list(label_histogram), [constants.PANDAS_COUNT_AGG_COLUMN, constants.PANDAS_COUNT_HIST_COLUMN])

        #Assert counts
        self.assertEqual(label_histogram.loc[0, constants.PANDAS_COUNT_AGG_COLUMN], 248)
        self.assertEqual(label_histogram.loc[1, constants.PANDAS_COUNT_AGG_COLUMN], 500)

        #Assert histogram
        self.assertEqual(label_histogram.loc[0, constants.PANDAS_COUNT_HIST_COLUMN], 1)
        self.assertEqual(label_histogram.loc[1, constants.PANDAS_COUNT_HIST_COLUMN], 1)

class TestModelEvaluation(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        #Arrange
        cls.input_data = get_input_data()
        cls.model = get_model()

    def test_init(self):
        with self.assertRaises(ValueError):
            #Input tuples
            _ = ModelEvaluation(None, TestModelEvaluation.model)

        with self.assertRaises(ValueError):
            #Model
            _ = ModelEvaluation(TestModelEvaluation.input_data, None)

    def test_evaluate(self):
        #Arrange
        model_evaluation = ModelEvaluation(TestModelEvaluation.input_data, TestModelEvaluation.model)

        #Act
        model_evaluation.evaluate(label_col)