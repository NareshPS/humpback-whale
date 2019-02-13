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

def get_tuples():
    #Tuples file name
    input_tuples_file_name = 'input_tuples_p5_n5.tuples'

    #Tuple file path
    input_tuples_file_path = ut_constants.DATA_STORE / input_tuples_file_name

    return read_csv(input_tuples_file_path)

def get_model():
    #Model file name
    model_file_name = 'cnn_model2d_1.h5'

    #Model file path
    model_file_path = ut_constants.DATA_STORE / model_file_name

    return load_model(str(model_file_path))

class TestTupleEvaluation(ut.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            _ = LabelEvaluation(None)

        #Label  DataFrame
        input_tuples_df = get_tuples()

        _ = LabelEvaluation(input_tuples_df)

    def test_evaluate(self):
        #Label  DataFrame
        input_tuples_df = get_tuples()

        #Tuple evaluation object
        label_evaluation = LabelEvaluation(input_tuples_df)
        label_pcts = label_evaluation.evaluate(label_col)

        #Assert
        self.assertAlmostEqual(label_pcts[0], 66.84491, places = 2)
        self.assertAlmostEqual(label_pcts[1], 33.15508, places = 2)

class TestModelEvaluation(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        #Arrange
        cls.input_tuples = get_tuples()
        cls.model = get_model()

    def test_init(self):
        with self.assertRaises(ValueError):
            #Input tuples
            _ = ModelEvaluation(None, TestModelEvaluation.model)

        with self.assertRaises(ValueError):
            #Model
            _ = ModelEvaluation(TestModelEvaluation.input_tuples, None)

    def test_evaluate(self):
        #Arrange
        model_evaluation = ModelEvaluation(TestModelEvaluation.input_tuples, TestModelEvaluation.model)

        #Act
        model_evaluation.evaluate(label_col)
