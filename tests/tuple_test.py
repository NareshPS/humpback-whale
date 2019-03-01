#Unittest
import unittest as ut

#Constants
from common import constants
from common import ut_constants

#Path manipulation
from pathlib import Path

#TupleGeneration
from siamese.tuple import TupleGeneration

#Data manipulation
from pandas import read_csv

#Logging
from common import logging

#Common params
anchors = ['0001f9222.jpg', '002b4615d.jpg', '00caa5c60.jpg']

class TestTupleGeneration(ut.TestCase):
    def get_label_df(self):
        return read_csv(ut_constants.LABEL_DATAFRAME_PATH)

    def get_tuples(self, num_positive_samples, num_negative_samples):
        #Arrange
        label_df = self.get_label_df()
        tup_gen = TupleGeneration(
                        label_df,
                        ut_constants.LABEL_DATAFRAME_IMAGE_COL,
                        ut_constants.LABEL_DATAFRAME_LABEL_COL,
                        ut_constants.TUPLE_DATAFRAME_COLS)
        
        return tup_gen.get_tuples(num_positive_samples, num_negative_samples)

    def verify(self, tuple_df, anchor, requested_positive_samples, requested_negative_samples):
        #Label DataFrame
        label_df = self.get_label_df()
        label = label_df.loc[label_df[ut_constants.LABEL_DATAFRAME_IMAGE_COL] == anchor].iloc[0][ut_constants.LABEL_DATAFRAME_LABEL_COL]
        max_positive_samples = len(label_df.loc[label_df[ut_constants.LABEL_DATAFRAME_LABEL_COL] == label])

        #Expected sample counts
        expected_positive_samples = min(max_positive_samples, requested_positive_samples)
        expected_negative_samples = requested_negative_samples

        #Columns
        anchor_col = ut_constants.TUPLE_DATAFRAME_COLS[0]
        label_col = ut_constants.TUPLE_DATAFRAME_COLS[2]

        positive_samples = tuple_df.loc[(tuple_df[anchor_col] == anchor) & (tuple_df[label_col] == 1)]
        negative_samples = tuple_df.loc[(tuple_df[anchor_col] == anchor) & (tuple_df[label_col] == 0)]

        #Assert counts
        self.assertEqual(len(positive_samples), expected_positive_samples)
        self.assertEqual(len(negative_samples), expected_negative_samples)

    def test_get_tuples_within_range(self):
        #Arrange
        num_positive_samples = 1
        num_negative_samples = 3

        #Act
        tuple_df = self.get_tuples(num_positive_samples, num_negative_samples)

        #Assert
        for anchor in anchors:
            self.verify(tuple_df, anchor, num_positive_samples, num_negative_samples)

    def test_get_tuples_too_large_sample_count(self):
        #Arrange
        num_positive_samples = 10
        num_negative_samples = 3

        #Act
        tuple_df = self.get_tuples(num_positive_samples, num_negative_samples)

        #Assert
        for anchor in anchors:
            self.verify(tuple_df, anchor, num_positive_samples, num_negative_samples)