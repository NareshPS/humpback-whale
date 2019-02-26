"""It computes statistics on input tuples.
"""
#pandas DataFrame
from pandas import DataFrame, Series
from pandas import cut

#Numpy imports
from numpy import linspace, argmax, where

#Constants
from common import constants

class LabelEvaluation(object):
    def __init__(self, dataframe):
        """It sets the input parameters
        
        Arguments:
            dataframe {A Pandas DataFrame} -- The candidate dataframe for evaluation.
        """
        self._dataframe = dataframe

        #Validation
        if self._dataframe is None:
            raise ValueError("The dataframe must be valid.")

    def distribution(self, label_col):
        """It creates a dictionary of labels with their percentages in the dataframe.

        Arguments:
            label_col {string} -- The name of the label column
        
        Returns:
            {A DataFrame} -- A DataFrame object with the label counts and percentages.
        """
        #Placeholder for the data frame
        label_statistics = DataFrame(columns = [label_col, constants.PANDAS_COUNT_AGG_COLUMN, constants.PANDAS_PCT_AGG_COLUMN])

        #Total number of items
        total = len(self._dataframe)

        #Count of respective labels
        label_counts = self._dataframe.groupby(label_col).size().reset_index(name = constants.PANDAS_COUNT_AGG_COLUMN)

        #Percentages
        for _, row in label_counts.iterrows():
            label = row[label_col]
            count = row[constants.PANDAS_COUNT_AGG_COLUMN]
            
            #Percentage calculation
            label_pct = (count / total) * 100.

            #Append the statistics to output data frame
            label_statistics = label_statistics.append({
                                                            label_col : label,
                                                            constants.PANDAS_COUNT_AGG_COLUMN : count,
                                                            constants.PANDAS_PCT_AGG_COLUMN : label_pct
                                                        },
                                                        ignore_index = True)

        #Sort by the percentage column
        label_statistics = label_statistics.sort_values(constants.PANDAS_PCT_AGG_COLUMN)

        return label_statistics

    def bin(self, label_col, number_of_bins):
        """It bins the inputs based on counts
        
        Arguments:
            label_col {string} -- The name of the label column.
            number_of_bins {int} -- The number of bins
        """
        #Compute the statistics first
        label_counts = self._dataframe.groupby(label_col).size().reset_index(name = constants.PANDAS_COUNT_AGG_COLUMN)
        label_counts_series = getattr(label_counts, constants.PANDAS_COUNT_AGG_COLUMN)

        count_bins = linspace(label_counts_series.min(), label_counts_series.max(), number_of_bins + 1)
        binned_counts = label_counts.groupby(cut(label_counts_series, count_bins))[constants.PANDAS_COUNT_AGG_COLUMN].sum().reset_index(name = constants.PANDAS_COUNT_BIN_COLUMN)

        return binned_counts

    def histogram(self, label_col):
        """It computes the histogram of the counts

        Arguments:
            label_col {string} -- The name of the label column.
        """
        #Count of respective labels
        label_counts = self._dataframe.groupby(label_col).size().reset_index(name = constants.PANDAS_COUNT_AGG_COLUMN)

        #Count the label counts
        label_histogram = label_counts.groupby(constants.PANDAS_COUNT_AGG_COLUMN).size().to_frame(constants.PANDAS_COUNT_HIST_COLUMN).reset_index()

        return label_histogram

class ModelEvaluation(object):
    def __init__(self, dataframe, label_col):
        """It sets the input parameters
        
        Arguments:
            dataframe {A Pandas DataFrame} -- The candidate dataframe for evaluation.
            label_col {string} -- The name of the label column.
        """
        self._dataframe = dataframe
        self._label_col = label_col

        #Validation
        if self._dataframe is None:
            raise ValueError("The dataframe must be valid.")

        if self._label_col is None:
            raise ValueError("The label column name must be valid.")

    def evaluate(self, predictions):
        """It uses the dataframe to calculate the model accuracy values per label.

        Arguments:
            
        """
        

    def _get_labels(self):
        """It calculates the number of distinct labels in the dataframe.
        """
        #Count labels
        label_counts = self._dataframe.groupby(self._label_col).size()

        #List of labels
        labels = label_counts.loc[:, self._label_col].tolist()

        return labels

class PredictionEvaluation(object):
    def __init__(self, input_data, predictions, label_col):
        """It sets the input parameters.
        
        Arguments:
            input_data {pandas.DataFrame} -- The input data frame.
            predictions {np.array} -- The predictions for the input data.
            label_col {string} -- The name of the label column in the input data frame
        """
        self._input_data = input_data
        self._predictions = predictions
        self._label_col = label_col

    def evaluate(self):
        self._input_data[constants.PANDAS_PREDICTION_COLUMN] = Series(argmax(self._predictions, axis = 1), index = self._input_data.index)
        match_series = where(
                            self._input_data[self._label_col] == self._input_data[constants.PANDAS_PREDICTION_COLUMN],
                            1, #True Value
                            0)
        self._input_data[constants.PANDAS_MATCH_COLUMN] = match_series

        return self._input_data