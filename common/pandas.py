"""It contains wrappers over pandas module.
"""
#Constants
from common import constants

#Pandas
from pandas import read_csv
from pandas import DataFrame

#Random choice support
from collections import defaultdict
from random import sample

def csv_to_dataframe(filepath):
    """It loads the input csv files as a pandas DataFrame

    Arguments:
        filepath {pathlib.Path} -- It indicates the path to the input file.
    """
    return read_csv(filepath, index_col = 0)

def dataframe_to_csv(dataframe, filepath):
    """It saves the data frame to the input file path.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame that will be saved to file.
    """
    dataframe.to_csv(filepath)

def unique_items(dataframe, column):
    """It gets the list of unique items in the data frame.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column from which the unique elements will be extracted.
    """
    return dataframe[column].unique()

def count_items(dataframe, column):
    """It counts the occurances of each value in the column

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column to count the values.
    """
    return dataframe.groupby([column]).size().reset_index(name = constants.PANDAS_COUNT_AGG_COLUMN)

def sum_items(dataframe, column):
    """It sums the values of the column in the data frame.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column to sum the values.
    """
    return dataframe[column].sum()

def min_freq(dataframe, column):
    """It calculates the frequency of the least frequent label.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column to calculate the frequeny.
    """
    #Column value frequencies
    frequencies = count_items(dataframe, column)

    #Minimum frequency
    min_freq = min(frequencies[constants.PANDAS_COUNT_AGG_COLUMN])

    return min_freq

def random_choice(dataframe, column, items_per_column_value):
    """It creates a new dataframe with the same number of items for each column value.
        The number of items are indicated by the input parameter.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column to be redistributed.
        items_per_column_value {int} -- The number of items per column values to be present in the final dataframe.
    """
    column_row_indices = defaultdict(list)

    #Create a dictionary of column value and the indices of the rows with that value
    for index, row in dataframe.iterrows():
        #Column value to use as dictionary key
        column_value = row[column]

        #Add the row index for the column value
        column_row_indices[column_value].append(index)

    #Randomly choose the indices for each column value
    chosen_indices = [index for indices in column_row_indices.values() for index in sample(indices, items_per_column_value)]

    #Select the chosen indices from the input dataframe to create the resulting dataframe
    results = dataframe.iloc[chosen_indices].reset_index(drop = True)

    return results

def randomize(dataframe):
    """It randomizes the input dataframe.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
    """
    return dataframe.sample(frac = 1).reset_index(drop = True)

def remove(dataframe, column, values):
    """It removes the rows with the input column value from the dataframe.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        column {string} -- The name of the column on which to operate.
        values{string} -- The list of values of the column to use for row removal.
    """
    dataframe = dataframe[~dataframe[column].isin(values)].reset_index(drop = True)

    return dataframe

def group_aggregate_as_list(dataframe, grouping_column, aggregate_column):
    """It groups the dataframe by a column aggregating the rest as a list.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        grouping_column {string} -- The name of the column on which to group the data frame.
        aggregate_column {string} -- The column to aggregate as list.

    """
    grouped_dataframe = dataframe.groupby(grouping_column, as_index = False)[aggregate_column].agg({aggregate_column : list})

    return grouped_dataframe

def to_dict(dataframe, key_column, value_column):
    """It creates a dictionary out of the dataframe. The keys of the dictionary are taken from the key column.
    The values are taken from the value column.

    Arguments:
        dataframe {pandas.DataFrame} -- The data frame on which to operate.
        key_column {string} -- The name of the column to use as key.
        value_column {string} -- The column to use as value.

    """
    #Group the input dataframe by key column. The values in value column are added as a list to the output dataframe.
    grouped_dataframe = group_aggregate_as_list(dataframe, key_column, value_column)

    #Get the data as records
    records = grouped_dataframe.to_dict('records')

    #Create resulting dictionary
    result = dict(map(lambda x : (x[key_column], x[value_column]), records))

    return result
