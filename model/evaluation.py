"""It computes statistics on input tuples.
"""

#Constants
from common import constants

class LabelEvaluation(object):
    count = 'Count'

    def __init__(self, dataframe):
        """It sets the input parameters
        
        Arguments:
            dataframe {A Pandas DataFrame} -- The candidate dataframe for evaluation.
        """
        self._dataframe = dataframe

        #Validation
        if self._dataframe is None:
            raise ValueError("The dataframe must be valid.")

    def evaluate(self, label_col):
        """It creates a dictionary of labels with their percentages in the dataframe.

        Arguments:
            label_col {string} -- The name of the label column
        
        Returns:
            {A dictionary} -- A dictionary of labels to their percentages.
        """
        #Placeholder for the dictionary
        label_percentages = {}

        #Total number of items
        total = len(self._dataframe)

        #Count of respective labels
        label_counts = self._dataframe.groupby(label_col).size().reset_index(name = constants.PANDAS_COUNT_AGG_COLUMN)

        #Percentages
        for _, row in label_counts.iterrows():
            label = row[label_col]
            count = row[LabelEvaluation.count]
            
            #Percentage calculation
            label_pct = (count / total) * 100.

            #Add to dictionary
            label_percentages[label] = label_pct

        return label_percentages

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


