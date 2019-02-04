"""It computes statistics on input tuples.
"""

#Constants
from common import constants

class TupleEvaluation(object):
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

    def evaluate(self):
        """It creates a dictionary of labels with their percentages in the dataframe.
        
        Returns:
            {A dictionary} -- A dictionary of labels to their percentages.
        """
        #Placeholder for the dictionary
        label_percentages = {}

        #Total number of items
        total = len(self._dataframe)

        #Count of respective labels
        label_counts = self._dataframe.groupby(constants.TRAIN_TUPLE_LABEL_COL).size().reset_index(name = TupleEvaluation.count)

        #Percentages
        for _, row in label_counts.iterrows():
            label = row[constants.TRAIN_TUPLE_LABEL_COL]
            count = row[TupleEvaluation.count]
            
            #Percentage calculation
            label_pct = (count / total) * 100.

            #Add to dictionary
            label_percentages[label] = label_pct

        return label_percentages