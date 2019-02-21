#Local pandas wrapper
from common.pandas import unique_items, count_items, min_freq, random_choice, randomize

#Label evaluation
from model.evaluation import LabelEvaluation

class Rebalancing(object):
    def __init__(self, data, label_col):
        """It sets up the input parameters.
        
        Arguments:
            data {pandas.DataFrame} -- A data frame that contains the input data in columnar form.
            label_col {string} -- The name of the label column in the data frame.
        """
        #Required parameters
        self._data = data
        self._label_col = label_col

    def rebalance(self, statistics = False):
        """It rebalances the input data frame label distribution to improve training performance.

        Arguments:
            statistics {boolean} -- It indicates to return the rebalancing statistics
        """
        #Least frequent label count
        min_label_freq =  min_freq(self._data, self._label_col)

        #Select the required rows for each column value, then randomize the output
        results = random_choice(self._data, self._label_col, min_label_freq)
        results = randomize(results)

        #Evaluation placeholders
        pre_evaluation_statistics = None
        post_evaluation_statistics = None

        #Compute statistics if statistics flag is enabled.
        if statistics:
            #Pre-balancing statistics
            pre_evaluation = LabelEvaluation(self._data)
            pre_evaluation_statistics = pre_evaluation.distribution(self._label_col)

            #Post-balancing statistics
            post_evaluation = LabelEvaluation(results)
            post_evaluation_statistics = post_evaluation.distribution(self._label_col)

        return results, pre_evaluation_statistics, post_evaluation_statistics