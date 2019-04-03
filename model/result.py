"""This classes handles the keras fit result.
"""
#Default valued dictionary
from collections import defaultdict

class EpochResponse(object):
    def __init__(self, epoch_id, metric_names):
        """It initializes the required and local variables.

        Arguments:
            epoch_id {int} -- The current epoch id.
            metric_names {[string]} -- The names of the metrics.
        """  
        #Required parameters
        self._epoch_id = epoch_id
        self._metric_names = metric_names

        #Local variables
        self._batch_ids = []
        self._metrics = defaultdict(list)

    @property
    def epoch_id(self):
        return self._epoch_id

    @property
    def batch_ids(self):
        return self._batch_ids

    @property
    def metric_names(self):
        return self._metric_names

    def _update(self, result, batch_id):
        """It updates the metrics for the epoch with the new batch result.

        Arguments:
            result {[float]} -- The return value of the training call.
            batch_id {int} -- The current batch id.
        """
        #Add the batch id to the processed list
        self._batch_ids.append(batch_id)

        #Response is a scalar
        if not isinstance(result, list):
            result = [result]

        for index, metric_name in enumerate(self._metric_names):
            self._metrics[metric_name].append(result[index])

    def append(self, result, batch_id):
        """It updates the metrics for the epoch with the new batch result.

        Arguments:
            result {[float]} -- The return value of the training call.
            batch_id {int} -- The current batch id.
        """  
        self._update(result, batch_id)

    def batches(self):
        return self._batch_ids

    def metrics(self):
        return self._metrics

    def get(self, metric_name):
        return self._metrics[metric_name]

    def __str__(self):
        return """
                    epoch_id: {} metric_names: {}
                    batch_ids: {}
                    metrics: {}
               """.format(
                        self._epoch_id,
                        self._metric_names,
                        self._batch_ids,
                        self._metrics)
