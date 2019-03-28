"""This class handles transformation of training metrics
to generate useful insights.
"""

from itertools import chain

class RunMetrics(object):
    def __init__(self, metrics):
        """It initializes the object with training metrics.

        Arguments:
            results {dict(string, dict)} -- A dictionary of metric name to a dictionary with its values ordered by epoch and batch ids.
        """
        self._metrics = metrics

    def get(self, metric, epoch_id = None, batch_id = None):
        """It computes the values of the input metric name optionally by epoch_id and batch_id.

        Arguments:
            metric {string} -- The name of the metric whose values are fetched.
            epoch_id {int} -- The epoch id for which the values are fetched.
            batch_id {int} -- The batch id for which the value is fetched.
        """
        #Validation
        if epoch_id is None and batch_id is not None:
            raise ValueError('Epoch id must be provided with batch id. epoch_id: {} batch_id: {}'.format(epoch_id, batch_id))

        metric_values = []

        #Both epoch_id and batch_id are specified.
        if batch_id is not None:
            #Metric value for an epoch and a single batch.
            value = self._metrics[metric][epoch_id][batch_id]

            #Add the value to the result
            metric_values.append(value)

        #epoch_id is specified.
        elif epoch_id is not None:
            #Metric values for an entire epoch.
            values = self._metrics[metric][epoch_id]

            #Add the values to the result
            metric_values.extend(values)

        #Both epoch_id and batch_id are not specified.
        else:
            #The number of epochs is same as the size of the metric dictionary
            n_epochs = len(self._metrics[metric])

            #Combine the metric values for all epochs.
            values = chain.from_iterable([self._metrics[metric][epoch_id] for epoch_id in range(n_epochs)])

            #Update the result
            metric_values = values

        return metric_values
