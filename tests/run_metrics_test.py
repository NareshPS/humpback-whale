#Unittest
import unittest as ut

#Run metrics
from evaluation.run_metrics import RunMetrics

#Constants
from common import ut_constants

#Common parameters
metrics = {
            'loss' : {
                        0 : [0.623, 0.152, 0.314],
                        1 : [0.123, 0.452, 0.214]
                    },
            'accuracy' : {
                        0 : [0.223, 0.352, 0.214],
                        1 : [0.323, 0.152, 0.314]
                        }
            }

metric = 'loss'

class TestRunMetrics(ut.TestCase):
    def get(self, result, epoch_id = None, batch_id = None):
        #Arrange
        r_metrics = RunMetrics(metrics)

        #Act
        values = r_metrics.get(metric, epoch_id, batch_id)

        #Assert
        self.assertCountEqual(values, result)

    def test_get(self):
        #Arrange result
        result = [0.623, 0.152, 0.314, 0.123, 0.452, 0.214]

        #Act & Assert
        self.get(result)

    def test_get_with_epoch_id(self):
        #Arrange epoch_id and result
        epoch_id = 1
        result = [0.123, 0.452, 0.214]

        #Act & Assert
        self.get(result, epoch_id)

    def test_get_with_epoch_id_and_batch_id(self):
        #Arrange epoch_id and result
        epoch_id = 1
        batch_id = 2
        result = [0.214]

        #Act & Assert
        self.get(result, epoch_id, batch_id)

