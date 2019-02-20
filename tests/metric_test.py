#Unittest
import unittest as ut

#Timing
from common.metric import Metric

#Constants
from common import ut_constants

#Common parameters
metric_name = 'metric_name'

class TestMetric(ut.TestCase):
    def test_create(self):
        Metric.create(metric_name)
        Metric.create(metric_name)

    def test_public_no_metric(self):
        Metric.clear()

        with self.assertRaises(KeyError):
            Metric.publish(metric_name, 5)

    def test_publish(self):
        #Arrange
        Metric.create(metric_name)
        Metric.publish(metric_name, 10)
        Metric.publish(metric_name, 5)

        #Act
        metric_data = Metric.get(metric_name)

        #Assert
        self.assertCountEqual(metric_data, [10, 5])