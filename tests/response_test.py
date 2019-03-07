#Unittests
import unittest as ut
from unittest.mock import MagicMock

#Response classes
from model.response import EpochResponse

#Default valued dictionary
from collections import defaultdict

#Test support
from tests.support.utils import create_model, input_shape, create_model_input_data

epoch_id = 2
metric_names = ['accuracy']

def new_model(metrics = metric_names):
    model = create_model(
                input_shape = input_shape,
                num_classes = 1,
                metrics = metrics)

    return model

class TestEpochResponse(ut.TestCase):
    def test_init(self):
        #Arrange & Act
        epoch_response = EpochResponse(epoch_id, metric_names)

        #Assert
        self.assertEqual(epoch_id, epoch_response._epoch_id)
        self.assertCountEqual(metric_names, epoch_response._metric_names)
        self.assertEqual([], epoch_response._batch_ids)
        self.assertEqual(defaultdict(list), epoch_response._metrics)

    def test_append_no_metrics(self):
        #Arrange
        model = new_model(metrics = None)
        X, Y = create_model_input_data(input_shape)
        result = model.train_on_batch(X, Y)
        response = EpochResponse(epoch_id, metric_names)

        #Act
        response.append(result, 0)

        #Assert
        self.assertCountEqual([0], response.batches())
        self.assertEqual(['accuracy'], list(response.metrics().keys()))
        self.assertEqual(1, len(response.metrics()))

    def test_append_metric(self):
        #Arrange
        model = new_model(metrics = metric_names)
        X, Y = create_model_input_data(input_shape)
        result = model.train_on_batch(X, Y)
        response = EpochResponse(epoch_id, metric_names)

        #Act
        response.append(result, 1)

        #Assert
        self.assertCountEqual([1], response.batches())
        self.assertEqual(['accuracy'], list(response.metrics().keys()))
        self.assertEqual(1, len(response.metrics()))