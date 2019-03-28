#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch
from unittest.mock import MagicMock, PropertyMock, Mock

#Constants
from common import ut_constants
from common import constants

#Keras imports
from keras import backend as K
from keras.models import Model

#Pandas import
from pandas import DataFrame

#Training
from operation.transform import ImageDataTransformation
from operation.training import ImageTraining
from operation.image import ImageDataIterator

#Random numbers
from random import random

#Input parameters
from operation.input import InputParameters, ImageGenerationParameters, TrainingParameters

#Training result
from model.result import EpochResponse

#Test support
from tests.support.utils import get_args, get_input_data, load_test_model, patch_imload

#Math operations
from math import ceil

#File operations
from pathlib import Path

def get_params():
    args = get_args(image_cols = ['Image'])

    input_params = InputParameters(args)
    training_params = TrainingParameters(args)
    image_generation_params = ImageGenerationParameters(args)
    transformation_params = ImageDataTransformation.Parameters(samplewise_mean = True)

    return input_params, training_params, image_generation_params, transformation_params

def get_train_args():
    input_data = get_input_data()
    model = load_test_model()
    input_params, training_params, image_generation_params, transformation_params = get_params()
    image_generation_params.num_classes = 64

    trainer = ImageTraining(
                    input_params,
                    training_params,
                    image_generation_params,
                    transformation_params,
                    MagicMock(),
                    summary = False)

    return model, input_data, trainer

class TestImageTraining(ut.TestCase):
    def test_init(self):
        #Arrange
        input_params, training_params, image_generation_params, transformation_params = get_params()

        #Act
        trainer = ImageTraining(
                        input_params,
                        training_params,
                        image_generation_params,
                        transformation_params,
                        MagicMock())

        #Assert input arguments
        self.assertEqual(input_params, trainer._input_params)
        self.assertEqual(training_params, trainer._training_params)
        self.assertEqual(image_generation_params, trainer._image_generation_params)
        self.assertEqual(transformation_params, trainer._transformer._parameters)
        self.assertIsNotNone(trainer._checkpoint_callback)

    def assert_response_and_learning_rate(self, model, response):
        #Assert learning rate
        K.set_value.assert_called_once()
        args, _  = K.set_value.call_args_list[0]
        self.assertEqual(args[0], model.optimizer.lr)
        self.assertEqual(args[1], 0.001)


        #Assert the train response
        self.assertEqual(2, len(response))
        self.assertTrue(isinstance(response[0], Model))
        self.assertTrue(isinstance(response[1], DataFrame))

    def test_train(self):
        #Arrange
        model, input_data, trainer = get_train_args()

        #Mock the relevant calls
        model.fit_generator = MagicMock()
        model.train_on_batch = MagicMock()
        K.set_value = MagicMock()

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            response = trainer.train(model, input_data)

            #Assert
            model.fit_generator.assert_called_once()
            model.train_on_batch.assert_not_called()
            _, args = model.fit_generator.call_args_list[0]
            self.assertTrue(isinstance(args['generator'], ImageDataIterator))
            self.assertTrue(isinstance(args['validation_data'], ImageDataIterator))
            self.assertTrue(args['epochs'], trainer._training_params.number_of_epochs)

            #Assert model and learning rate
            self.assert_response_and_learning_rate(model, response)

    def assert_checkpoint(self, checkpoint, total_training_batches, total_training_epochs):
        call_args_list = checkpoint.on_batch_begin.call_args_list
        self.assertEqual(len(call_args_list), total_training_batches)

        call_args_list = checkpoint.on_batch_end.call_args_list
        self.assertEqual(len(call_args_list), total_training_batches)

        call_args_list = checkpoint.on_epoch_begin.call_args_list
        self.assertEqual(len(call_args_list), total_training_epochs)

        call_args_list = checkpoint.on_epoch_end.call_args_list
        self.assertEqual(len(call_args_list), total_training_epochs)

        call_args_list = checkpoint.set_input_data.call_args_list
        self.assertEqual(len(call_args_list), total_training_epochs)

        call_args_list = checkpoint.set_model.call_args_list
        self.assertEqual(len(call_args_list), total_training_epochs + total_training_batches)

        call_args_list = checkpoint.set_result.call_args_list
        self.assertEqual(len(call_args_list), total_training_batches)

    def batch_train(self, batch_id = 0, epoch_id = 0, number_of_epochs = 1):
        #Arrange
        model, input_data, trainer = get_train_args()

        #Arrange batch size
        batch_size = 40
        trainer._image_generation_params.batch_size = batch_size
        number_of_batches = ceil(len(input_data) / batch_size)
        first_epoch_batches = number_of_batches - batch_id
        total_training_epochs = number_of_epochs - epoch_id
        total_training_batches = first_epoch_batches + number_of_batches * (total_training_epochs - 1)

        #Arrange batch and epoch parameters
        trainer._training_params.batch_id = batch_id
        trainer._training_params.epoch_id = epoch_id
        trainer._training_params.number_of_epochs = number_of_epochs

        #Mock the relevant calls
        model.fit_generator = Mock(return_value = [random() for _ in model.metrics_names])
        model.train_on_batch = Mock(return_value = [random() for _ in model.metrics_names])
        trainer._transformer.fit = MagicMock()
        K.set_value = MagicMock()

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            response = trainer.batch_train(model, input_data)

            #Assert
            model.fit_generator.assert_not_called()

            #Assert train batch calls
            call_args_list = model.train_on_batch.call_args_list
            self.assertEqual(len(call_args_list), total_training_batches)

            #Assert model and learning rate
            self.assert_response_and_learning_rate(model, response)

            #Assert checkpoint calls
            self.assert_checkpoint(trainer._checkpoint_callback, total_training_batches, total_training_epochs)

    def test_batch_train(self):
        #Single epoch, all batches
        self.batch_train()

        #Multiple epoch, all batches
        self.batch_train(number_of_epochs = 2)

        #Single epoch, partial batches
        self.batch_train(batch_id = 1)

        #Multiple epoch, partial batches
        self.batch_train(batch_id = 1, number_of_epochs = 2)

        #Partial epochs, all batches
        self.batch_train(epoch_id = 1, number_of_epochs = 2)

        #Partial epochs, partial batches
        self.batch_train(batch_id = 1, epoch_id = 1, number_of_epochs = 2)

    def test_batch_train_no_lr(self):
        #Arrange
        model, input_data, trainer = get_train_args()
        trainer._training_params.learning_rate = None
        trainer._training_params.number_of_epochs = 1
        trainer._image_generation_params.batch_size = 40
        trainer._dropbox_auth = None
        trainer._dropbox_dir = None

        #Mock the relevant calls
        model.train_on_batch = Mock(return_value = [random() for _ in model.metrics_names])
        K.set_value = MagicMock()

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            _ = trainer.batch_train(model, input_data)

            #Assert
            K.set_value.assert_not_called()

    def batch_train_input_data_randomized(self, start_batch_id, is_randomized = True):
        #Arrange
        model, input_data, trainer = get_train_args()
        trainer._training_params.batch_id = start_batch_id
        trainer._training_params.epoch_id = 0
        trainer._training_params.number_of_epochs = 1
        trainer._image_generation_params.batch_size = 40
        trainer._dropbox_auth = None
        trainer._dropbox_dir = None

        #Mock the relevant calls
        model.train_on_batch = Mock(return_value = [random() for _ in model.metrics_names])
        trainer._transformer.fit = MagicMock()
        input_data.sample = MagicMock()

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            _ = trainer.batch_train(model, input_data)

            #Assert
            if is_randomized:
                input_data.sample.assert_called_once()
            else:
                input_data.sample.assert_not_called()

    def test_batch_train_input_data_randomized(self):
        self.batch_train_input_data_randomized(0, is_randomized = True)
        self.batch_train_input_data_randomized(1, is_randomized = False)
