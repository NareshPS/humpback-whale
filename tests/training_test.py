#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch
from unittest.mock import MagicMock, PropertyMock

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
from model.callback import ModelDropboxCheckpoint

#Input parameters
from operation.input import InputParameters, ImageGenerationParameters, TrainingParameters

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
    dropbox_auth, dropbox_dir = args.dropbox_parameters[0], Path(args.dropbox_parameters[1])

    return input_params, training_params, image_generation_params, transformation_params, dropbox_auth, dropbox_dir

def get_train_args():
    input_data = get_input_data()
    model = load_test_model()
    input_params, training_params, image_generation_params, transformation_params, dropbox_auth, dropbox_dir = get_params()
    image_generation_params.num_classes = 64
    trainer = ImageTraining(
                    input_params,
                    training_params,
                    image_generation_params,
                    transformation_params,
                    dropbox_auth,
                    dropbox_dir,
                    summary = False)

    return model, input_data, trainer

class TestImageTraining(ut.TestCase):
    def test_init(self):
        #Arrange
        input_params, training_params, image_generation_params, transformation_params, dropbox_auth, dropbox_dir = get_params()
        
        #Act
        trainer = ImageTraining(
                        input_params,
                        training_params,
                        image_generation_params,
                        transformation_params,
                        dropbox_auth,
                        dropbox_dir)

        #Assert input arguments
        self.assertEqual(input_params, trainer._input_params)
        self.assertEqual(training_params, trainer._training_params)
        self.assertEqual(image_generation_params, trainer._image_generation_params)
        self.assertEqual(transformation_params, trainer._transformer._parameters)
        self.assertEqual(dropbox_auth, trainer._dropbox_auth)
        self.assertEqual(dropbox_dir, trainer._dropbox_dir)

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
            self.assertEqual(len(args['callbacks']), 1)
            #self.assertTrue(isinstance(args['callbacks'][0], ModelDropboxCheckpoint))

            #Assert model and learning rate
            self.assert_response_and_learning_rate(model, response)

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
        model.fit_generator = MagicMock()
        model.train_on_batch = MagicMock()
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

    def test_batch_train(self):
        #Single epoch, all batches
        self.batch_train()

        #Multiple epoch, all batches
        self.batch_train(number_of_epochs = 5)

        #Single epoch, partial batches
        self.batch_train(batch_id = 3)

        #Multiple epoch, partial batches
        self.batch_train(batch_id = 2, number_of_epochs = 4)

        #Partial epochs, all batches
        self.batch_train(epoch_id = 2, number_of_epochs = 5)

        #Partial epochs, partial batches
        self.batch_train(batch_id = 2, epoch_id = 1, number_of_epochs = 4)
