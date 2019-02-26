#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch
from unittest.mock import MagicMock, PropertyMock

#Constants
from common import ut_constants
from common import constants

#Training
from operation.transform import ImageDataTransformation
from operation.training import ImageTraining
from operation.image import ImageDataIterator
from model.callback import ModelDropboxCheckpoint

#Input parameters
from operation.input import InputParameters, ImageGenerationParameters, TrainingParameters, RunParameters

#Test support
from tests.support.utils import get_args, get_input_data, load_test_model, patch_imload

#File operations
from pathlib import Path

def get_params():
    args = get_args(image_cols = ['Image'])

    input_params = InputParameters(args)
    image_generation_params = ImageGenerationParameters(args)
    training_params = TrainingParameters(args)
    session_params = RunParameters(args)
    dropbox_auth, dropbox_dir = args.dropbox_parameters[0], Path(args.dropbox_parameters[1])

    return input_params, image_generation_params, training_params, session_params, dropbox_auth, dropbox_dir

class TestImageTraining(ut.TestCase):
    def test_init(self):
        #Arrange
        input_params, image_generation_params, training_params, _, dropbox_auth, dropbox_dir = get_params()
        transformation_params = ImageDataTransformation.Parameters(samplewise_mean = True)

        #Act
        trainer = ImageTraining(
                        input_params,
                        image_generation_params,
                        training_params,
                        transformation_params,
                        dropbox_auth,
                        dropbox_dir)

        #Assert input arguments
        self.assertEqual(input_params, trainer._input_params)
        self.assertEqual(image_generation_params, trainer._image_generation_params)
        self.assertEqual(training_params, trainer._training_params)
        self.assertEqual(transformation_params, trainer._transformer._parameters)
        self.assertEqual(dropbox_auth, trainer._dropbox_auth)
        self.assertEqual(dropbox_dir, trainer._dropbox_dir)

    def test_train(self):
        #Arrange
        input_data = get_input_data()
        model = load_test_model()
        input_params, image_generation_params, training_params, session_params, dropbox_auth, dropbox_dir = get_params()
        transformation_params = ImageDataTransformation.Parameters(samplewise_mean = True)
        trainer = ImageTraining(input_params, image_generation_params, training_params, transformation_params, dropbox_auth, dropbox_dir)

        #Mock the fit call
        model.fit_generator = MagicMock()

        with mock_patch('operation.utils.imload', side_effect = patch_imload):
            #Act
            trainer.train(model, input_data, session_params)

            #Assert
            model.fit_generator.assert_called_once()
            _, args = model.fit_generator.call_args_list[0]
            self.assertTrue(isinstance(args['generator'], ImageDataIterator))
            self.assertTrue(isinstance(args['validation_data'], ImageDataIterator))
            self.assertTrue(args['epochs'], training_params.number_of_epochs)
            self.assertEqual(len(args['callbacks']), 1)
            self.assertTrue(isinstance(args['callbacks'][0], ModelDropboxCheckpoint))