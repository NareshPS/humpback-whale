#Unittest
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants

#Path parsing
from pathlib import Path

#Input file and parameters
from operation.input import TrainingParameters
from iofiles.input_file import InputFiles, ModelInput

#Test support
from tests.support.utils import get_train_params

#Random numbers
from random import randint

#Common parameters
file_paths = [Path('a.txt'), Path('b.txt'), Path('c.txt')]
remote_auth_token = 'remote_auth_token'
remote_dir_path = Path('.')

model_name = 'model_name'

class TestModelInput(ut.TestCase):
    @classmethod
    def setUpClass(cls):
        #Arrange
        cls.train_params = get_train_params()

    def test_init(self):
        #Valid inputs
        _ = ModelInput(model_name, self.train_params)

        #Invalid inputs
        with self.assertRaises(ValueError):
            _ = ModelInput(None, self.train_params)

    def test_file_name(self):
        model_input = ModelInput(model_name, self.train_params)
        self.assertEqual('{}.batch.0.epoch.0.h5'.format(model_name), str(model_input.file_name()))

class TestInputFiles(ut.TestCase):
    def test_init(self):
        #Valid inputs
        _ = InputFiles(remote_auth_token, remote_dir_path)

        #Invalid inputs
        with self.assertRaises(ValueError):
            _ = InputFiles(remote_auth_token, None)

    def get_all(self, inputs, func):
        #Arrange
        with mock_patch.object(Path, 'exists') as mock_exists:
            mock_exists.side_effect = func

            valid_file_paths = inputs.get_all(file_paths)
            
            #Assert
            for file_path_hint, file_path_verified in valid_file_paths.items():
                self.assertEqual(file_path_hint, file_path_verified)

            #Assert
            self.assertCountEqual(file_paths, list(valid_file_paths.keys()))

    def test_get_all_just_local_files(self):
        #Arrange
        inputs = InputFiles(remote_auth_token, remote_dir_path)

        #Act & Assert
        self.get_all(inputs, lambda: True)

    def test_get_all_just_remote_files(self):
        #Arrange
        inputs = InputFiles(remote_auth_token, remote_dir_path)
        inputs._dropbox.download = MagicMock()

        self.get_all(inputs, lambda: False)

    def test_get_all_local_and_remote_files(self):
        #Arrange
        inputs = InputFiles(remote_auth_token, remote_dir_path)
        inputs._dropbox.download = MagicMock()

        self.get_all(inputs, lambda: bool(randint(0, 1)))

    def test_get_all_remote_files_dropbox_not_initialized(self):
        #Arrange
        inputs = InputFiles()

        with self.assertRaises(ValueError):
            self.get_all(inputs, lambda: False)