#Unittest
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants

#Path parsing
from pathlib import Path

#Input file
from operation.input_file import InputFiles, ModelInput

#Random numbers
from random import randint

#Common parameters
file_paths = [Path('a.txt'), Path('b.txt'), Path('c.txt')]
remote_auth_token = 'remote_auth_token'
remote_dir_path = Path('.')

model_name = 'model_name'

class TestModelInput(ut.TestCase):
    def test_init(self):
        #Valid inputs
        _ = ModelInput(model_name, 1, 1, 20)
        _ = ModelInput(model_name, 1, 20, 20)
        _ = ModelInput(model_name, 3, 1, 5)
        _ = ModelInput(model_name, 2, 5, 9)

        #Invalid inputs
        with self.assertRaises(ValueError):
            _ = ModelInput(model_name, 0, 1, 20)

        with self.assertRaises(ValueError):
            _ = ModelInput(model_name, 1, 0, 20)

        with self.assertRaises(ValueError):
            _ = ModelInput(model_name, 1, 1, 0)

        with self.assertRaises(ValueError):
            _ = ModelInput(model_name, 1, 6, 5)

    def test_last_saved_file_name(self):
        #First input file
        model_input = ModelInput(model_name, 1, 1, 10)
        self.assertEqual('{}.session_id.0.set_id.0.epoch.1.h5'.format(model_name), str(model_input.last_saved_file_name()))

        #Non-first iteration and first set input file
        model_input = ModelInput(model_name, 2, 1, 10)
        self.assertEqual('{}.session_id.1.set_id.10.epoch.1.h5'.format(model_name), str(model_input.last_saved_file_name()))

        #Non-first iteration and non-first set input file
        model_input = ModelInput(model_name, 2, 2, 10)
        self.assertEqual('{}.session_id.2.set_id.1.epoch.1.h5'.format(model_name), str(model_input.last_saved_file_name()))

    def test_file_name(self):
        model_input = ModelInput(model_name, 1, 1, 10)
        self.assertEqual('{}.session_id.1.set_id.1.epoch.1.h5'.format(model_name), str(model_input.file_name()))

class TestInputFiles(ut.TestCase):
    def test_init(self):
        #Valid inputs
        _ = InputFiles(remote_auth_token, remote_dir_path)

        #Invalid inputs
        with self.assertRaises(ValueError):
            _ = InputFiles(None, remote_dir_path)

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