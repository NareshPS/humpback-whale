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
from iofiles.input_file import InputFiles, ModelInput, InputDataFile

#Test support
from tests.support.utils import get_input_data, load_test_model

#Random numbers
from random import randint

#Common parameters
file_paths = [Path('a.txt'), Path('b.txt'), Path('c.txt')]
remote_auth_token = 'remote_auth_token'
remote_dir_path = Path('.')

model_name = 'model_name'
batch_id = 2
epoch_id = 4

class TestModelInput(ut.TestCase):
    def test_init(self):
        #Valid inputs
        _ = ModelInput(model_name)

    def test_save(self):
        #Arrange
        model = load_test_model()
        model_input = ModelInput(model_name)
        
        #Mocks
        model.save = MagicMock()

        #Act
        model_input.save(model, batch_id, epoch_id)

        #Assert
        model.save.assert_called_with(str(model_input.file_name(batch_id, epoch_id)))

    def test_file_name(self):
        #Arrange
        model_input = ModelInput(model_name)
        self.assertEqual('{}.batch.2.epoch.4.h5'.format(model_name), str(model_input.file_name(batch_id, epoch_id)))

class TestInputDataFile(ut.TestCase):
    def test_init(self):
        #Valid inputs
        _ = InputDataFile()

    def test_save(self):
        #Arrange
        input_data = get_input_data()
        input_data_file = InputDataFile()

        #Mocks
        input_data.to_csv = MagicMock()

        #Act
        input_data_file.save(input_data, batch_id, epoch_id)

        #Assert
        input_data.to_csv.assert_called_with(input_data_file.file_name(batch_id, epoch_id))

    def test_file_name(self):
        #Arrange
        input_data_file = InputDataFile()
        file_name = Path('input_data.batch.0.epoch.{}.csv'.format(epoch_id))

        #Act & Assert
        self.assertEqual(input_data_file.file_name(batch_id, epoch_id), file_name)

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