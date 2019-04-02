#Unittest
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch as mock_patch

#Constants
from common import constants
from common import ut_constants

#Dropbox
from client.dropbox import DropboxConnection

#File size calculation
from os import path

#Path manipulation
from pathlib import Path

#Logging
from common import logging

#Common parameters
remote_dir = Path('remote_dir')
small_file = 'small_file.txt'
cnn_model_file = 'cnn_model2d_1.h5'

class TestDropboxConnection(ut.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDropboxConnection, self).__init__(*args, **kwargs)

        #Initialize logging
        logging.initialize('unittest')

    def get_client(self):
        return DropboxConnection(self._dropbox_params)

    @classmethod
    def setUpClass(cls):
        cls._dropbox_params = DropboxConnection.Parameters('auth_token', remote_dir)

    def test_get_client(self):
        #Act
        client = DropboxConnection.get_client('auth_token', remote_dir)

        #Assert
        self.assertIsNotNone(client)

    def test_get_client_from_params_no_params(self):
        #Act
        client = DropboxConnection.get_client_from_params(None)

        #Assert
        self.assertIsNone(client)

    def test_get_client_from_params_with_params(self):
        #Act
        client = DropboxConnection.get_client_from_params(['auth_token', remote_dir])

        #Assert
        self.assertIsNotNone(client)

    def test_init(self):
        _ = DropboxConnection(self._dropbox_params)

    def test_upload_invalid_source_file(self):
        #Arrange
        client = self.get_client()

        #Act & Assert
        with self.assertRaises(ValueError):
            client.upload(None)

        with self.assertRaises(FileNotFoundError):
            client.upload('a_random_file')

    def test_upload_small_source_file(self):
        #Arrange
        client = self.get_client()
        client._upload_small_file = MagicMock()
        remote_file_path = constants.DROPBOX_APP_PATH_PREFIX / remote_dir / small_file
        local_file_path = ut_constants.DROPBOX_STORE / small_file

        #Act
        client.upload(local_file_path)

        #Assert
        args, _ = client._upload_small_file.call_args_list[0]
        self.assertEqual(2, len(args))
        self.assertEqual(remote_file_path.as_posix(), args[1])

    def test_upload_large_source_file(self):
        #Arrange
        client = self.get_client()
        client._upload_large_file = MagicMock()
        remote_file_path = constants.DROPBOX_APP_PATH_PREFIX / remote_dir / cnn_model_file
        local_file_path = ut_constants.DATA_STORE / cnn_model_file

        #Act
        client.upload(local_file_path)

        #Assert
        args, _ = client._upload_large_file.call_args_list[0]
        self.assertEqual(3, len(args))
        self.assertEqual(path.getsize(local_file_path), args[1])
        self.assertEqual(remote_file_path.as_posix(), args[2])

    def test_download_invalid_remote_file_name(self):
        #Arrange
        client = self.get_client()

        with self.assertRaises(ValueError):
            client.download(None)

    def test_download_file(self):
        #Arrange
        client = self.get_client()
        client._download_file = MagicMock()

        #Mock dropbox file metadata API
        download_size = 100
        client._client.files_get_metadata = MagicMock()
        type(client._client.files_get_metadata.return_value).size = PropertyMock(return_value = download_size)
        remote_file_path = constants.DROPBOX_APP_PATH_PREFIX / remote_dir / cnn_model_file

        #Act
        client.download(cnn_model_file)

        #Assert
        args, _ = client._download_file.call_args_list[0]
        self.assertEqual(3, len(args))
        self.assertEqual(cnn_model_file, args[0])
        self.assertEqual(remote_file_path, args[1])
        self.assertEqual(download_size, args[2])

    def test_download_file_path(self):
        #Arrange
        client = self.get_client()
        client._download_file = MagicMock()
        container_name = 'container'

        #Mock dropbox file metadata API
        download_size = 100
        client._client.files_get_metadata = MagicMock()
        type(client._client.files_get_metadata.return_value).size = PropertyMock(return_value = download_size)
        container_relative_path = Path(container_name) / cnn_model_file
        remote_file_path = constants.DROPBOX_APP_PATH_PREFIX / remote_dir / container_relative_path

        #Act
        client.download(container_relative_path)

        #Assert
        args, _ = client._download_file.call_args_list[0]
        self.assertEqual(3, len(args))
        self.assertEqual(container_relative_path, args[0])
        self.assertEqual(remote_file_path, args[1])
        self.assertEqual(download_size, args[2])

    def upload(self):
        #Arrange
        auth_token = None
        client = DropboxConnection.get_client(auth_token, 'test_1')

        #Act
        client.upload(ut_constants.DATA_STORE / cnn_model_file)
        client.upload(ut_constants.DROPBOX_STORE / small_file)

    def download(self):
        #Arrange
        auth_token = None
        client = DropboxConnection.get_client(auth_token, 'test_1')

        #Act
        client.download(small_file)
        client.download(cnn_model_file)

    def test_list_no_files(self):
        #Arrange
        client = DropboxConnection.get_client('auth_token', 'test_1')

        #Arrange response object
        response = MagicMock()
        response.entries = None

        #Mock calls
        client._client.files_list_folder = MagicMock(return_value = response)

        #Act
        files, size = client.list()

        #Assert
        self.assertCountEqual(files, [])

    def test_list(self):
        #Arrange
        client = DropboxConnection.get_client('auth_token', 'test_1')

        #Arrange input files
        input_file_1_info = ('input_file.1.dmp', 100)
        input_file_1 = MagicMock()
        input_file_1.name = input_file_1_info[0]
        input_file_1.size = input_file_1_info[1]

        result_file_1_info = ('result_file.1.dmp', 100)
        result_file_1 = MagicMock()
        result_file_1.name = result_file_1_info[0]
        result_file_1.size = result_file_1_info[1]

        #Arrange response object
        response = MagicMock()
        response.entries = [input_file_1, result_file_1]

        #Mock calls
        client._client.files_list_folder = MagicMock(return_value = response)

        #Act
        files, sizes = client.list(file_name_prefix = 'input_file')

        #Assert
        self.assertCountEqual(list(files), [Path(input_file_1_info[0])])
        self.assertCountEqual(list(sizes), [input_file_1_info[1]])
