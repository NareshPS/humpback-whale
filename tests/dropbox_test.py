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
remote_dir_path = Path('/remote/file/path')
auth_token = 'hello i am called a token'
small_file = 'small_file.txt'
cnn_model_file = 'cnn_model2d_1.h5'

class TestDropboxConnection(ut.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDropboxConnection, self).__init__(*args, **kwargs)

        #Initialize logging
        logging.initialize('unittest')

    def test_init(self):
        with self.assertRaises(ValueError):
            _ = DropboxConnection(None, remote_dir_path)

        with self.assertRaises(ValueError):
            _ = DropboxConnection(auth_token, None)

        #Success
        _ = DropboxConnection(auth_token, remote_dir_path)

    def get_client(self):
        return DropboxConnection(auth_token, remote_dir_path)

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
        remote_file_path = Path(remote_dir_path) / small_file
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
        remote_file_path = Path(remote_dir_path) / cnn_model_file
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
        remote_file_path = remote_dir_path / cnn_model_file

        #Act
        client.download(cnn_model_file)

        #Assert
        args, _ = client._download_file.call_args_list[0]
        self.assertEqual(3, len(args))
        self.assertEqual(cnn_model_file, args[0])
        self.assertEqual(remote_file_path, args[1])
        self.assertEqual(download_size, args[2])

    def upload(self):
        #Arrange
        auth_token = None
        client = DropboxConnection(auth_token, 'test_1')

        #Act
        client.upload(ut_constants.DATA_STORE / cnn_model_file)
        client.upload(ut_constants.DROPBOX_STORE / small_file)

    def download(self):
        #Arrange
        auth_token = None
        client = DropboxConnection(auth_token, 'test_1')

        #Act
        client.download(small_file)
        client.download(cnn_model_file)