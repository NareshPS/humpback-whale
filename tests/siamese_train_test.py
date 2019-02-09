#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants

#Dropbox connection
from client.dropbox import DropboxConnection

#Siamese train imports
from siamese_train import download_files

files_to_download = ['hello.txt', 'hello1.txt']
auth_token = 'hello.auth.token'
remote_dir_path = 'remote_path'

class TestSiamese_Train(ut.TestCase):
    def test_download_files_invalid_args(self):
        with self.assertRaises(ValueError):
            download_files(None, auth_token, remote_dir_path)

        with self.assertRaises(ValueError):
            download_files([], auth_token, remote_dir_path)

        with self.assertRaises(ValueError):
            download_files(files_to_download, None, remote_dir_path)

        with self.assertRaises(ValueError):
            download_files(files_to_download, auth_token, None)

    def test_download_files(self):
        with mock_patch.object(DropboxConnection, 'download') as mock_download:
            #Act
            download_files(files_to_download, auth_token, remote_dir_path)

            #Assert
            call_args_list = mock_download.call_args_list
            self.assertEqual(len(files_to_download), len(call_args_list))

            for index, call_args in enumerate(call_args_list):
                args, _ = call_args
                self.assertEqual(files_to_download[index], args[0])