#Unittest
import unittest as ut
from unittest.mock import patch as mock_patch

#Constants
from common import ut_constants

#Dropbox connection
from client.dropbox import DropboxConnection

files_to_download = ['hello.txt', 'hello1.txt']
auth_token = 'hello.auth.token'
remote_dir_path = 'remote_path'

class TestSiamese_Train(ut.TestCase):
    pass