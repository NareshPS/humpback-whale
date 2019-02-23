#Unittests
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Callback
from model.callback import ModelDropboxCheckpoint

#Path manipulation
from pathlib import Path

#Test support
from tests.support.utils import get_session_params

#Parameters
model_name = 'model_1'
session_id = 2
set_id = 2
num_sets = 5
dropbox_auth = 'xyz'
dropbox_path = '/root'
epoch = 2

class TestModelDropboxCheckpoint(ut.TestCase):
    def test_init(self):
        #Arrange
        session_params = get_session_params(session_id, set_id, num_sets)

        #Act and Assert
        with self.assertRaises(ValueError):
            ModelDropboxCheckpoint(model_name, session_params, dropbox_auth)

        ModelDropboxCheckpoint(model_name, session_params)
        ModelDropboxCheckpoint(model_name, session_params, dropbox_auth, dropbox_path)

    def on_epoch_end(self, checkpoint, call_dropbox):
        #Arrange
        checkpoint.model = MagicMock()
        model_file = Path('model_1.session_id.2.set_id.2.epoch.3.h5')
        
        if call_dropbox is True:
            checkpoint._dropbox.upload = MagicMock()

            #Act
            with mock_patch.object(Path, 'unlink') as mock_unlink:
                checkpoint.on_epoch_end(epoch)

                #Assert
                mock_unlink.assert_called_once()
        else:
            checkpoint.on_epoch_end(epoch)

        #Assert
        checkpoint.model.save.assert_called_with(str(model_file))

        if call_dropbox:
            checkpoint._dropbox.upload.assert_called_with(model_file)
        else:
            self.assertIsNone(checkpoint._dropbox)

    def test_on_epoch_end_dropbox_called(self):
        #Arrange
        session_params = get_session_params(session_id, set_id, num_sets)
        checkpoint = ModelDropboxCheckpoint(model_name, session_params, dropbox_auth, dropbox_path)

        #Act & Assert
        self.on_epoch_end(checkpoint, True)

    def test_on_epoch_end_dropbox_not_called(self):
        #Arrange
        session_params = get_session_params(session_id, set_id, num_sets)
        checkpoint = ModelDropboxCheckpoint(model_name, session_params)

        #Act & Assert
        self.on_epoch_end(checkpoint, False)