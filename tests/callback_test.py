#Unittests
import unittest as ut
from unittest.mock import MagicMock

#Callback
from model.callback import ModelDropboxCheckpoint

#Parameters
model_name = 'model_1'
dropbox_auth = 'xyz'
dropbox_path = '/root'
epoch = 2
model_file = "{}.{}.h5".format(model_name, epoch)

class TestModelDropboxCheckpoint(ut.TestCase):
    def test_init(self):
        #Act and Assert
        with self.assertRaises(ValueError):
            ModelDropboxCheckpoint(model_name, dropbox_auth)

        ModelDropboxCheckpoint(model_name)
        ModelDropboxCheckpoint(model_name, dropbox_auth, dropbox_path)

    def on_epoch_end(self, checkpoint, call_dropbox):
        #Arrange
        checkpoint.model = MagicMock()
        checkpoint._upload = MagicMock()

        #Act
        checkpoint.on_epoch_end(epoch)

        #Assert
        checkpoint.model.save.assert_called_with(model_file)

        if call_dropbox:
            checkpoint._upload.assert_called_with(model_file)
        else:
            checkpoint._upload.assert_not_called()


    def test_on_epoch_end_dropbox_called(self):
        #Arrange
        checkpoint = ModelDropboxCheckpoint(model_name, dropbox_auth, dropbox_path)

        #Act & Assert
        self.on_epoch_end(checkpoint, True)

    def test_on_epoch_end_dropbox_not_called(self):
        #Arrange
        checkpoint = ModelDropboxCheckpoint(model_name)

        #Act & Assert
        self.on_epoch_end(checkpoint, False)