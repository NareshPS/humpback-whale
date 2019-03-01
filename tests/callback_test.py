#Unittests
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Callback
from model.callback import BatchTrainStateCheckpoint

#Path manipulation
from pathlib import Path

#Test support
from tests.support.utils import load_test_model

#Input files
from iofiles.input_file import ModelInput

#Dropbox
from client.dropbox import DropboxConnection

#Parameters
model_name = 'model_1'
model_name_1 = 'model_1_1'
batch_id = 2
epoch_id = 3
dropbox_auth = 'xyz'
dropbox_path = '/root'
checkpoint_batch_interval = 2

def get_checkpoint(
            dropbox_auth = dropbox_auth,
            dropbox_dir = dropbox_path,
            checkpoint_batch_interval = checkpoint_batch_interval):
    #Arrange
    batch_input = ModelInput(model_name)
    epoch_input = ModelInput(model_name_1)
    batch_input_files = [batch_input]
    epoch_input_files = [epoch_input]

    checkpoint = BatchTrainStateCheckpoint(
                        batch_input_files = batch_input_files,
                        checkpoint_batch_interval = checkpoint_batch_interval,
                        epoch_input_files = epoch_input_files,
                        dropbox_auth = dropbox_auth,
                        dropbox_dir = dropbox_path)

    return checkpoint, batch_input_files, epoch_input_files

class TestBatchTrainStateCheckpoint(ut.TestCase):
    def test_init_invalid_args(self):
        #Act and Assert
        with self.assertRaises(ValueError):
            BatchTrainStateCheckpoint(dropbox_auth = dropbox_auth)

    def init(self, dropbox_auth, dropbox_dir, dropbox_connection = True):
        #Arrange
        checkpoint, batch_input_files, epoch_input_files = get_checkpoint(
                                                                dropbox_auth = dropbox_auth,
                                                                dropbox_dir = dropbox_dir)
        
        #Assert
        self.assertEqual(checkpoint._batch_id, 0)
        self.assertEqual(checkpoint._epoch_id, 0)
        self.assertEqual(checkpoint._batch_input_files, batch_input_files)
        self.assertEqual(checkpoint._epoch_input_files, epoch_input_files)
        self.assertEqual(checkpoint._checkpoint_batch_interval, checkpoint_batch_interval)
        self.assertEqual(checkpoint._dropbox_dir, dropbox_path)
        self.assertIsNone(checkpoint._model)

        if dropbox_connection:
            self.assertIsNotNone(checkpoint._dropbox)
            self.assertTrue(isinstance(checkpoint._dropbox, DropboxConnection))
        else:
            self.assertIsNone(checkpoint._dropbox)

    def test_init_dropbox(self):
        self.init(dropbox_auth, dropbox_path)

    def test_init_no_dropbox(self):
        self.init(None, None, dropbox_connection = False)

    def test_set_model(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        model = load_test_model()

        #Act
        checkpoint.set_model(model)

        #Assert
        self.assertIsNotNone(checkpoint._model)

    def test_on_epoch_begin(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()

        #Act
        checkpoint.on_epoch_begin(epoch_id)

        #Assert
        self.assertEqual(epoch_id, checkpoint._epoch_id)

    def on_epoch_end(self, checkpoint, call_dropbox):
        #Arrange
        model_1_file = Path('model_1_1.batch.2.epoch.3.h5')

        #Mocks
        checkpoint._model.save = MagicMock()
        
        #Act
        if call_dropbox:
            checkpoint._dropbox.upload = MagicMock()

            #Act
            with mock_patch.object(Path, 'unlink') as mock_unlink:
                checkpoint.on_epoch_end(epoch_id)

                #Assert
                mock_unlink.assert_called_once()
        else:
            checkpoint.on_epoch_end(epoch_id)

        #Assert
        checkpoint._model.save.assert_called_with(str(model_1_file))

        if call_dropbox:
            checkpoint._dropbox.upload.assert_called_with(model_1_file)
        else:
            self.assertIsNone(checkpoint._dropbox)

    def test_on_epoch_end_dropbox_called(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_epoch_end(checkpoint, True)

    def test_on_epoch_end_dropbox_not_called(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint(dropbox_auth = None, dropbox_dir = None)
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_epoch_end(checkpoint, False)

    def test_on_batch_begin(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()

        #Act
        checkpoint.on_batch_begin(batch_id)

        #Assert
        self.assertEqual(batch_id, checkpoint._batch_id)

    def on_batch_end_no_dropbox(self, checkpoint, batch_id, save_called = True):
        #Arrange
        model_file = Path('model_1.batch.{}.epoch.3.h5'.format(batch_id))

        #Mocks
        checkpoint._model.save = MagicMock()
        
        #Act
        checkpoint.on_batch_end(batch_id)

        #Assert
        if save_called:
            checkpoint._model.save.assert_called_with(str(model_file))
        else:
            checkpoint._model.save.assert_not_called()

        self.assertIsNone(checkpoint._dropbox)

    def on_batch_end_with_dropbox(self, checkpoint, batch_id):
        #Arrange
        model_file = Path('model_1.batch.{}.epoch.3.h5'.format(batch_id))

        #Mocks
        checkpoint._model.save = MagicMock()
        checkpoint._dropbox.upload = MagicMock()
        
        #Act
        with mock_patch.object(Path, 'unlink') as mock_unlink:
            checkpoint.on_batch_end(batch_id)

            #Assert
            mock_unlink.assert_called_once()

        #Assert
        checkpoint._dropbox.upload.assert_called_with(model_file)

    def test_on_batch_end_save_called(self):
        #Arrange
        batch_id = 1
        checkpoint, _, _ = get_checkpoint(dropbox_auth = None, dropbox_dir = None)
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_no_dropbox(checkpoint, batch_id)

    def test_on_batch_end_save_called_with_dropbox(self):
        #Arrange
        batch_id = 1
        checkpoint, _, _ = get_checkpoint()
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_with_dropbox(checkpoint, batch_id)

    def test_on_batch_end_save_not_called(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint(dropbox_auth = None, dropbox_dir = None)
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_no_dropbox(checkpoint, batch_id, save_called = False)