#Unittests
import unittest as ut
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

#Callback
from model.callback import BatchTrainStateCheckpoint

#Path manipulation
from pathlib import Path

#Test support
from tests.support.utils import load_test_model, get_input_data

#Input files
from iofiles.input_file import ModelInput, InputDataFile, ResultFile

#Dropbox
from client.dropbox import DropboxConnection

#Parameters
model_name = 'model_1'
model_name_1 = 'model_1_1'
batch_id = 2
epoch_id = 3
checkpoint_batch_interval = 2

def get_dropbox(token = 'remote_auth_token', dir_path = Path('.')):
    dropbox_params = DropboxConnection.Parameters(token, dir_path)
    dropbox = DropboxConnection(dropbox_params)

    return dropbox

def get_checkpoint(dropbox = None, checkpoint_batch_interval = checkpoint_batch_interval):
    #Arrange
    batch_input = ModelInput(model_name)
    input_data_file = InputDataFile()
    result_file = ResultFile()
    batch_input_files = [batch_input, result_file]
    epoch_input_files = [input_data_file]

    checkpoint = BatchTrainStateCheckpoint(
                        batch_input_files = batch_input_files,
                        checkpoint_batch_interval = checkpoint_batch_interval,
                        epoch_input_files = epoch_input_files,
                        dropbox = dropbox)

    return checkpoint, batch_input_files, epoch_input_files

class TestBatchTrainStateCheckpoint(ut.TestCase):
    def test_init(self):
        #Arrange
        checkpoint, batch_input_files, epoch_input_files = get_checkpoint(dropbox = get_dropbox())
        
        #Assert
        self.assertEqual(checkpoint._batch_id, 0)
        self.assertEqual(checkpoint._epoch_id, 0)
        self.assertEqual(checkpoint._batch_input_files, batch_input_files)
        self.assertEqual(checkpoint._epoch_input_files, epoch_input_files)
        self.assertEqual(checkpoint._checkpoint_batch_interval, checkpoint_batch_interval)
        self.assertIsNotNone(checkpoint._dropbox)
        self.assertIsNone(checkpoint._model)
        self.assertIsNone(checkpoint._input_data)
        self.assertIsNone(checkpoint._result)
        self.assertIsNone(checkpoint._epoch_response)

    def test_set_model(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        model = load_test_model()

        #Act
        checkpoint.set_model(model)

        #Assert
        self.assertIsNotNone(checkpoint._model)

    def test_set_result(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        result = 3

        #Act
        checkpoint.set_result(result)

        #Assert
        self.assertIsNotNone(checkpoint._result)

    def test_set_input_data(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        input_data = get_input_data()

        #Act
        checkpoint.set_input_data(input_data)

        #Assert
        self.assertIsNotNone(checkpoint._input_data)
        self.assertEqual(len(input_data), len(checkpoint._input_data))

    def test_on_epoch_begin(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        checkpoint._model = MagicMock()

        #Act
        checkpoint.on_epoch_begin(epoch_id)

        #Assert
        self.assertEqual(epoch_id, checkpoint._epoch_id)
        self.assertIsNotNone(checkpoint._epoch_response)
        self.assertEqual(epoch_id, checkpoint._epoch_response._epoch_id)

    def on_epoch_end(self, checkpoint, call_dropbox):
        #Arrange
        input_data_file = Path('input_data.batch.0.epoch.3.csv')

        #Mocks
        checkpoint._model.save = MagicMock()
        checkpoint._input_data.to_csv = MagicMock()
        
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
        checkpoint._model.save.assert_not_called()
        checkpoint._input_data.to_csv.assert_called_with(input_data_file)

        if call_dropbox:
            checkpoint._dropbox.upload.assert_called_with(input_data_file)
        else:
            self.assertIsNone(checkpoint._dropbox)

    def test_on_epoch_end_dropbox_called(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint(dropbox = get_dropbox())
        model = load_test_model()
        input_data = get_input_data()
        checkpoint.set_model(model)
        checkpoint.set_input_data(input_data)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_epoch_end(checkpoint, True)

    def test_on_epoch_end_dropbox_not_called(self):
        #Arrange
        checkpoint, _, _ = get_checkpoint()
        model = load_test_model()
        input_data = get_input_data()
        checkpoint.set_model(model)
        checkpoint.set_input_data(input_data)
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

    def on_batch_end_no_dropbox(self, checkpoint, batch_id, batch_input_files, save_called = True):
        #Arrange
        model_file = Path('model_1.batch.{}.epoch.3.h5'.format(batch_id))
        checkpoint.set_result([0.5, .2])

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
        self.assertEqual(1, len(checkpoint._epoch_response._batch_ids))
        self.assertEqual(batch_id, checkpoint._epoch_response._batch_ids[0])

    def on_batch_end_with_dropbox(self, checkpoint, batch_id, batch_input_files):
        #Arrange
        model_file = Path('model_1.batch.{}.epoch.3.h5'.format(batch_id))
        result_file = Path('result.batch.{}.epoch.3.dmp'.format(batch_id))
        checkpoint.set_result([0.5, .2])

        #Mocks
        checkpoint._model.save = MagicMock()
        checkpoint._dropbox.upload = MagicMock()
        
        #Act
        with mock_patch.object(Path, 'unlink') as mock_unlink:
            checkpoint.on_batch_end(batch_id)

            #Assert
            call_args_list = mock_unlink.call_args_list
            self.assertEqual(len(call_args_list), len(batch_input_files))

        #Assert
        arg_tuples = [args[0] for args in checkpoint._dropbox.upload.call_args_list]
        upload_list = [item[0] for item in arg_tuples]
        self.assertCountEqual([model_file, result_file], upload_list)
        self.assertEqual(1, len(checkpoint._epoch_response._batch_ids))
        self.assertEqual(batch_id, checkpoint._epoch_response._batch_ids[0])

    def test_on_batch_end_save_called(self):
        #Arrange
        batch_id = 1
        checkpoint, batch_input_files, _ = get_checkpoint()
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_no_dropbox(checkpoint, batch_id, batch_input_files)

    def test_on_batch_end_save_called_with_dropbox(self):
        #Arrange
        batch_id = 1
        checkpoint, batch_input_files, _ = get_checkpoint(dropbox = get_dropbox())
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_with_dropbox(checkpoint, batch_id, batch_input_files)

    def test_on_batch_end_save_not_called(self):
        #Arrange
        checkpoint, batch_input_files, _ = get_checkpoint()
        model = load_test_model()
        checkpoint.set_model(model)
        checkpoint.on_batch_begin(batch_id)
        checkpoint.on_epoch_begin(epoch_id)

        #Act & Assert
        self.on_batch_end_no_dropbox(checkpoint, batch_id, batch_input_files, save_called = False)