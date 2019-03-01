"""It defines useful callbacks to track and evaluate training performance.
"""
#Keras imports for callbacks
from keras import backend as K
from keras.callbacks import Callback

#Dropbox
from client.dropbox import DropboxConnection

#Inputs
from iofiles.input_file import ModelInput, InputDataFile

#Constants
from common import constants

#Logging
from common import logging

class BatchTrainStateCheckpoint(Callback):
    """It creates a model checkpoint and upload it to the dropbox.
    """
    def __init__(
            self,
            batch_input_files = [],
            checkpoint_batch_interval = 1,
            epoch_input_files = [],
            dropbox_auth = None,
            dropbox_dir = None):
        """It initializes the parameters.
        
        Arguments:
            batch_input_files [iofiles.input_files.object] -- The list of input file objects to checkpoint on batch end.
            checkpoint_batch_interval {int} -- The number of batches after which to upload checkpoint the files.
            epoch_input_files [iofiles.input_files.object] -- The list of input file objects to checkpoint on epoch end.

        Keyword Arguments:
            dropbox_auth {string} -- The authentication token to access dropbox. (default: {None})
            dropbox_dir {pathlib.Path} -- The path to destination dropbox folder. (default: {None})
        
        Raises:
            ValueError -- If dropbox_auth is specified without dropbox_dir.
        """
        super(BatchTrainStateCheckpoint, self).__init__()

        #Required parameters
        self._batch_input_files = batch_input_files
        self._checkpoint_batch_interval = checkpoint_batch_interval
        self._epoch_input_files = epoch_input_files

        #Additional parameters
        self._dropbox_dir = dropbox_dir

        #Derived parameters
        self._dropbox = None
        self._model = None
        self._input_data = None
        self._batch_id = 0
        self._epoch_id = 0

        #Validation
        if dropbox_auth is not None and self._dropbox_dir is None:
            raise ValueError("No dropbox dir provided.")

        #Dropbox client
        if dropbox_auth:
            self._dropbox = DropboxConnection(dropbox_auth, self._dropbox_dir)

        #Logging
        self._logger = logging.get_logger(__name__)

        #Save TF session
        self._tf_session = K.get_session()

    def set_model(self, model):
        """It updates the model object.

        Arguments:
            model {keras.Model} -- The model object to be set.
        """
        self._model = model

    def set_input_data(self, input_data):
        """It updates the current input data.

        Arguments:
            input_data {pandas.DataFrame} -- It is the input training data.
        """
        self._input_data = input_data

    def _upload(self, input_files, batch_id, epoch_id):
        """It generates the checkpoint files locally, then uploads them to dropbox.
        
        Arguments:
            input_files [iofiles.input_files.object] -- The list of input file objects to checkpoint.
            batch_id {int} -- The id of the current batch.
            epoch_id {int} -- The id of the current epoch.
        """
        self._logger.info('Got %d files to upload to dropbox', len(input_files))
        
        #Iterate over to generate input files.
        for input_file in input_files:
            #Save file locally
            if isinstance(input_file, ModelInput):
                input_file.save(self._model, batch_id, epoch_id)
            elif isinstance(input_file, InputDataFile):
                input_file.save(self._input_data, batch_id, epoch_id)

        if self._dropbox:
            #Upload the input files
            for input_file in input_files:
                file_name = input_file.file_name(batch_id, epoch_id)

                #Start the upload
                self._dropbox.upload(file_name)
                self._logger.info('Uploaded the model to dropbox: %s', file_name)

                #Remove the file after the upload is complete
                file_name.unlink()
                self._logger.info('Deleted the local file: %s', file_name)

    def on_batch_begin(self, batch_id, logs = None):
        """It execute the required batch start operations
        
        Arguments:
            batch_id {int} -- The id of the current batch.
            logs {dict} -- A dictionary of metrics.
        """
        self._logger.info('on_batch_begin:: batch_id: %d', batch_id)

        #Save the last batch id
        self._batch_id = batch_id

    def on_batch_end(self, batch_id, logs = None):
        """It execute the required batch end operations
        
        Arguments:
            batch_id {int} -- The id of the current batch.
            logs {dict} -- A dictionary of metrics.
        """
        self._logger.info('on_batch_end:: batch_id: %d', batch_id)

        #Checkpoint the state if required
        if (batch_id + 1) % self._checkpoint_batch_interval == 0:
            self._upload(self._batch_input_files, batch_id, self._epoch_id)
 
    def on_epoch_begin(self, epoch_id, logs = None):
        """It execute the required epoch start operations
        
        Arguments:
            epoch_id {int} -- The id of the current batch.
            logs {dict} -- A dictionary of metrics.
        """
        self._logger.info('on_epoch_begin:: epoch_id: %d', epoch_id)

        #Save the last epoch id
        self._epoch_id = epoch_id

    def on_epoch_end(self, epoch_id, logs = None):
        """It execute the required epoch end operations
        
        Arguments:
            epoch_id {int} -- The id of the current batch.
            logs {dict} -- A dictionary of metrics.
        """
        self._logger.info('on_epoch_end:: epoch_id: %d', epoch_id)

        #Initiate the download
        self._upload(self._epoch_input_files, self._batch_id, epoch_id)