"""It defines useful callbacks to track and evaluate training performance.
"""
#Keras imports for callbacks
from keras import backend as K
from keras.callbacks import Callback

#Dropbox
from client.dropbox import DropboxConnection

#Constants
from common import constants

#Path manipulations
from pathlib import Path
from os import path

#Logging
from common import logging

class ModelDropboxCheckpoint(Callback):
    """It creates a model checkpoint and upload it to the dropbox.
    """
    def __init__(self, model_name, input_tuples_batch_id, dropbox_auth = None, dropbox_dir = None):
        """It initializes the parameters.
        
        Arguments:
            model_name {string} -- The name of the model
            input_tuples_batch_id {int} -- The input tuples batch id.
        
        Keyword Arguments:
            dropbox_auth {string} -- The authentication token to access dropbox. (default: {None})
            dropbox_dir {Path} -- The path to destination dropbox folder. (default: {None})
        
        Raises:
            ValueError -- If dropbox_auth is specified without dropbox_dir.
        """
        super(ModelDropboxCheckpoint, self).__init__()

        #Required parameters
        self._model_name = model_name
        self._input_tuples_batch_id = input_tuples_batch_id

        #Additional parameters
        self._dropbox_dir = dropbox_dir

        #Derived parameters
        self._dropbox = None

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
            
    def on_epoch_end(self, epoch, logs = None):
        #Model file name
        model_file = "{}.{}.{}.h5".format(self._model_name, self._input_tuples_batch_id + 1, epoch + 1)

        #Save the trained model.
        self.model.save(model_file)
        self._logger.info('Wrote the model object: %s', model_file)

        #Upload the model
        if self._dropbox:
            self._dropbox.upload(model_file)
            self._logger.info('Uploaded the model to dropbox: %s', model_file)

            #Remove the file after the upload is complete
            Path(model_file).unlink()
            self._logger.info('Deleted the local file: %s', model_file)