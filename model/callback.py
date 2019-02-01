"""It defines useful callbacks to track and evaluate training performance.
"""
#Keras imports for callbacks
from keras import backend as K
from keras.callbacks import Callback

#Dropbox
from dropbox import Dropbox
from dropbox.files import UploadSessionCursor as Dropbox_UploadSessionCursor
from dropbox.files import CommitInfo as Dropbox_CommitInfo

#Constants
from common import constants

#Path manipulations
from pathlib import Path
from os import path

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

class ModelDropboxCheckpoint(Callback):
    """It creates a model checkpoint and upload it to the dropbox.
    """
    def __init__(self, model_name, dropbox_auth = None, dropbox_dir = None):
        super(ModelDropboxCheckpoint, self).__init__()

        #Required parameters
        self._model_name = model_name

        #Additional parameters
        self._dropbox_dir = dropbox_dir

        #Validation
        if dropbox_auth is not None and self._dropbox_dir is None:
            raise ValueError("No dropbox dir provided.")

        #Dropbox client
        self._dropbox = Dropbox(dropbox_auth)

        #Logging
        self._logger = logging.get_logger(__name__)

        #Save TF session
        self._tf_session = K.get_session()
            
    def on_epoch_end(self, epoch, logs = None):
        #Model file name
        model_file = "{}.{}.h5".format(self._model_name, epoch)

        #Save the trained model.
        self.model.save(model_file)
        self._logger.info('Wrote the model object: %s', model_file)

        #Upload the model
        self._upload(model_file)
        self._logger.info('Uploaded the model to dropbox: %s', model_file)
        
    def _upload(self, filename):
        with open(filename, 'rb') as handle:
            #Target path
            dropbox_path = (Path(self._dropbox_dir) / filename).as_posix()

            #File size
            upload_size = path.getsize(filename)

            #Upload session
            session = self._dropbox.files_upload_session_start(handle.read(constants.DROPBOX_CHUNK_SIZE))
            cursor = Dropbox_UploadSessionCursor(session_id = session.session_id, offset = handle.tell())
            commit = Dropbox_CommitInfo(path = dropbox_path)

            #Upload look
            with tqdm(desc = 'Uploading file: {}'.format(dropbox_path), total = upload_size) as pbar:
                #Update the progress bar for the session start reads
                pbar.update(handle.tell())

                while handle.tell() < upload_size:
                    #Calculate remaining bytes
                    remaining_bytes = upload_size - handle.tell()

                    #If it is the last chunk, finalize the upload
                    if remaining_bytes <= constants.DROPBOX_CHUNK_SIZE:
                        self._dropbox.files_upload_session_finish(
                                        handle.read(remaining_bytes),
                                        cursor,
                                        commit)

                        #Update progress
                        pbar.update(remaining_bytes)
                    #More than chunk size remaining to upload
                    else:
                        self._dropbox.files_upload_session_append_v2(
                                        handle.read(constants.DROPBOX_CHUNK_SIZE),
                                        cursor,
                                        close = True)
                        
                        #Update the cursor
                        cursor.offset = handle.tell()

                        #Update the progress
                        pbar.update(constants.DROPBOX_CHUNK_SIZE)