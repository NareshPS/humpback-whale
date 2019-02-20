#Dropbox
from client.dropbox import DropboxConnection

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

#Path manipulations
from pathlib import Path

class ModelInput(object):
    def __init__(self, model_name, session_id, set_id, total_number_of_sets):
        #Required parameters
        self._model_name = model_name
        self._session_id = session_id
        self._set_id = set_id
        self._total_number_of_sets = total_number_of_sets

        #Validation
        if model_name is None:
            raise ValueError('model_name must be valid')
    
        if self._session_id < 1:
            raise ValueError('session_id must be > 1')

        if self._set_id < 1:
            raise ValueError('set_id must be > 1')

        if self._total_number_of_sets < 1:
            raise ValueError('total_number_of_sets must be >= 1')

        if self._set_id > self._total_number_of_sets:
            raise ValueError('set_id: {} must be <= total_number_of_sets: {}'.format(self._set_id, self._total_number_of_sets))

    def last_saved_file_name(self):
        """It creates the last saved model file name.
        
        Returns:
            {string} -- The name of the last saved model file.
        """

        #Last saved placeholders
        last_saved_session_id = 0
        last_saved_set_id = 0

        if self._set_id == 1:
            #For the first set, the last saved state is the last set of previous iteration
            if self._session_id != 1:
                last_saved_set_id = self._total_number_of_sets

            #For the first set, the last saved iteration is the last iteration.
            last_saved_session_id = self._session_id - 1
        #Normal case
        else:
            last_saved_session_id = self._session_id
            last_saved_set_id = self._set_id - 1

        model_file_name = Path("{}.session_id.{}.set_id.{}.epoch.{}.h5".format(
                                                                        self._model_name,
                                                                        last_saved_session_id,
                                                                        last_saved_set_id,
                                                                        1))

        return model_file_name

    def file_name(self):
        """It creates the file name for the current iteration.
        
        Returns:
            {string} -- The name of the model file for the current iteration.
        """

        model_file_name = Path("{}.session_id.{}.set_id.{}.epoch.{}.h5".format(self._model_name, self._session_id, self._set_id, 1))

        return model_file_name

class InputFiles(object):
    def __init__(self, remote_auth_token = None, remote_dir_path = None):
        """It initializes and validates the input parameters.
        
        Arguments:
            remote_auth_token {string} -- The authentication token to access the remote store.
            remote_dir_path {A Path object} -- The path to the remote store.
        
        Raises:
            ValueError -- If remote_auth_token in invalid.
            ValueError -- If remote_dir_path is invalid.
        """

        #Keyword parameters
        self._remote_auth_token = remote_auth_token
        self._remote_dir_path = remote_dir_path

        #Validation
        if self._remote_auth_token and self._remote_dir_path is None:
            raise ValueError('remote_dir_path must be valid.')

        #Derived parameters
        self._dropbox = None
        
        if self._remote_auth_token:
            self._dropbox = DropboxConnection(self._remote_auth_token, self._remote_dir_path)

        #Logging
        self._logger = logging.get_logger(__name__)

    def get_all(self, file_paths):
        #Categorize files into two lists: those that are locally available and the ones that needs to be downloaded.  
        local_files = [file_path for file_path in file_paths if file_path.exists()]
        remote_files_to_download = [file_path for file_path in file_paths if file_path not in local_files]

        self._logger.info('Local files: %s', local_files)
        self._logger.info('Download candidates: %s', remote_files_to_download)

        #Results mapping from original file to their new paths
        valid_file_paths = {}

        #Add the locally available files
        valid_file_paths.update({local_file : local_file for local_file in local_files})

        if remote_files_to_download:
            if self._dropbox is None:
                raise ValueError('Files to download: {}. Remote client is not initialized.'.format(remote_files_to_download))

            #Download remote files
            valid_remote_file_paths = self._download(remote_files_to_download)

            #Update the result mapping
            valid_file_paths.update(valid_remote_file_paths)

        return valid_file_paths

    def _download(self, file_paths):
        #Result dictionary
        valid_file_paths = {file_path : Path(file_path.name) for file_path in file_paths}

        #Download files
        for file_name in valid_file_paths.values():
            self._dropbox.download(file_name)

            #Trace download completion
            self._logger.info('Downloaded: %s', file_name)

        return valid_file_paths