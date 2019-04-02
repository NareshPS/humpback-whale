#Dropbox
from client.dropbox import DropboxConnection

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

#Path manipulations
from pathlib import Path

#Pickle
from pickle import dump as pickle_dump

class ModelInput(object):
    def __init__(self, model_name):
        """It initializes the parameters.

        Arguments:
            model_name {string} -- The name of the model
        """
        #Required parameters
        self._model_name = model_name

        #Validation
        if model_name is None:
            raise ValueError('model_name must be valid')

    def save(self, model, batch_id, epoch_id):
        """It saves the model object to the disk.

        Arguments:
            model {keras.Model} -- The model object to be set.
            batch_id {int} -- The id of the current batch.
            epoch_id {int} -- The id of the current epoch.
        """
        model.save(str(self.file_name(batch_id, epoch_id)))

    def file_name(self, batch_id, epoch_id):
        """It creates the file name for the current iteration.

        Arguments:
            batch_id {int} -- The id of the current batch.
            epoch_id {int} -- The id of the current epoch.

        Returns:
            {Path} -- The name of the model file
        """
        model_file_name = Path("{}.batch.{}.epoch.{}.h5".format(self._model_name, batch_id, epoch_id))

        return model_file_name

class InputDataFile(object):
    def __init__(self, name = 'input_data'):
        #Required parameters
        self._name = name

    def save(self, input_data, batch_id, epoch_id):
        input_data.to_csv(self.file_name(0, epoch_id))

    def file_name(self, batch_id, epoch_id):
        """It creates the file name for the current iteration.

        Arguments:
            batch_id {int} -- The id of the current batch.
            epoch_id {int} -- The id of the current epoch.

        Returns:
            {Path} -- The name of the model file
        """
        input_file_name = Path("{}.batch.{}.epoch.{}.csv".format(self._name, 0, epoch_id))

        return input_file_name

class ResultFile(object):
    def __init__(self, name = 'result'):
        self._name = name

    def save(self, result, batch_id, epoch_id):
        with self.file_name(batch_id, epoch_id).open(mode = 'wb') as handle:
            pickle_dump(result, handle)

    def file_name(self, batch_id, epoch_id):
        """It creates the file name for the current iteration.

        Arguments:
            batch_id {int} -- The id of the current batch.
            epoch_id {int} -- The id of the current epoch.

        Returns:
            {Path} -- The name of the result file
        """
        result_file_name = Path("{}.batch.{}.epoch.{}.dmp".format(self._name, batch_id, epoch_id))

        return result_file_name

class InputFiles(object):
    def __init__(self, dropbox):
        """It initializes and validates the input parameters.

        Arguments:
            dropbox {client.dropbox.DropboxConnection} -- The dropbox client.
        """

        #Keyword parameters
        self._dropbox = dropbox

        #Logging
        self._logger = logging.get_logger(__name__)

    def get_all(self, file_paths):
        """It downloads the files in file_paths if they are not locally available.

        Arguments:
            file_paths {[pathlib.Path]} -- The list of file paths
        """
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

    def put_all(self, file_paths):
        """It uploads the file paths to the dropbox

        Arguments:
            file_paths {[pathlib.Path]} -- The list of file paths
        """
        for file_path in file_paths:
            self._dropbox.upload(file_path)

    def _download(self, file_paths):
        """It downloads the files in file_paths.

        Arguments:
            file_paths {[pathlib.Path]} -- The list of file paths
        """
        #Result dictionary
        valid_file_paths = {file_path : file_path for file_path in file_paths}

        #Download files
        for file_path in valid_file_paths.values():
            self._dropbox.download(file_path)

            #Trace download completion
            self._logger.info('Downloaded: %s', file_path)

        return valid_file_paths
