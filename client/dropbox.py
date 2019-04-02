"""It allows access, upload and download of the dropbox data.
"""
#Dropbox
from dropbox import Dropbox
from dropbox.files import UploadSessionCursor as Dropbox_UploadSessionCursor
from dropbox.files import CommitInfo as Dropbox_CommitInfo
from dropbox.files import WriteMode as Dropbox_WriteMode
from dropbox.exceptions import ApiError

#Constants
from common import constants

#Path manipulations
from pathlib import Path
from os import path
from os import rename

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

class DropboxConnection(object):
    class Parameters(object):
        def __init__(self, auth_token, remote_dir_path):
            self._auth_token = auth_token
            self._remote_dir_path = remote_dir_path

        @property
        def auth_token(self):
            return self._auth_token

        @property
        def remote_dir_path(self):
            return constants.DROPBOX_APP_PATH_PREFIX / self._remote_dir_path

        def __str__(self):
            return """
                        remote_dir_path: {}
                   """.format(self.remote_dir_path)

    @classmethod
    def get_client_from_params(cls, dropbox_parameters):
        #Dropbox connection placeholder
        dropbox = None

        if dropbox_parameters:
            dropbox_params = DropboxConnection.Parameters(dropbox_parameters[0], dropbox_parameters[1])
            dropbox = DropboxConnection(dropbox_params)

        return dropbox

    @classmethod
    def get_client(cls, auth_token, remote_dir_path):
        #Initialize the paramters
        params = DropboxConnection.Parameters(auth_token, remote_dir_path)

        #Create dropbox client
        client = DropboxConnection(params)

        return client

    def __init__(self, params):
        #Required parameters
        self._params = params

        #Derived parameters
        self._client = Dropbox(self._params.auth_token)

        #Logging
        self._logger = logging.get_logger(__name__)

    def upload(self, source_file_path):
        """It upload the source files to the dropbox.

        Arguments:
            source_file_path {string} -- The source file path.

        Raises:
            ValueError -- It raise value error for invalid files.
        """

        #Validate parameters
        if not source_file_path:
            raise ValueError("Invalid source file path")

        with open(source_file_path, 'rb') as handle:
            #Source file name
            source_file_name = Path(source_file_path).name

            #Remote path
            remote_file_path = (self._params.remote_dir_path / source_file_name).as_posix()

            #File size
            upload_size = path.getsize(source_file_path)

            #Upload the files based on the upload size
            if upload_size <= constants.DROPBOX_CHUNK_SIZE:
                self._logger.info(
                                'Preparing to upload small file: %s with size: %d to: %s',
                                source_file_path,
                                upload_size,
                                remote_file_path)

                self._upload_small_file(handle, remote_file_path)
            else:
                self._logger.info(
                                'Preparing to upload large file: %s with size: %d to: %s',
                                source_file_path,
                                upload_size,
                                remote_file_path)

                self._upload_large_file(handle, upload_size, remote_file_path)

            self._logger.info('Uploaded: %s', source_file_path)

    def _upload_small_file(self, handle, remote_file_path):
        """It uploads a small source files to the dropbox.

        Arguments:
            handle {A File handle} -- The source file handle.
            remote_file_path {string} -- The destination path of the file.
        """
        self._client.files_upload(handle.read(), remote_file_path, mode = Dropbox_WriteMode.overwrite)

    def _upload_large_file(self, handle, upload_size, remote_file_path):
        """It uploads a large source files to the dropbox.

        Arguments:
            handle {A File handle} -- The source file handle.
            upload_size {int} -- The number of bytes to be uploaded.
            remote_file_path {string} -- The destination path of the file.
        """
        #Upload session
        session = self._client.files_upload_session_start(handle.read(constants.DROPBOX_CHUNK_SIZE))
        cursor = Dropbox_UploadSessionCursor(session_id = session.session_id, offset = handle.tell())

        #Upload look
        with tqdm(desc = 'Uploading: {}'.format(remote_file_path), total = upload_size) as pbar:
            #Update the progress bar for the session start reads
            pbar.update(handle.tell())

            while handle.tell() < upload_size:
                #Calculate remaining bytes
                remaining_bytes = upload_size - handle.tell()

                #If it is the last chunk, finalize the upload
                if remaining_bytes <= constants.DROPBOX_CHUNK_SIZE:
                    #Commit info
                    commit = Dropbox_CommitInfo(path = remote_file_path, mode = Dropbox_WriteMode.overwrite)

                    #Finish upload
                    self._client.files_upload_session_finish(
                                    handle.read(remaining_bytes),
                                    cursor,
                                    commit)

                    #Update progress
                    pbar.update(remaining_bytes)
                #More than chunk size remaining to upload
                else:
                    self._client.files_upload_session_append_v2(
                                    handle.read(constants.DROPBOX_CHUNK_SIZE),
                                    cursor)

                    #Update the cursor
                    cursor.offset = handle.tell()

                    #Update the progress
                    pbar.update(constants.DROPBOX_CHUNK_SIZE)

                #Refresh the progress bar
                pbar.refresh()

    def download(self, remote_file_path):
        """It downloads the remote files from the dropbox.

        Arguments:
            remote_file_path {Path} -- The path to the remote file.

        Raises:
            ValueError -- It raise value error for invalid file name.
        """
        #Validate parameters
        if not remote_file_path:
            raise ValueError("Invalid remote file path")

        #Destination file path
        dest_file_path = remote_file_path

        #Full remote file path
        remote_file_path = self._params.remote_dir_path / remote_file_path

        #Download file size placeholder
        download_size = 0

        try:
            download_size = self._client.files_get_metadata(remote_file_path.as_posix()).size
        except ApiError as e:
            raise FileNotFoundError('File: {} is not found'.format(remote_file_path.as_posix()), e)


        self._logger.info('Preparing file download: %s with size: %d to: %s', remote_file_path, download_size, dest_file_path)

        #Download the file
        self._download_file(dest_file_path, remote_file_path, download_size)

        self._logger.info('Completed the file download: %s to: %s', remote_file_path, dest_file_path)

    def _download_file(self, dest_file_path, remote_file_path, download_size):
        """It downloads the remote files from the dropbox.

        Arguments:
            remote_file_path {A Path object} -- The path of the remote file.
            dest_file_path {string} -- The destination file path.
            download_size {int} -- The number of bytes to be downloaded.
        """
        #Download
        _, result = self._client.files_download(remote_file_path.as_posix())

        #Temporary dest_file_name
        tmp_dest_file_path = "{}.tmp".format(dest_file_path)

        with open(tmp_dest_file_path, 'wb') as handle:
            with tqdm(desc = 'Downloading: {}'.format(remote_file_path.as_posix()), total = download_size) as pbar:
                for bytes_read in result.iter_content(constants.DROPBOX_CHUNK_SIZE):
                    handle.write(bytes_read)

                    #Update the progress
                    pbar.update(len(bytes_read))

        if Path(tmp_dest_file_path).exists():
            rename(tmp_dest_file_path, dest_file_path)

    def list(self, dir_path = Path(), file_name_prefix = ''):
        """It lists the files in the dropbox folder that starts with the given prefix.

        Arguments:
            file_name_prefix {string} -- The prefix to filter the results.
        """
        #Candidate directory whose contents are to be listed
        candidate_dir_path = self._params.remote_dir_path / dir_path
        self._logger.info('Enumerating: %s with file_name_prefix: %s', candidate_dir_path, file_name_prefix)

        #Call the downstream API
        response = self._client.files_list_folder(candidate_dir_path.as_posix())

        #Output list placeholder
        files = []
        sizes = []

        if response.entries:
            #Log the response summary
            self._logger.info('Got %d files in: %s', len(response.entries), candidate_dir_path)

            #Extract the name of files satisfying the input criteria from the response entries.
            file_infos = [(dir_path / entry.name, entry.size) for entry in response.entries if entry.name.startswith(file_name_prefix)]

            files, sizes = zip(*file_infos)

        return files, sizes
