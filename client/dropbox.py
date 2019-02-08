"""It allows access, upload and download of the dropbox data.
"""
#Dropbox
from dropbox import Dropbox
from dropbox.files import UploadSessionCursor as Dropbox_UploadSessionCursor
from dropbox.files import CommitInfo as Dropbox_CommitInfo
from dropbox.files import WriteMode as Dropbox_WriteMode

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

class DropboxConnection:
    def __init__(self, auth_token, remote_dir_path):
        #Required parameters
        self._remote_dir_path = remote_dir_path

        #Validation
        if not remote_dir_path:
            raise ValueError("No remote dir provided.")

        if not auth_token:
            raise ValueError("No authentication token provided")

        #Derived parameters
        self._client = Dropbox(auth_token)

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
            remote_file_path = (Path(constants.DROPBOX_APP_PATH_PREFIX) / self._remote_dir_path / source_file_name).as_posix()

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

    def download(self, remote_file_name):
        """It downloads the remote files from the dropbox.
        
        Arguments:
            remote_file_name {string} -- The name of the remote file.
        
        Raises:
            ValueError -- It raise value error for invalid file name.
        """
        #Validate parameters
        if not remote_file_name:
            raise ValueError("Invalid remote file name")

        #Remote file path
        remote_file_path = constants.DROPBOX_APP_PATH_PREFIX / self._remote_dir_path / remote_file_name

        #Get file size
        download_size = self._client.files_get_metadata(remote_file_path.as_posix()).size

        #Destination file name
        dest_file_name = remote_file_path.name

        self._logger.info('Preparing file download: %s with size: %d to: %s', remote_file_path, download_size, dest_file_name)

        #Download the file
        self._download_file(dest_file_name, remote_file_path, download_size)

        self._logger.info('Completed the file download: %s to: %s', remote_file_path, dest_file_name)

    def _download_file(self, dest_file_name, remote_file_path, download_size):
        """It downloads the remote files from the dropbox.
        
        Arguments:
            remote_file_path {A Path object} -- The path of the remote file.
            dest_file_name {string} -- The destination file name.
            download_size {int} -- The number of bytes to be downloaded.
        """
        #Download
        _, result = self._client.files_download(remote_file_path.as_posix())

        #Temporary dest_file_name
        tmp_dest_file_name = "{}.tmp".format(dest_file_name)

        with open(tmp_dest_file_name, 'wb') as handle:
            with tqdm(desc = 'Downloading: {}'.format(remote_file_path.as_posix()), total = download_size) as pbar:
                for bytes_read in result.iter_content(constants.DROPBOX_CHUNK_SIZE):
                    handle.write(bytes_read)

                    #Update the progress
                    pbar.update(len(bytes_read))

        if Path(tmp_dest_file_name).exists():
            rename(tmp_dest_file_name, dest_file_name)