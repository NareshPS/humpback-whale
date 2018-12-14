"""Utility methods to preprocess image datasets.
"""

#Basic imports
from os import path
import pathlib

#Imports for image load/unload/process
import cv2
import numpy as np
from skimage import transform

#Progress bar
from tqdm import tqdm

def locate_file(source_loc, file_name):
    """Generates the file path given a source location and file name.

    Arguments:
        source_loc {string} -- Indicates the relative path to the location of file.
        file_name {string} -- Name of the file.

    Returns:
        string -- Relative path of the file.
    """
    return path.join(source_loc, file_name)
   
def batch(iterable, batch_size = 1):
    """Creates a batch iterator to iterate over the iterable in batches.

    Arguments:
        iterable {[]} -- Iterable
        batch_size {int} -- Number on items yielded at once.

    Returns:
        [] -- An iterable containing number of items as indicated by batch_size.
    """
    count = len(iterable)
    for batch_idx in range(0, count, batch_size):
        yield iterable[batch_idx:min(batch_idx + batch_size, count)]

def _load_images(source_loc, img_names, img_shape):
    """It loads the list of input image files from source location.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_names {[string]} -- A list of image file names.
        img_shape {(int, int, int)} -- The target image shape.

    Returns:
        [string] -- A list of images as numpy array.
    """
    imgs = []
    for img in img_names:
        data = cv2.imread(locate_file(source_loc, img), cv2.IMREAD_GRAYSCALE)
        if img_shape is not None:
            data = np.reshape(data, img_shape)

        imgs.append(data)

    return imgs

def resize_images(images, target_size):
    """It loads the list of input image files from source location.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_files {[string]} -- A list of image file names.

    Returns:
        [string] -- A list of images as numpy array.
    """
    #transform.resize() changes the image data from [0..255] range to [0..1]
    return [transform.resize(image, target_size, anti_aliasing = True) for image in images]

def store_dataset(target_loc, img_files, imgs):
    """Stores the images to a target location.
    
    Arguments:
        target_loc {string} -- Target location to store images.
        img_files {[string]} -- List of image file names.
        imgs {[Numpy array]} -- List of numpy arrays containing the image data.
    """
    for idx, img_file in enumerate(img_files):
        cv2.imwrite(locate_file(target_loc, img_file), (imgs[idx]*255).astype('uint8'))

def split_dataset(input_set, split_ratio = 0.2):
    """Split the input set into two fragments. First fragment gets (1 - split_ratio) items. Second one gets the rest.
    
    Arguments:
        input_set {[string]} -- A list of input items.
    
    Keyword Arguments:
        split_ratio {float} -- A float between (0, 1]) to use as split marker. (default: {0.2})
    """
    n_images = len(input_set)
    split_marker = int(n_images*(1 - split_ratio))
    return input_set[:split_marker], input_set[split_marker:]

def list_files(source_loc, n_files = None):
    """Creates a list of files in a given directory.
    
    Arguments:
        source_loc {string} -- Source directory to scan.
    
    Returns:
        [string] -- A list of file names.
    """
    source_path = pathlib.Path(source_loc)
    files = []

    for idx, file in enumerate(source_path.iterdir()):
        if n_files is not None and idx == n_files:
            break
        
        files.append(file.name)
    
    return files

def load_images_batch(source_loc, img_files, img_shape, batch_size = 1, progress_bar = None):
    """It loads the list of input image files from source location in batches.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_files {[string]} -- A list of image file names.
        img_shape {(int, int, int)} -- The target image shape.
        batch_size {int} -- Indicates the size of the batch of images to be fetches per call.

    Returns:
        [string] -- A list of images as numpy array.
    """
    for batch_id, img_batch in enumerate(batch(img_files, batch_size)):
        if progress_bar is not None:
            progress_bar.set_description("Processing batch: {batch_id}".format(batch_id = batch_id))

        imgs = _load_images(source_loc, img_batch, img_shape)

        if progress_bar is not None:
            progress_bar.update(len(imgs))

        yield img_batch, imgs