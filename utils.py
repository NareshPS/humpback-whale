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

def locate_img(source_loc, img_name):
    """Generates the full image path given a source location and image name.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_name {string} -- Name of the image.

    Returns:
        string -- Full path of image.
    """
    return path.join(source_loc, img_name)
   
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

def _load_images(source_loc, img_files):
    """It loads the list of input image files from source location.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_files {[string]} -- A list of image file names.

    Returns:
        [string] -- A list of images as numpy array.
    """
    return [cv2.imread(locate_img(source_loc, image), cv2.IMREAD_GRAYSCALE) for image in img_files]

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
        cv2.imwrite(locate_img(target_loc, img_file), (imgs[idx]*255).astype('uint8'))

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
        if n_files is not None and idx > n_files:
            break
        
        files.append(file.name)
    
    return files

def load_images_batch(source_loc, img_files, batch_size = 1, progress_bar = None):
    """It loads the list of input image files from source location in batches.

    Arguments:
        source_loc {string} -- Indicates the relative path of the image.
        img_files {[string]} -- A list of image file names.
        batch_size {int} -- Indicates the size of the batch of images to be fetches per call.

    Returns:
        [string] -- A list of images as numpy array.
    """
    for batch_id, img_batch in enumerate(batch(img_files, batch_size)):
        if progress_bar is not None:
            progress_bar.set_description("Processing batch: {batch_id}".format(batch_id = batch_id))

        imgs = _load_images(source_loc, img_batch)

        if progress_bar is not None:
            progress_bar.update(len(imgs))

        yield img_batch, imgs