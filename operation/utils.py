"""Operations to process images.
"""

#Convolution
from scipy.signal import convolve as scipy_convolve

#Path manipulations
from os import path

#Image operations
from PIL import Image
import numpy as np

#Constants
from common import constants

img_normalizing_factor = 255.

def convolve(images, kernel, mode = 'same'):
    """It convolves the kernel on input images.
    
    Arguments:
        images {[numpy.array]} -- Images as numpy matrices.
        kernel {numpy.array} -- A numpy array to convolute.
        mode {string} -- A string indicating the mode to apply convolution. 
            The values accepted by scipy.signal.convolve() are valid.
    
    Returns:
        {[np.array]} -- Convoluted images.
    """
    input_images = np.asarray(images).astype(np.int16)
    convoluted_images = np.asarray([scipy_convolve(image, kernel, mode='same') for image in input_images])
    convoluted_images[convoluted_images > 255] = 255
    convoluted_images[convoluted_images < 0] = 0

    return convoluted_images.astype(np.uint8)

def imload(source, images, shape = None):
    """It loads the images from the source location and returns a numpy object of images.
    
    Arguments:
        source {string} -- A string indicating the location of images.
        images {[string]} -- A list of string of image names.
        shape {(int, int)} -- A (W, H) tuple to indicate the shape of the target image.
    
    Returns:
        numpy.ndarray -- A numpy array of image objects.
    """
    #Resolve image paths
    img_paths = [path.join(source, image) for image in images]

    #Load image files
    img_objs = [Image.open(img_path) for img_path in img_paths]

    #Convert to RGB
    color_mode = constants.PIL_IMAGE_RGB_COLOR_MODE
    img_objs = list(map(lambda x : x if x.mode == color_mode else x.convert(color_mode), img_objs))

    #Resize images
    if shape is not None:
        img_objs = [img_obj.resize(shape) for img_obj in img_objs]

    #Convert them to numpy arrays
    img_objs_arrays = np.asarray([np.asarray(img_obj, dtype = 'uint8') for img_obj in img_objs])

    return img_objs_arrays

def imwrite(destination, images_with_name):
    """It write the images to the destination location.
    
    Arguments:
        destination {A Path object} -- The destination location of the image.
        images {{string, A numpy array}} -- A dictionary of image name to its value.
    """
    #Iterate over all the images.
    for image_name, image in images_with_name.items():
        #PIL image
        pil_image = Image.fromarray(image.astype(np.uint8))

        #Destination path
        image_path = destination / image_name

        #Save the image
        pil_image.save(image_path)

def normalize(img_objs):
    return img_objs/img_normalizing_factor

def random_transform(img_objs):
    None



