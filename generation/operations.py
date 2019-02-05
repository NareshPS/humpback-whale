"""Operations to process images.
"""

#Convolution
from scipy.signal import convolve as scipy_convolve

#Path manipulations
from os import path

#Data processing
from keras.preprocessing import image as keras_image
import numpy as np

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

def imload(source, images, shape):
    """It loads the images from the source location and returns a numpy object of images.
    
    Arguments:
        source {string} -- A string indicating the location of images.
        images {[string]} -- A list of string of image names.
        shape {(int, int)} -- A tuple to indicate the shape of the target image.
    
    Returns:
        numpy.ndarray -- A numpy array of image objects.
    """

    #Resolve image paths
    img_paths = [path.join(source, image) for image in images]

    #Load image files
    img_objs = [keras_image.load_img(img_path, target_size = shape) for img_path in img_paths]

    #Convert them to numpy arrays
    img_objs_arrays = np.asarray([keras_image.img_to_array(img_obj) for img_obj in img_objs])

    return img_objs_arrays

def normalize(img_objs):
    return img_objs/img_normalizing_factor

def random_transform(img_objs):
    None



