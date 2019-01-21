"""It provides utilities for image transformation.
"""
#Image augmentation
from imgaug import augmenters as img_augmenters

#Numpy
import numpy as np

#Logging
from common import logging

class ImageDataTransformation:

    def __init__(
            self,
            samplewise_mean = False,
            samplewise_std_normalization = False,
            horizontal_flip = False,
            horizontal_flip_prob = 0.5):
        """It initializes the input parameters.

        Arguments:
            samplewise_mean {boolean} -- It is a boolean value to indicate the transformer to center the image around the mean.
            samplewise_std_normalization {boolean} -- It is a boolean value to indicate the transformer to normalize the image using standard deviation. 
            horizontal_flip {boolean} -- It is a boolean flag to enable horizontal flip transformation.
            horizontal_flip_prob {A floating point number} -- It indicates the changes of horizontal flip.
        """
        #Input parameters
        self._samplewise_mean = samplewise_mean
        self._samplewise_std_normalization = samplewise_std_normalization
        self._horizontal_flip = horizontal_flip
        self._horizontal_flip_prob = horizontal_flip_prob

        #Logging
        self._logger = logging.get_logger(__name__)

        #Process flag dependencies
        if self._samplewise_std_normalization:
            self._samplewise_mean = True

        #Augmentation list
        augmentations = []

        #Horizontal flip
        if self._horizontal_flip:
            #Flips images randomly with input probability.
            augmentations.append(img_augmenters.Fliplr(self._horizontal_flip_prob))
        
        #Creates augmentor with the list of augmentations
        self._augmenter = img_augmenters.Sequential(augmentations, random_order = True) if len(augmentations) > 0 else None

        #Log input parameters
        self._logger.info(
                        "Samplewise transformations:: samplewise_mean: %s samplewise_std_normalization: %s",
                        self._samplewise_mean,
                        self._samplewise_std_normalization)

        self._logger.info(
                        "Horizontal flip:: horizontal_flip: %s horizontal_flip_prob: %f",
                        self._horizontal_flip,
                        self._horizontal_flip_prob)

    def fit(self, images):
        """It calculates statistics on the input dataset. These are used to perform transformation.
        
        Arguments:
            images {An numpy.array object} -- It is a 4-D numpy array containing image data.
        """
        pass

    def transform(self, images):
        """It applies the transformations to the input images.
        
        Arguments:
            images {numpy.array} -- It is a 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = images

        if self._samplewise_mean:
            transformed_images = np.asarray([image - np.mean(image) for image in transformed_images])

        if self._samplewise_std_normalization:
            transformed_images = self._apply_samplewise_std_normalization(transformed_images)

        if self._augmenter:
            transformed_images = self._apply_augmentations(transformed_images)

        return transformed_images

    def _apply_samplewise_std_normalization(self, images):
        """It normalizes an image using its standard deviation.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = []

        for image in images:
            transformed_image = image/np.std(image)
            transformed_images.append(transformed_image)

        return np.asarray(transformed_images)

    def _apply_augmentations(self, images):
        """It applies image augmentations to the input images.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return self._augmenter.augment_images(images)