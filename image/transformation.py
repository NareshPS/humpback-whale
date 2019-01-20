"""It provides utilities for image transformation.
"""

#Numpy
import numpy as np

class ImageDataTransformation:

    def __init__(
            self,
            samplewise_mean = False,
            samplewise_std_normalization = False):
        """It initializes the input parameters.

        Arguments:
            samplewise_mean {boolean} -- It is a boolean value to indicate the transformer to center the image around the mean.
            samplewise_std_normalization {boolean} -- It is a boolean value to indicate the transformer to normalize the image using standard deviation. 
        """
        #Input parameters
        self._samplewise_mean = samplewise_mean
        self._samplewise_std_normalization = samplewise_std_normalization

        #Process flag dependencies
        if self._samplewise_std_normalization:
            self._samplewise_mean = True

    def transform(self, images):
        """It applies the transformations to the input images.
        
        Arguments:
            images {numpy.array} -- It is a 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = images

        if self._samplewise_mean:
            transformed_images = [image - np.mean(image) for image in transformed_images]

        if self._samplewise_std_normalization:
            transformed_images = self._transform_samplewise_std_normalization(transformed_images)

        return np.asarray(transformed_images)

    def _transform_samplewise_std_normalization(self, images):
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