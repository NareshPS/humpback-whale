"""It provides utilities for image transformation.
"""

#Numpy
import numpy as np

class ImageDataTransformation:

    def __init__(
            self,
            samplewise_mean = True):
        """It initializes the input parameters.
        """
        #Input parameters
        self._samplewise_mean = samplewise_mean

    def transform(self, images):
        """It applies the transformations to the input images.
        
        Arguments:
            images {numpy.array} -- It is a 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = images

        if self._samplewise_mean:
            transformed_images = [image - np.mean(image) for image in images]

        return np.asarray(transformed_images)