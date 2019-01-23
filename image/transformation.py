"""It provides utilities for image transformation.
"""
#Image augmentation
from imgaug import augmenters as img_augmenters

#Numpy
import numpy as np

#Logging
from common import logging

class ImageDataTransformation:
    class Parameters(object):
        """It contains transformation parameters.
        """
        def __init__(
                self,
                samplewise_mean = False,
                samplewise_std_normalization = False,
                featurewise_mean = False,
                featurewise_std_normalization = False,
                horizontal_flip = False,
                horizontal_flip_prob = 0.5,
                rotation_range = None):
            """It initializes the input parameters.

            Arguments:
                samplewise_mean {boolean} -- It is a boolean value to indicate the transformer to center the image around the mean.
                samplewise_std_normalization {boolean} -- It is a boolean value to normalize the image using standard deviation. 
                featurewise_mean {boolean} -- It is a boolean flag to set input mean to zero over the dataset, feature wise.
                featurewise_std_normalization {boolean} -- It is a boolean flag to normalize the image using feature wise standard deviation calculated over the dataset.
                horizontal_flip {boolean} -- It is a boolean flag to enable horizontal flip transformation.
                horizontal_flip_prob {A floating point number} -- It indicates the changes of horizontal flip.
                rotation_range {A  number} -- It indicates the maximum amount of rotational transformations. 
            """
            #Input parameters
            self.samplewise_mean = samplewise_mean
            self.samplewise_std_normalization = samplewise_std_normalization
            self.featurewise_mean = featurewise_mean
            self.featurewise_std_normalization = featurewise_std_normalization
            self.horizontal_flip = horizontal_flip
            self.horizontal_flip_prob = horizontal_flip_prob
            self.rotation_range = rotation_range

        def __str__(self):
            return """Parameters::
                        samplewise_mean: {} samplewise_std_normalization: {}
                        featurewise_mean: {} featurewise_std_normalization: {}
                        horizontal_flip: {} horizontal_flip_prob: {}
                        rotation_range: {}""".format(
                                                                            self.samplewise_mean,
                                                                            self.samplewise_std_normalization,
                                                                            self.featurewise_mean,
                                                                            self.featurewise_std_normalization,
                                                                            self.horizontal_flip,
                                                                            self.horizontal_flip_prob,
                                                                            self.rotation_range)

        @classmethod
        def parse(cls, param_names):
            """It decodes the input list to identify parameters.
            
            Arguments:
                params {[string]} -- A list of transformation parameters.
            """
            parameters = cls()

            for name in param_names:
                #Set the class parameters
                getattr(parameters, name)
                setattr(parameters, name, True)

            return parameters

    def __init__(self, parameters = Parameters()):
        """It initializes the input parameters.

        Arguments:
            parameters {A ImageDataTransformation.Parameters object} -- It is a Parameters object that contains the transformation parameters.
        """
        #Input parameters
        self._parameters = parameters

        #Secondary parameters
        self._fit_called = False
        self._featurewise_mean = None
        self._featurewise_std = None

        #Logging
        self._logger = logging.get_logger(__name__)

        #Process flag dependencies
        if self._parameters.samplewise_std_normalization:
            self._parameters.samplewise_mean = True

        if self._parameters.featurewise_std_normalization:
            self._parameters.featurewise_mean = True

        #Augmentation list
        augmentations = []

        #Horizontal flip
        if self._parameters.horizontal_flip:
            #Flips images randomly with input probability.
            augmentations.append(
                    img_augmenters.Fliplr(self._parameters.horizontal_flip_prob))

        #Rotation range
        if self._parameters.rotation_range:
            #Rotate images randomly limited with maximum rotation limited by rotation range.
            augmentations.append(
                    img_augmenters.Affine(rotate = (-self._parameters.rotation_range, self._parameters.rotation_range)))
        
        #Creates augmentor with the list of augmentations
        self._augmenter = img_augmenters.Sequential(augmentations, random_order = True) if len(augmentations) > 0 else None

        #Log input parameters
        self._logger.info(
                        "Samplewise transformations:: samplewise_mean: %s samplewise_std_normalization: %s",
                        self._parameters.samplewise_mean,
                        self._parameters.samplewise_std_normalization)

        self._logger.info(
                        "Featurewise transformations:: featurewise_mean: %s featurewise_std_normalization: %s",
                        self._parameters.featurewise_mean,
                        self._parameters.featurewise_std_normalization)

        self._logger.info(
                        "Horizontal flip:: horizontal_flip: %s horizontal_flip_prob: %f",
                        self._parameters.horizontal_flip,
                        self._parameters.horizontal_flip_prob)

        self._logger.info(
                        "Rotation:: rotation_range: %s",
                        self._parameters.rotation_range)

    def fit(self, images):
        """It calculates statistics on the input dataset. These are used to perform transformation.
        
        Arguments:
            images {An numpy.array object} -- It is a 4-D numpy array containing image data.
        """
        #Apply sample wise corrections.
        corrected_images = self._apply_samplewise_before_featurewise_operations(images)

        #Calculate feature wise mean.
        self._featurewise_mean = corrected_images.mean(axis = 0)

        #Calculate feature wise standard deviation.
        self._featurewise_std = corrected_images.std(axis = 0)

        self._logger.info(
                        """fit::
                            input images: {}
                            featurewise_mean: {}
                            featurewise_std: {}
                        """.format(images.shape, self._featurewise_mean.shape, self._featurewise_std))

        #Set the flag to indicate dataset statistics are available.
        self._fit_called = True

    def transform(self, images):
        """It applies the transformations to the input images.
        
        Arguments:
            images {numpy.array} -- It is a 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = images

        if self._parameters.samplewise_mean:
            transformed_images = self._apply_samplewise_mean(transformed_images)

        if self._parameters.samplewise_std_normalization:
            transformed_images = self._apply_samplewise_std_normalization(transformed_images)

        if self._parameters.featurewise_mean and not self._fit_called:
            raise ValueError("Cannot calculate feature wise mean without calling fit()")

        if self._parameters.featurewise_std_normalization and not self._fit_called:
            raise ValueError("Cannot calculate feature wise standard deviation without calling fit()")

        if self._parameters.featurewise_mean:
            transformed_images = self._apply_featurewise_mean(transformed_images)

        if self._parameters.featurewise_std_normalization:
            transformed_images = self._apply_featurewise_std_normalization(transformed_images)

        if self._augmenter:
            transformed_images = self._apply_augmentations(transformed_images)

        return transformed_images

    def _apply_samplewise_mean(self, images):
        """It transform the sample to zero mean.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return np.asarray([image - np.mean(image) for image in images])

    def _apply_samplewise_std_normalization(self, images):
        """It normalizes an image using its standard deviation.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return np.asarray([image/np.std(image) for image in images])

    def _apply_samplewise_before_featurewise_operations(self, images):
        """It applies sample wise operations on the fit dataset before calculating feature wise statistics.
            When both sample wise mean and feature wise means are enabled, feature wise means must be calculated over
            sample wise mean adjusted inputs. If sample wise std normalization is also enabled, then it must be applied prior
            to feature wise calculations
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        transformed_images = images

        if self._parameters.samplewise_mean:
            transformed_images = self._apply_samplewise_mean(transformed_images)

        if self._parameters.samplewise_std_normalization:
            transformed_images = self._apply_samplewise_std_normalization(transformed_images)

        return transformed_images

    def _apply_featurewise_mean(self, images):
        """It sets input mean to zero over the dataset, feature wise.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return images - self._featurewise_mean

    def _apply_featurewise_std_normalization(self, images):
        """It normalizes the images with the feature wise standard deviation calculated over the dataset.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return images/self._featurewise_std

    def _apply_augmentations(self, images):
        """It applies image augmentations to the input images.
        
        Arguments:
            images {A numpy.array} -- A 4-D numpy array containing image data.

        Returns:
            A 4-D numpy array containing transformed image data.
        """
        return self._augmenter.augment_images(images)