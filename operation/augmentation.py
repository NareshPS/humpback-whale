"""It augments the input dataset with random and intentional transformations.
"""
#Transformation
from operation.transform import ImageDataTransformation

#Data processing
import numpy as np

#Logging
from common import logging

class ImageAugmentation(object):
    class Instance(object):
        """It executes a transformation instance.
        """
        def __init__(self, transformation_params, num_output_images = 1):
            """It initializes the augmentation parameters.

            Arguments:
                transformation_params {A ImageDataTransformation.Parameters object} -- The transformation parameters.
                num_output_images {int} -- The number of transformed images.
            """
            #Required parameters
            self._transformation_params = transformation_params
            self._num_output_images = num_output_images

            #Validation
            if self._transformation_params is None:
                raise ValueError('transformation_params must be valid')

            #Derived parameters
            self._transformer = ImageDataTransformation(self._transformation_params)

        def augmentations(self, image_objs):
            """It executes the tranformations to produce output images.

            Arguments:
                image_objs {A numpy array} -- The input image to be transformed.

            Returns:
                {A numpy array} - The list of output images.
            """
            #Output images placeholder
            output_image_objs = []

            #Generate the required number of the transformed images
            for _ in range(self._num_output_images):
                #Transform
                transformed_image_obj = self._transformer.transform(image_objs)

                #Add to the list of output images
                output_image_objs.append(transformed_image_obj)

            #Convert to numpy array
            augmented_images = np.concatenate(output_image_objs, axis = 0)

            return augmented_images

        def __str__(self):
                return """Parameters::
                            transformation_params: {}
                            num_output_images: {}""".format(
                                                            self._transformation_params,
                                                            self._num_output_images)

        def __len__(self):
            return self._num_output_images

    ### ImageAugmentation members ###    
    def __init__(self, instances):
        """It initializes the augmentation parameters

        Arguments:
            instances {[A ImageAugmentation.Instance object]} -- The list of augmentation instances.
        """
        #Required parameters
        self._instances = instances

        #Validation
        if self._instances is None:
            raise ValueError('instances must be a list of ImageAugmentation.Instance')

        #Logging
        self._logger = logging.get_logger(__name__)

        self._logger.info('Got %d augmentation instances', len(self._instances))

    def augmentations(self, image_obj):
        """It augments the input images.

            Arguments:
                image_obj {A numpy array} -- The input image to be transformed.

            Returns:
                {A numpy array} - The list of output images.
        """
        #Placeholder for augmented images.
        augmented_images = []

        #Execute augmentation instances
        for instance in self._instances:
            augmented_images.append(instance.augmentations(image_obj))

        #Convert to numpy array
        augmented_images = np.concatenate(augmented_images, axis = 0)

        return augmented_images

    def __len__(self):
        """It calculates the number of augmented outputs per image.
        """
        return sum([len(instance) for instance in self._instances])
