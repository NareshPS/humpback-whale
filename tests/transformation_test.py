#Unittest
import unittest as ut

#Constants
from common import ut_constants

#Numpy
import numpy as np

#Transformation
from image.transformation import ImageDataTransformation

class TestImageDataTransformation(ut.TestCase):
    @staticmethod
    def get_examples():
        image_shape =  (1, 3, 3, 1)
        images = np.ones(shape = image_shape)
        
        images[:, :, 0, :] *= 3
        images[:, :, 1, :] *= 2
        images[:, :, 2, :] *= 1

        mean_transformed = np.ones(shape = image_shape)
        mean_transformed[:, :, 0, :] *= 1
        mean_transformed[:, :, 1, :] *= 0
        mean_transformed[:, :, 2, :] *= -1

        return images, mean_transformed

    def transform_samplewise_mean(self, samplewise_mean, images, results):
        #Transformation object
        transformation = ImageDataTransformation(samplewise_mean = samplewise_mean)

        #Transform
        transformed_images = transformation.transform(images)

        #Assert
        self.assertTrue(
            np.array_equal(transformed_images, results),
            "transformed_images: {} != expected: {}".format(transformed_images, results))

    def test_transform_with_samplewise_mean(self):
        #Get an image example.
        images, results = TestImageDataTransformation.get_examples()

        self.transform_samplewise_mean(
                True, #samplewise_mean
                images,
                results)

    def test_transform_without_samplewise_mean(self):
        #Get an image example.
        images, _ = TestImageDataTransformation.get_examples()

        self.transform_samplewise_mean(
                False, #samplewise_mean
                images,
                images)