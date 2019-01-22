#Unittest
import unittest as ut

#Constants
from common import ut_constants

#Numpy
import numpy as np

#Data creation
import itertools

#Transformation
from image.transformation import ImageDataTransformation

class TestParameters(ut.TestCase):
    def test_parse_without_samplewise_mean(self):
        param_names = []
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertFalse(parameters.samplewise_mean)

    def test_parse_with_samplewise_mean(self):
        param_names = ['samplewise_mean']
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertTrue(parameters.samplewise_mean)
    
    def test_parse_without_samplewise_std_normalization(self):
        param_names = []
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertFalse(parameters.samplewise_std_normalization)

    def test_parse_with_samplewise_std_normalization(self):
        param_names = ['samplewise_std_normalization']
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertTrue(parameters.samplewise_std_normalization)

    def test_parse_without_horizontal_flip(self):
        param_names = []
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertFalse(parameters.horizontal_flip)

    def test_parse_with_horizontal_flip(self):
        param_names = ['horizontal_flip']
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertTrue(parameters.horizontal_flip)

    def test_parse_without_featurewise_mean(self):
        param_names = []
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertFalse(parameters.featurewise_mean)

    def test_parse_with_featurewise_mean(self):
        param_names = ['featurewise_mean']
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertTrue(parameters.featurewise_mean)

    def test_parse_without_featurewise_std_normalization(self):
        param_names = []
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertFalse(parameters.featurewise_std_normalization)

    def test_parse_with_featurewise_std_normalization(self):
        param_names = ['featurewise_std_normalization']
        parameters = ImageDataTransformation.Parameters.parse(param_names)
        self.assertTrue(parameters.featurewise_std_normalization)
    
class TestImageDataTransformation(ut.TestCase):
    @staticmethod
    def get_mean_transformed_examples():
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
        parameters = ImageDataTransformation.Parameters(samplewise_mean = samplewise_mean)
        transformation = ImageDataTransformation(parameters = parameters)

        #Transform
        transformed_images = transformation.transform(images)

        #Assert
        self.assertTrue(
            np.array_equal(transformed_images, results),
            "transformed_images: {} != expected: {}".format(transformed_images, results))

    def test_transform_with_samplewise_mean(self):
        #Get an image example.
        images, results = TestImageDataTransformation.get_mean_transformed_examples()

        self.transform_samplewise_mean(
                True, #samplewise_mean
                images,
                results)

    def test_transform_without_samplewise_mean(self):
        #Get an image example.
        images, _ = TestImageDataTransformation.get_mean_transformed_examples()

        self.transform_samplewise_mean(
                False, #samplewise_mean
                images,
                images)

    def transform_samplewise_std_normalization(self, samplewise_std_normalization, images):
        #Transformation object
        parameters = ImageDataTransformation.Parameters(samplewise_std_normalization = samplewise_std_normalization)
        transformation = ImageDataTransformation(parameters = parameters)

        #Transform
        transformed_images = transformation.transform(images)

        #Compute standard deviation
        standard_deviations = np.std(transformed_images, axis = (1, 2, 3))

        #Assert
        if samplewise_std_normalization:
            self.assertAlmostEqual(
                np.sum(standard_deviations), 
                2.,
                places = 2,
                msg = "standard_deviations: {} != expected: 2.".format(standard_deviations))
        else:
            self.assertNotEqual(np.sum(standard_deviations), 2.)

    def test_transform_with_samplewise_std_normalization(self):
        images = np.random.rand(2, 5, 5, 3)

        self.transform_samplewise_std_normalization(
                True, #samplewise_std_normalization
                images)

    def test_transform_without_samplewise_std_normalization(self):
        images = np.random.rand(2, 5, 5, 3)

        self.transform_samplewise_std_normalization(
                False, #samplewise_std_normalization
                images)

    @staticmethod
    def get_horizontal_flip_examples(flipped = False):
        #Create image data
        pixel_values = range(1, 6)
        image_slice = list(itertools.chain.from_iterable(itertools.repeat(value, 3) for value in pixel_values))
        image_slice = list(reversed(image_slice)) if flipped else image_slice
        image_slice = np.asarray(image_slice).reshape(5, 3)
        image = [image_slice*multiplier for multiplier in range(1, 6)]
        images = np.asarray(image).reshape(1, 5, 5, 3)
        
        return images

    def transform_horizontal_flip(self, horizontal_flip, images, results):
        #Transformation object
        parameters = ImageDataTransformation.Parameters(horizontal_flip = horizontal_flip, horizontal_flip_prob = 1.0)
        transformation = ImageDataTransformation(parameters = parameters)

        #Transform
        transformed_images = transformation.transform(images)

        #Assert
        self.assertTrue(
                np.array_equal(transformed_images, results),
                "transformed_images: {} != expected: {}".format(transformed_images, results))

    def test_transform_with_horizontal_flip(self):
        images = TestImageDataTransformation.get_horizontal_flip_examples(flipped = False)
        flipped_images = TestImageDataTransformation.get_horizontal_flip_examples(flipped = True)

        self.transform_horizontal_flip(
                    True, #Horizontal flip
                    images,
                    flipped_images)

    def test_transform_without_horizontal_flip(self):
        images = TestImageDataTransformation.get_horizontal_flip_examples(flipped = False)

        self.transform_horizontal_flip(
                    False, #Horizontal flip
                    images,
                    images)

    @staticmethod
    def get_featurewise_mean_examples():
        #Create image data
        images = np.ones((5, 5, 5, 3))

        for image_id in range(5):
            images[image_id] *= image_id

        return images

    def transform_featurewise_mean(self, featurewise_mean, images, result):
        #Transformation object
        parameters = ImageDataTransformation.Parameters(featurewise_mean = featurewise_mean)
        transformation = ImageDataTransformation(parameters = parameters)

        #Fit and perform transformation
        transformation.fit(images)
        transformed_images = transformation.transform(images)
        sum_image = transformed_images.sum(axis = 0)

        #Assert
        self.assertTrue(
                np.array_equal(sum_image, result),
                "Sum images: {} expected: {}".format(sum_image, result))

    def test_transform_with_featurewise_mean(self):
        #Arrange
        images = TestImageDataTransformation.get_featurewise_mean_examples()
        result = np.zeros(images.shape[1:])

        #Act
        self.transform_featurewise_mean(
                    True, #Enable feature wise mean
                    images,
                    result)

    def test_transform_without_featurewise_mean(self):
        #Arrange
        images = TestImageDataTransformation.get_featurewise_mean_examples()
        result = np.ones(images.shape[1:]) * 10 #Initialize with the sum of all features

        #Act
        self.transform_featurewise_mean(
                    False, #Disable feature wise mean
                    images,
                    result)

    def test_transform_with_featurewise_mean_fit_not_called(self):
        #Arrange
        images = TestImageDataTransformation.get_featurewise_mean_examples()

        #Transformation object
        parameters = ImageDataTransformation.Parameters(featurewise_mean = True)
        transformation = ImageDataTransformation(parameters = parameters)

        with self.assertRaises(ValueError):
            transformation.transform(images)

    @staticmethod
    def get_featurewise_std_normalization_examples(standard_deviation):
        #Create image data
        images = np.random.normal(
                                0, #Mean
                                standard_deviation, #Standard deviation
                                (5, 5, 5, 3))

        return images

    def transform_featurewise_std_normalization(self, featurewise_std_normalization, images, result):
        #Transformation object
        parameters = ImageDataTransformation.Parameters(featurewise_std_normalization = featurewise_std_normalization)
        transformation = ImageDataTransformation(parameters = parameters)

        #Fit and perform transformation
        transformation.fit(images)
        transformed_images = transformation.transform(images)
        transformed_std = transformed_images.std(axis = 0)

        #Assert
        self.assertTrue(np.allclose(transformed_std, result))

    def test_transform_with_featurewise_std_normalization(self):
        #Arrange
        standard_deviation = 0.1
        images = TestImageDataTransformation.get_featurewise_std_normalization_examples(standard_deviation)
        result = np.ones((5, 5, 3), dtype = float)

        #Act
        self.transform_featurewise_std_normalization(
                    True, #Enable feature wise standard deviation normalization
                    images,
                    result)

    def test_transform_without_featurewise_std_normalization(self):
        #Arrange
        standard_deviation = 0.1
        images = TestImageDataTransformation.get_featurewise_std_normalization_examples(standard_deviation)
        result = images.std(axis = 0)

        #Act
        self.transform_featurewise_std_normalization(
                    False, #Disable feature wise standard deviation normalization
                    images,
                    result)