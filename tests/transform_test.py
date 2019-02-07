#Unittest
import unittest as ut

#Constants
from common import ut_constants

#Numpy
import numpy as np

#Transformation
from operation.transform import ImageDataTransformation

#Itertools
import itertools

#Logging
from common import logging

class TestParameters(ut.TestCase):
    def parse_parameters(self, param_dict, name, default = None):
        parameters = ImageDataTransformation.Parameters.parse(param_dict)
        value = getattr(parameters, name)
        input_value = default if param_dict.get(name) is None else param_dict.get(name)
        self.assertEqual(input_value, value)

    def parse_parameters_when_set_or_unset(self, name, value, default = None):
        #Property is unset
        self.parse_parameters({}, name, default = default)

        #Property is set
        self.parse_parameters({name: value}, name)

    def test_parse_without_samplewise_mean(self):
        param_dict = {}
        parameters = ImageDataTransformation.Parameters.parse(param_dict)
        self.assertFalse(parameters.samplewise_mean)

    def test_parse_samplewise_mean(self):
        self.parse_parameters_when_set_or_unset('samplewise_mean', True, default = False)

    def test_parse_samplewise_std_normalization(self):
        self.parse_parameters_when_set_or_unset('samplewise_std_normalization', True, default = False)

    def test_parse_horizontal_flip(self):
        self.parse_parameters_when_set_or_unset('horizontal_flip', True, default = False)

    def test_parse_horizontal_flip_prob(self):
        self.parse_parameters_when_set_or_unset('horizontal_flip_prob', .3, default = .5)

    def test_parse_featurewise_mean(self):
        self.parse_parameters_when_set_or_unset('featurewise_mean', True, default = False)

    def test_parse_featurewise_std_normalization(self):
        #Property name
        self.parse_parameters_when_set_or_unset('featurewise_std_normalization', True, default = False)

    def test_parse_rotation_range(self):
        #Property name
        self.parse_parameters_when_set_or_unset('rotation_range', 20)

    def test_parse_shear_range(self):
        #Property name
        self.parse_parameters_when_set_or_unset('shear_range', 10)

    def test_parse_zoom_range(self):
        #Property name
        self.parse_parameters_when_set_or_unset('zoom_range', .2)

    def test_parse_width_shift_range(self):
        #Property name
        self.parse_parameters_when_set_or_unset('width_shift_range', .2)

    def test_parse_height_shift_range(self):
        #Property name
        self.parse_parameters_when_set_or_unset('height_shift_range', .2)
    
class TestImageDataTransformation(ut.TestCase):
    def __init__(self, methodName = 'runTest'):
        super(TestImageDataTransformation, self).__init__(methodName)

        #Logging
        logging.initialize(ut_constants.LOGGING_CLASS)
        self._logger = logging.get_logger(__name__)

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

    def transform_affine(self, parameters):
        #Image dataset
        images = np.random.rand(5, 50, 50, 3)

        #Transformation object
        transformation = ImageDataTransformation(parameters = parameters)
        no_transformation = ImageDataTransformation()

        #Act
        transformed_images = transformation.transform(images)
        no_transformed_images = no_transformation.transform(images)

        #No transformation assertions
        np.testing.assert_array_almost_equal(
                        no_transformed_images,
                        images,
                        err_msg = "Unexpected transformation for parameters: {}".format(parameters))

        #Transformation assertions
        self.assertFalse(
                np.array_equal(transformed_images, images),
                "Expected rotation transformation for parameters: {}".format(parameters))

    def test_transform_affine(self):
        #Rotation transformation
        parameters = ImageDataTransformation.Parameters(rotation_range = 20)
        self.transform_affine(parameters)

        #Shear transformation
        parameters = ImageDataTransformation.Parameters(shear_range = 10)
        self.transform_affine(parameters)

        #Zoom transformation
        parameters = ImageDataTransformation.Parameters(zoom_range = 0.2)
        self.transform_affine(parameters)

        #Width translation
        parameters = ImageDataTransformation.Parameters(width_shift_range = 0.2)
        self.transform_affine(parameters)

        #Height translation
        parameters = ImageDataTransformation.Parameters(height_shift_range = 0.2)
        self.transform_affine(parameters)