#Unittest
import unittest as ut
from unittest.mock import MagicMock

#Constants
from common import constants
from common import ut_constants

#Image augmentation
from operation.transform import ImageDataTransformation
from operation.augmentation import ImageAugmentation

#Data processing
import numpy as np

hflip_tranformation_params = ImageDataTransformation.Parameters(horizontal_flip = True)
hflip_num_output_images = 1

shear_transformation_params = ImageDataTransformation.Parameters(shear_range = 10)
shear_num_output_images = 3

class TestImageAugmentationInstance(ut.TestCase):
    def test_init_invalid_params(self):
        with self.assertRaises(ValueError):
            _ = ImageAugmentation.Instance(None, 1)

    def test_init(self):
        #Arrange
        augmentation_instance = ImageAugmentation.Instance(hflip_tranformation_params, hflip_num_output_images)

        #Assert
        self.assertEqual(augmentation_instance._transformation_params, hflip_tranformation_params)
        self.assertEqual(augmentation_instance._num_output_images, hflip_num_output_images)
        self.assertEqual(len(augmentation_instance), hflip_num_output_images)
        self.assertIsNotNone(augmentation_instance._transformer)

    def test_augmentations(self):
        #Arrange inputs
        shape = (1, 5, 5, 3)
        image_obj = np.ones(shape)

        #Arrange augmentation instance
        augmentation_instance = ImageAugmentation.Instance(shear_transformation_params, shear_num_output_images)
        augmentation_instance._transformer = MagicMock()
        augmentation_instance._transformer.transform.return_value = image_obj

        #Act
        augmented_img_objs = augmentation_instance.augmentations(image_obj)

        #Assert
        self.assertEqual(shear_num_output_images, augmentation_instance._transformer.transform.call_count)
        self.assertEqual(shear_num_output_images, len(augmented_img_objs))
        self.assertEqual(len(augmentation_instance), shear_num_output_images)

        for obj in augmented_img_objs:
            np.testing.assert_array_almost_equal(np.squeeze(image_obj), obj)

class TestImageAugmentation(ut.TestCase):
    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            _ = ImageAugmentation(None)

    def test_init(self):
        #Arrange
        augmentation_instances = [
                                    ImageAugmentation.Instance(hflip_tranformation_params, hflip_num_output_images),
                                    ImageAugmentation.Instance(shear_transformation_params, shear_num_output_images)
                                ]
        image_augmentation = ImageAugmentation(augmentation_instances)

        #Assert
        self.assertEqual(len(augmentation_instances), len(image_augmentation._instances))  
        self.assertEqual(len(image_augmentation), hflip_num_output_images + shear_num_output_images)

    def test_augmentations(self):
        #Arrange inputs
        shape = (1, 50, 50, 3)
        image_obj = np.ones(shape)

        #Arrange augmentations
        augmentation_instances = [
                                    ImageAugmentation.Instance(hflip_tranformation_params, hflip_num_output_images),
                                    ImageAugmentation.Instance(shear_transformation_params, shear_num_output_images)
                                ]
        image_augmentation = ImageAugmentation(augmentation_instances)

        #Arrange augmentation mocks
        for instance in image_augmentation._instances:
            instance.augmentations = MagicMock()
            instance.augmentations.return_value = np.concatenate([image_obj, image_obj], axis = 0) #Add two images

        #Act
        augmented_objs = image_augmentation.augmentations(image_obj)

        print(augmented_objs.shape)

        #Assert
        self.assertEqual(4, augmented_objs.shape[0])
        for instance in image_augmentation._instances:
            instance.augmentations.assert_called_once()